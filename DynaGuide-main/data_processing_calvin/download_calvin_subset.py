"""Download a consistent fraction of a CALVIN split (e.g. 1/4 of task_D_D) without fetching the whole zip.

The CALVIN zips are served with HTTP range support, so we
  1. read only the zip's central directory (~20 s) to learn where every file lives,
  2. pick whole continuous recordings (from ep_start_end_ids.npy) until the requested
     fraction of frames is reached; the last recording is cut to hit the target exactly,
  3. download just those episode_XXXXXXX.npz frames in large contiguous byte ranges,
  4. rewrite ep_start_end_ids.npy, ep_lens.npy and the language annotations so they only
     reference frames that exist. statistics.yaml, scene_info.npy and .hydra/ are copied as-is.

The download is resumable: rerun the same command and finished files are skipped.

Usage:
    python download_calvin_subset.py --out /path/to/calvin_data --fraction 0.25
    python download_calvin_subset.py --out ... --fraction 0.25 --val-fraction 0.1 --all-lang

Requires: pip install remotezip   (pulls in requests)
"""

import argparse
import io
import json
import os
import random
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from remotezip import RemoteZip

DEFAULT_URL = "http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
SPLITS = ("training", "validation")
CHUNK_BYTES = 128 * 1024 * 1024  # max size of a single range request


# ----------------------------------------------------------------------------- remote zip access

class RemoteArchive:
    def __init__(self, url):
        self.url = url
        print(f"Reading central directory of {url} ...")
        t = time.time()
        with RemoteZip(url) as z:
            infos = [i for i in z.infolist() if not i.filename.endswith("/")]
        self.by_name = {i.filename: i for i in infos}
        # The byte extent of an entry runs from its local header to the next entry's local header.
        offsets = sorted(i.header_offset for i in infos)
        total = int(requests.head(url, allow_redirects=True).headers["Content-Length"])
        nxt = dict(zip(offsets, offsets[1:] + [total]))
        self.extent = {i.filename: (i.header_offset, nxt[i.header_offset]) for i in infos}
        print(f"  {len(infos)} files, {total / 1e9:.1f} GB ({time.time() - t:.0f} s)")
        self._local = threading.local()

    def _session(self):
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def fetch_range(self, start, end, retries=5):
        """Bytes [start, end) of the archive."""
        for attempt in range(retries):
            try:
                r = self._session().get(self.url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
                if r.status_code != 206:
                    raise IOError(f"server answered {r.status_code} to a range request")
                if len(r.content) != end - start:
                    raise IOError(f"short read: {len(r.content)} of {end - start} bytes")
                return r.content
            except (requests.RequestException, IOError) as e:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  range {start}-{end} failed ({e}); retrying in {wait} s")
                time.sleep(wait)

    def decode_entry(self, name, raw):
        """Extract one file's content from the raw bytes of its extent (local header + data)."""
        info = self.by_name[name]
        if raw[:4] != b"PK\x03\x04":
            raise IOError(f"{name}: no local file header at expected offset")
        n = int.from_bytes(raw[26:28], "little")
        m = int.from_bytes(raw[28:30], "little")
        data = raw[30 + n + m: 30 + n + m + info.compress_size]
        if info.compress_type == 0:
            out = data
        elif info.compress_type == 8:
            out = zlib.decompress(data, -15)
        else:
            raise IOError(f"{name}: unsupported compression {info.compress_type}")
        if zlib.crc32(out) != info.CRC or len(out) != info.file_size:
            raise IOError(f"{name}: CRC/size mismatch")
        return out

    def read(self, name):
        return self.decode_entry(name, self.fetch_range(*self.extent[name]))


# ----------------------------------------------------------------------------- selection

def select_frames(ep_ids, fraction, seed):
    """Choose whole recordings in random order until `fraction` of all frames is covered;
    the last chosen recording is truncated (keeping its start) to hit the target exactly.
    Returns a list of (start, end) inclusive ranges, sorted by start."""
    lens = ep_ids[:, 1] - ep_ids[:, 0] + 1
    target = int(round(fraction * lens.sum()))
    order = list(range(len(ep_ids)))
    random.Random(seed).shuffle(order)
    chosen, have = [], 0
    for idx in order:
        if have >= target:
            break
        start, end = map(int, ep_ids[idx])
        take = min(end - start + 1, target - have)
        chosen.append((start, start + take - 1))
        have += take
    return sorted(chosen)


def filter_lang_ann(ann, ranges):
    """Keep only annotations whose (start, end) window lies completely inside one kept range."""
    indx = ann["info"]["indx"]
    keep = [i for i, (s, e) in enumerate(indx) if any(a <= s and e <= b for a, b in ranges)]
    out = {"info": dict(ann["info"]), "language": dict(ann["language"])}
    out["info"]["indx"] = [indx[i] for i in keep]
    for key, val in ann["language"].items():
        if isinstance(val, np.ndarray) and len(val) == len(indx):
            out["language"][key] = val[keep]
        elif isinstance(val, list) and len(val) == len(indx):
            out["language"][key] = [val[i] for i in keep]
    return out, len(keep), len(indx)


# ----------------------------------------------------------------------------- download

def write_atomic(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(content)
    os.replace(tmp, path)


def plan_chunks(archive, names):
    """Group files that sit next to each other in the zip into few large range requests."""
    spans = sorted((archive.extent[n][0], archive.extent[n][1], n) for n in names)
    chunks, cur = [], []
    for s, e, n in spans:
        if cur and (s != cur[-1][1] or e - cur[0][0] > CHUNK_BYTES):
            chunks.append(cur)
            cur = []
        cur.append((s, e, n))
    if cur:
        chunks.append(cur)
    return chunks


def download_files(archive, names, out_dir, workers):
    todo = [n for n in names
            if not (os.path.exists(os.path.join(out_dir, n))
                    and os.path.getsize(os.path.join(out_dir, n)) == archive.by_name[n].file_size)]
    if not todo:
        print("  all frames already present")
        return
    chunks = plan_chunks(archive, todo)
    total = sum(c[-1][1] - c[0][0] for c in chunks)
    print(f"  {len(todo)} files to fetch ({len(names) - len(todo)} already done), "
          f"{total / 1e9:.2f} GB in {len(chunks)} requests, {workers} workers")

    done_bytes, lock, t0 = [0], threading.Lock(), time.time()

    def work(chunk):
        base = chunk[0][0]
        raw = archive.fetch_range(base, chunk[-1][1])
        for s, e, n in chunk:
            write_atomic(os.path.join(out_dir, n), archive.decode_entry(n, raw[s - base:e - base]))
        with lock:
            done_bytes[0] += len(raw)
            rate = done_bytes[0] / max(time.time() - t0, 1e-6)
            eta = (total - done_bytes[0]) / max(rate, 1)
            print(f"  {done_bytes[0] / 1e9:7.2f} / {total / 1e9:.2f} GB  "
                  f"{rate / 1e6:6.1f} MB/s  ETA {eta / 60:5.1f} min", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(work, c) for c in chunks]):
            fut.result()


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output directory (task_D_D/... is created inside)")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--fraction", type=float, default=0.25, help="fraction of training frames")
    p.add_argument("--val-fraction", type=float, default=None, help="fraction of validation frames (default: --fraction)")
    p.add_argument("--seed", type=int, default=0, help="which recordings get picked")
    p.add_argument("--all-lang", action="store_true",
                   help="also keep the lang_<model>/ folders with precomputed embeddings of other language models")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="only print the selection and download size")
    args = p.parse_args()
    fractions = {"training": args.fraction,
                 "validation": args.fraction if args.val_fraction is None else args.val_fraction}

    archive = RemoteArchive(args.url)
    root = next(n.split("/")[0] for n in archive.by_name if "/training/" in n)  # e.g. "task_D_D"
    summary = {"url": args.url, "seed": args.seed, "splits": {}}

    for split in SPLITS:
        prefix = f"{root}/{split}/"
        print(f"\n== {split}: fraction {fractions[split]}")
        ep_ids = np.load(io.BytesIO(archive.read(prefix + "ep_start_end_ids.npy")))
        ranges = select_frames(ep_ids, fractions[split], args.seed)
        n_frames = sum(e - s + 1 for s, e in ranges)
        print(f"  keeping {len(ranges)} of {len(ep_ids)} recordings, "
              f"{n_frames} of {int((ep_ids[:, 1] - ep_ids[:, 0] + 1).sum())} frames")

        frames = [f"{prefix}episode_{i:07d}.npz" for s, e in ranges for i in range(s, e + 1)]
        missing = [f for f in frames if f not in archive.by_name]
        if missing:
            raise RuntimeError(f"{len(missing)} expected frames missing in archive, e.g. {missing[:3]}")
        size = sum(archive.by_name[f].compress_size for f in frames)
        print(f"  download size: {size / 1e9:.2f} GB")
        summary["splits"][split] = {"fraction": fractions[split], "ranges": ranges, "frames": n_frames,
                                    "bytes": size}
        if args.dry_run:
            continue

        # Small metadata, copied unchanged (statistics.yaml keeps full-dataset normalization stats).
        lang_dirs = [d for d in {n[len(prefix):].split("/")[0] for n in archive.by_name if n.startswith(prefix)}
                     if d == "lang_annotations" or (args.all_lang and d.startswith("lang_"))]
        for name in archive.by_name:
            if not name.startswith(prefix) or name.endswith(".npz"):
                continue
            rel = name[len(prefix):]
            top = rel.split("/")[0]
            if top.startswith("lang_") and top not in lang_dirs:
                continue
            if rel in ("ep_start_end_ids.npy", "ep_lens.npy") or rel.endswith("auto_lang_ann.npy"):
                continue  # rewritten below
            write_atomic(os.path.join(args.out, name), archive.read(name))

        # Rewritten metadata.
        new_ids = np.array(ranges, dtype=ep_ids.dtype)
        split_dir = os.path.join(args.out, prefix)
        os.makedirs(split_dir, exist_ok=True)
        np.save(os.path.join(split_dir, "ep_start_end_ids.npy"), new_ids)
        np.save(os.path.join(split_dir, "ep_lens.npy"), new_ids[:, 1] - new_ids[:, 0] + 1)
        for d in lang_dirs:
            name = f"{prefix}{d}/auto_lang_ann.npy"
            if name not in archive.by_name:
                continue
            ann = np.load(io.BytesIO(archive.read(name)), allow_pickle=True).item()
            ann, kept, total = filter_lang_ann(ann, ranges)
            os.makedirs(os.path.join(split_dir, d), exist_ok=True)
            np.save(os.path.join(split_dir, d, "auto_lang_ann.npy"), ann, allow_pickle=True)
            print(f"  {d}: kept {kept} of {total} language annotations")

        download_files(archive, frames, args.out, args.workers)

    total = sum(s["bytes"] for s in summary["splits"].values())
    print(f"\nTotal download: {total / 1e9:.2f} GB")
    if not args.dry_run:
        with open(os.path.join(args.out, root, "subset_info.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Done. Data in {os.path.join(args.out, root)}")


if __name__ == "__main__":
    main()
