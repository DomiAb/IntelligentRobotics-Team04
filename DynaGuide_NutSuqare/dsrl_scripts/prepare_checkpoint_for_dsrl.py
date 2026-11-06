import argparse
from pathlib import Path
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align checkpoint EMA state dict with base nets architecture."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the input checkpoint file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.dataset)

    # 1. Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_dict = checkpoint["model"]

    # 2. Overwrite the mismatched EMA dict with the valid base model architecture
    if "nets" in model_dict:
        print("Aligning keys: Copying base 'nets' state_dict over to the 'ema' slot...")
        model_dict["ema"] = model_dict["nets"]

        # 3. Save out the clean checkpoint dynamically (appends '_dsrl' before extension)
        output_path = ckpt_path.with_name(f"{ckpt_path.stem}_dsrl{ckpt_path.suffix}")
        torch.save(checkpoint, output_path)
        print(f"Success! Saved aligned checkpoint to: {output_path}")
    else:
        print("Error: Could not find 'nets' inside the checkpoint's model field.")


if __name__ == "__main__":
    main()