"""
Use this script to view all of the single frame guidance conditions.
python show_guidance_images.py \
    --dataset dataset/square_splits/square_pos_examples.hdf5 \
    --output evaluation/pos_images_extracted \
    --camera agentview_image
"""

import argparse
import os
import h5py
import cv2

def main():
    parser = argparse.ArgumentParser(description="Extract embedded visual frames directly from an HDF5 dataset as PNG images.")
    parser.add_argument("--dataset", required=True, help="Path to the input HDF5 dataset file")
    parser.add_argument("--output", required=True, help="Target destination directory for the generated PNG files")
    parser.add_argument("--camera", default="agentview_image", help="The observation key name for the image dataset stream (default: agentview_image)")
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"Source file not found: {args.dataset}")

    os.makedirs(args.output, exist_ok=True)

    with h5py.File(args.dataset, "r") as f:
        if "data" not in f:
            raise KeyError("Invalid tracking schema: missing top-level 'data' group.")
            
        demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[-1]) if "_" in x else 0)
        
        print(f"Extracting visual snapshot targets from {len(demos)} demonstrations...")
        extracted_count = 0

        for idx, demo in enumerate(demos):
            demo_grp = f[f"data/{demo}"]
            
            # Form standard structural dataset paths
            img_path = f"obs/{args.camera}"
            
            if img_path in demo_grp:
                # Extract array frame at t=0
                img_data = demo_grp[img_path][0]
            elif args.camera in demo_grp:
                # Fallback layout check if stored without observation prefix
                img_data = demo_grp[args.camera][0]
            else:
                continue

            # Convert matrix layout from RGB to BGR for accurate OpenCV representation
            img_bgr = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
            
            output_filename = os.path.join(args.output, f"snapshot_{idx:04d}.png")
            cv2.imwrite(output_filename, img_bgr)
            extracted_count += 1

    print(f"Successfully serialized {extracted_count} visual files inside: {args.output}/")

if __name__ == "__main__":
    main()