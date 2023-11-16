import argparse
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import pandas as pd
from transformers import pipeline

GEN_KWARGS = {
    'max_length': 75,
    'min_length': 5,
    'do_sample': False,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing script for FaceSynthetics")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="Path to FaceSynthetics dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name for the dataset (metadata)"
    )

    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="The first data in the split (metadata)"
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=100000,
        help="The number of data in the dataset, default to the full 100,000 dataset"
    )

    return parser.parse_args()

def main(args):
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device=0)

    text = []
    for i in tqdm(range(args.start_id, args.start_id + args.num_data)):
        image_path = os.path.join(args.dataset_path, f"{i:06d}.png")
        assert os.path.exists(image_path), f"Missing data: {image_path}"

        caption = captioner(image_path, generate_kwargs=GEN_KWARGS)[0]['generated_text']
        text.append(caption)
    
    metadata_path = os.path.join(args.dataset_path, f'metadata_{args.split}.jsonl')
    df = pd.DataFrame(data={"id": range(args.start_id, args.start_id + args.num_data), "text": text})
    df.to_json(metadata_path, orient='records', lines=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
