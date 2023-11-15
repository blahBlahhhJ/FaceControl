import argparse
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import pandas as pd

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
        "--num_data",
        type=int,
        default=100000,
        help="The number of data in the dataset, default to the full 100,000 dataset"
    )

    return parser.parse_args()

def draw(image, landmarks, radius=2.5, color="white"):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def main(args):
    text = []
    for i in tqdm(range(args.num_data)):
        caption_path = os.path.join(args.dataset_path, f"{i:06d}.txt")
        landmark_path = os.path.join(args.dataset_path, f"{i:06d}_ldmks.txt")
        out_path = os.path.join(args.dataset_path, f"{i:06d}_cond.png")

        assert os.path.exists(caption_path), f"Missing caption data: {caption_path}"
        assert os.path.exists(landmark_path), f"Missing landmark data: {landmark_path}"

        with open(caption_path, 'r') as f:
            text.append(f.read())

        with open(landmark_path, 'r') as f:
            lines = f.readlines()

        landmarks = []
        for j in range(68):
            l = lines[j].split(' ')
            landmarks.append((float(l[0].strip()), float(l[1].strip())))

        cond_img = Image.new('RGB', (512, 512), color=(0, 0, 0))
        draw(cond_img, landmarks)
        cond_img.save(out_path)
    
    metadata_path = os.path.join(args.dataset_path, 'metadata.jsonl')
    df = pd.DataFrame(data={"id": range(args.num_data), "text": text})
    df.to_json(metadata_path, orient='records', lines=True)


if __name__ == '__main__':
    args = parse_args()
    main(args)
