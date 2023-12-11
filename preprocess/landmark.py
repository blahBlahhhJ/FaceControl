import argparse
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import pandas as pd
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import numpy as np

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

def bbox_from_landmarks(landmarks_str, processor):
    landmarks = landmarks_str.strip().split('\n')
    landmarks = [k.split(' ') for k in landmarks]
    landmarks = [(float(x), float(y)) for x, y in landmarks]
    landmarks_x, landmarks_y = zip(*landmarks)
    
    x_min, x_max = min(landmarks_x), max(landmarks_x)
    y_min, y_max = min(landmarks_y), max(landmarks_y)
    width = x_max - x_min
    height = y_max - y_min

    # Give it a little room; I think it works anyway
    x_min -= 5
    y_min -= 5
    width += 10
    height += 10
    bbox = (x_min, y_min, width, height)
    return bbox

def draw(image, landmarks, radius=2.5, color="white"):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def draw_path(draw, points, color="white", width=3):
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=color, width=width)

def draw2(image, landmarks, radius=2.5, color="white"):
    draw = ImageDraw.Draw(image)
    ldmks = [tuple(dot) for dot in landmarks]
    draw_path(draw, ldmks[0: 17], color='green')
    draw_path(draw, ldmks[17: 22], color='yellow')
    draw_path(draw, ldmks[22: 27], color='yellow')
    draw_path(draw, ldmks[27: 31], color='orange')
    draw_path(draw, ldmks[31: 36], color='orange')
    draw.polygon(ldmks[36: 42], fill="#ff80ea", outline='magenta')
    draw.polygon(ldmks[42: 48], fill="#ff80ea", outline='magenta')
    draw.polygon(ldmks[48: 60], fill="#99eeff", outline='cyan')
    draw.polygon(ldmks[60: 68], fill="#66b3ff", outline='blue')


def main(args):
    processor = SPIGAFramework(ModelConfig("300wpublic"))

    for i in tqdm(range(args.num_data)):
        image_path = os.path.join(args.dataset_path, f"{i:06d}.png")
        landmark_path = os.path.join(args.dataset_path, f"{i:06d}_ldmks.txt")
        out_path = os.path.join(args.dataset_path, f"{i:06d}_cond2.png")

        assert os.path.exists(image_path), f"Missing data: {image_path}"
        assert os.path.exists(landmark_path), f"Missing data: {landmark_path}"

        with open(landmark_path, 'r') as f:
            landmarks_3d = f.read()

        image = np.array(Image.open(image_path))
        # BGR
        image = image[:, :, ::-1]
        bbox = bbox_from_landmarks(landmarks_3d, processor)
        features = processor.inference(image, [bbox])
        landmarks = features["landmarks"][0]

        cond_img = Image.new('RGB', (512, 512), color=(0, 0, 0))
        draw2(cond_img, landmarks)
        cond_img.save(out_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
