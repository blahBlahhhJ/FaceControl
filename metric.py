from transformers import pipeline
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import numpy as np
import torch
from torchvision.ops import box_iou


class FaceSimilarity:
    def __init__(self, resolution=512):
        self.detector = pipeline(model="aditmohan96/detr-finetuned-face", device=0)
        self.processor = SPIGAFramework(ModelConfig("300wpublic"))

        self.resolution = resolution
        self.diagonal = 2 ** 0.5 * resolution

        self.worse_similarity = 1.0
        self.worse_iou = 0.0

    def extract_bbox(self, img):
        results = self.detector(img, threshold=0.7)

        # If detr cannot detect faces, set return worse case
        if len(results) == 0:
            return None
        else:
            best_score = 0.0
            box = None

            # pick face with largest score
            for res in results:
                if res['score'] > best_score:
                    best_score = res['score']
                box = [res['box']['xmin']-5, res['box']['ymin']-5, res['box']['xmax']+5, res['box']['ymax']+5]
        return box

    def compute_iou(self, box1, box2):
        box1 = torch.tensor(box1).view(1, 4)
        box2 = torch.tensor(box2).view(1, 4)
        return box_iou(box1, box2).item()

    def compute_similarity(self, ldmk1, ldmk2):
        # (68, 2)
        ldmk1 = np.array(ldmk1)
        ldmk2 = np.array(ldmk2)

        dist = np.mean(np.sum((ldmk1 - ldmk2) ** 2, axis=1) ** 0.5)

        # normalize to range 0-1
        return dist / self.diagonal

    def extract_landmark(self, box, img):
        img = np.array(img)[:, :, ::-1]
        bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        features = self.processor.inference(img, [bbox])
        landmarks = features["landmarks"][0]
        return landmarks

    def __call__(self, img1, img2):
        box1 = self.extract_bbox(img1)
        box2 = self.extract_bbox(img2)
        
        if box1 is None or box2 is None:
            return self.worse_iou, self.worse_similarity

        ldmk1 = self.extract_landmark(box1, img1)
        ldmk2 = self.extract_landmark(box2, img2)

        return self.compute_iou(box1, box2), self.compute_similarity(ldmk1, ldmk2)


            

            