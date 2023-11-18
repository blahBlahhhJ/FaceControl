from transformers import pipeline, CLIPModel, CLIPProcessor
import torch
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import numpy as np
import torch
from torchvision.ops import box_iou


class FaceSimilarity:
    def __init__(self, resolution=512):
        self.face_detector = pipeline(model="aditmohan96/detr-finetuned-face", device=0)
        self.face_processor = SPIGAFramework(ModelConfig("300wpublic"))
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").cuda()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        self.resolution = resolution
        self.diagonal = 2 ** 0.5 * resolution

        self.worse_distance = 1.0
        self.worse_iou = 0.0

    def extract_bbox(self, img):
        results = self.face_detector(img, threshold=0.7)

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

    def extract_landmark(self, box, img):
        img = np.array(img)[:, :, ::-1]
        bbox = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
        features = self.face_processor.inference(img, [bbox])
        landmarks = features["landmarks"][0]

        return landmarks
    
    def compute_iou(self, box1, box2):
        box1 = torch.tensor(box1).view(1, 4)
        box2 = torch.tensor(box2).view(1, 4)
        iou = box_iou(box1, box2).item()

        return iou

    def compute_distance(self, ldmk1, ldmk2):
        # (68, 2)
        ldmk1 = np.array(ldmk1)
        ldmk2 = np.array(ldmk2)

        dist = np.mean(np.sum((ldmk1 - ldmk2) ** 2, axis=1) ** 0.5)

        # normalize to range 0-1
        return dist / self.diagonal
    
    def compute_clip_score(self, img, prompt):
        with torch.no_grad():
            processed_input = self.clip_processor(text=prompt, images=img, return_tensors="pt")

            img_features = self.clip.get_image_features(processed_input["pixel_values"].cuda())
            img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

            max_position_embeddings = self.clip.config.text_config.max_position_embeddings
            if processed_input["attention_mask"].shape[-1] > max_position_embeddings:
                processed_input["attention_mask"] = processed_input["attention_mask"][..., :max_position_embeddings]
                processed_input["input_ids"] = processed_input["input_ids"][..., :max_position_embeddings]

            txt_features = self.clip.get_text_features(
                processed_input["input_ids"].cuda(), processed_input["attention_mask"].cuda()
            )
            txt_features = txt_features / txt_features.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity between feature vectors
            score = 100 * (img_features * txt_features).sum(axis=-1).item()

        return score

    def __call__(self, img1, img2, prompt):
        clip_score = self.compute_clip_score(img2, prompt)

        box1 = self.extract_bbox(img1)
        box2 = self.extract_bbox(img2)
        
        if box1 is None or box2 is None:
            return clip_score, self.worse_iou, self.worse_distance

        ldmk1 = self.extract_landmark(box1, img1)
        ldmk2 = self.extract_landmark(box2, img2)

        return clip_score, self.compute_iou(box1, box2), self.compute_distance(ldmk1, ldmk2)


            

            