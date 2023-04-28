import sys
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
# import mask_to_polygons from inference.py inside the inference class
# from inference import mask_to_polygons

# create a sam predictor class with funcions to predict and visualize and results


class SamPredictor():
    def __init__(self, model_type, checkpoint_path):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.device = "cuda"
        self.predictor = SamPredictor(self.model)
        self.image = None

    def set_image(self, image):
        self.image = image
        self.predictor.set_image(image)

    def predict(self, point_coords=None, point_labels=None, mask_input=None, multimask_output=False):
        masks, scores, logits = self.predictor.predict(point_coords, point_labels,
                                                       mask_input, multimask_output)
        return masks, scores, logits
