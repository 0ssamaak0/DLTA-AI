import sys
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.measure
import torch

# import mask_to_polygons from inference.py inside the inference class
# from inference import mask_to_polygons

# create a sam predictor class with funcions to predict and visualize and results


class Sam_Predictor():
    def __init__(self, model_type, checkpoint_path, device):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device = self.device)
        self.predictor = SamPredictor(self.model)
        self.image = None
        self.mask_logit = None
        

    def set_new_image(self, image):
        self.image = image
        self.predictor.set_image(image)
    
    def clear_logit(self):
        self.mask_logit = None


    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True, image=None):
        # print(point_coords , point_labels)
        # print(f'----------------------- into SAM predict')
        # print(f'point_coords: {point_coords}, point_labels: {point_labels}, box: {box}')
        if box is None:
            # print(f'----------------------- no boxes')
            if self.mask_logit is None:
                masks, scores, logits = self.predictor.predict(point_coords=point_coords, 
                                                               point_labels=point_labels, 
                                                               multimask_output=multimask_output)
            else:
                masks, scores, logits = self.predictor.predict(point_coords=point_coords, 
                                                               point_labels=point_labels,
                                                               mask_input=self.mask_logit[None, :, :],
                                                               multimask_output=multimask_output)
        else:
            # print(f'----------------------- boxes')
            if len(box) == 1:
                # print(f'----------------------- only one box')
                input_box = np.array(box[0])
                masks, scores, logits = self.predictor.predict(point_coords=point_coords, 
                                                            point_labels=point_labels,
                                                            box=input_box[None, :],
                                                            multimask_output=multimask_output)
                
            else:
                # print(f'----------------------- multiple boxes')
                input_box = np.array(box[0])
                box_tensor = torch.tensor(box, device=self.predictor.device)
                box_transformed = self.predictor.transform.apply_boxes_torch(box_tensor, image.shape[:2])
                masks, scores, logits = self.predictor.predict_torch(point_coords=None, 
                                                            point_labels=None,
                                                            boxes=box_transformed,
                                                            multimask_output=False)
        
        if multimask_output:
            if box is not None and len(box) != 1:
                logits = torch.Tensor.cpu(logits).numpy().reshape(-1, logits.shape[-2], logits.shape[-1])
                masks = torch.Tensor.cpu(masks).numpy().reshape(-1, masks.shape[-2], masks.shape[-1])
                scores = torch.Tensor.cpu(scores).numpy().reshape(-1)
            self.mask_logit = logits[np.argmax(scores), :, :]  # Choose the model's best mask logit
            mask = masks[np.argmax(scores), :, :]  # Choose the model's best mask
            score = np.max(scores)  # Choose the model's best score
        
        return mask, score
    
    
    def predict_batch(self,  boxes=None, image=None):
        boxes = np.array(boxes)
        input_boxes = torch.tensor(boxes, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, scores, logits = self.predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
        return masks, scores

    
    
    def get_contour_length(self, contour):
        contour_start = contour
        contour_end = np.r_[contour[1:], contour[0:1]]
        return np.linalg.norm(contour_end - contour_start, axis=1).sum()

    def mask_to_polygons(self, mask, n_points=25, resize_factors=[1.0, 1.0]):
        mask = mask > 0.0
        contours = skimage.measure.find_contours(mask)
        if len(contours) == 0:
            return []
        contour = max(contours, key=self.get_contour_length)
        coords = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() / 100,
        )

        coords = coords * resize_factors
        # convert coords from x y to y x
        coords = np.fliplr(coords)

        # segment_points are a list of coords
        segment_points = coords.astype(int)
        polygon = segment_points
        return polygon

    def polygon_to_shape(self, polygon,score):
        shape = {}
        shape["label"] = "SAM instance"
        shape["content"] = str(round(score, 2))
        shape["group_id"] = None
        shape["shape_type"] = "polygon"
        shape["bbox"] = self.get_bbox(polygon)

        shape["flags"] = {}
        shape["other_data"] = {}

        # shape_points is result["seg"] flattened
        shape["points"] = [item for sublist in polygon
                               for item in sublist]
        # print(shape)
        return shape
    
    def get_bbox(self, segmentation):
        x = []
        y = []
        for i in range(len(segmentation)):
            x.append(segmentation[i][0])
            y.append(segmentation[i][1])
        # get the bbox in xyxy format
        if len(x) == 0 or len(y) == 0:
            return []
        bbox = [min(x), min(y), max(x), max(y)]
        return bbox
        
    def check_image(self , new_image):
        if not np.array_equal(self.image, new_image):
            # print("image changed_1")
            self.mask_logit = None
            self.image = new_image
            self.predictor.set_image(new_image)
            # print("image changed_2")
            return False
        return True

    