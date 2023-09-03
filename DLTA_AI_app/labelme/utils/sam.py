from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import numpy as np
import torch
from .helpers import mathOps


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

        
    def check_image(self , new_image):
        if not np.array_equal(self.image, new_image):
            # print("image changed_1")
            self.mask_logit = None
            self.image = new_image
            self.predictor.set_image(new_image)
            # print("image changed_2")
            return False
        return True

    def get_all_shapes(self, image, iou_threshold):
        
        # self.mask_generator = SamAutomaticMaskGenerator(
        #     model: Sam,
        #     points_per_side: Optional[int] = 32,
        #     points_per_batch: int = 64,
        #     pred_iou_thresh: float = 0.88,
        #     stability_score_thresh: float = 0.95,
        #     stability_score_offset: float = 1.0,
        #     box_nms_thresh: float = 0.7,
        #     crop_n_layers: int = 0,
        #     crop_nms_thresh: float = 0.7,
        #     crop_overlap_ratio: float = 512 / 1500,
        #     crop_n_points_downscale_factor: int = 1,
        #     point_grids: Optional[List[np.ndarray]] = None,
        #     min_mask_region_area: int = 0,
        #     output_mode: str = "binary_mask",
        # )
        
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model = self.model,
            # points_per_side = 32,
            # points_per_batch = 64,
            # pred_iou_thresh = 0.88,
            # stability_score_thresh = 0.95,
            # stability_score_offset = 1.0,
            # box_nms_thresh = 0.3,
            # crop_n_layers = 0,
            # crop_nms_thresh = 0.7,
            # crop_overlap_ratio = 512 / 1500,
            # crop_n_points_downscale_factor = 1,
            # point_grids = None,
            # min_mask_region_area = image.shape[0] * image.shape[1] * 0.0005,
            # output_mode = "binary_mask",
        )
        
        # Arguments(
        #   model (Sam): The SAM model to use for mask prediction.
        
        #   points_per_side (int or None): The number of points to be sampled
        #     along one side of the image. The total number of points is
        #     points_per_side**2. If None, 'point_grids' must provide explicit
        #     point sampling.
        
        #   points_per_batch (int): Sets the number of points run simultaneously
        #     by the model. Higher numbers may be faster but use more GPU memory.
        
        #   pred_iou_thresh (float): A filtering threshold in [0,1], using the
        #     model's predicted mask quality.
          
        #   stability_score_thresh (float): A filtering threshold in [0,1], using
        #     the stability of the mask under changes to the cutoff used to binarize
        #     the model's mask predictions.
          
        #   stability_score_offset (float): The amount to shift the cutoff when
        #     calculated the stability score.
          
        #   box_nms_thresh (float): The box IoU cutoff used by non-maximal
        #     suppression to filter duplicate masks.
          
        #   crop_n_layers (int): If >0, mask prediction will be run again on
        #     crops of the image. Sets the number of layers to run, where each
        #     layer has 2**i_layer number of image crops.
          
        #   crop_nms_thresh (float): The box IoU cutoff used by non-maximal
        #     suppression to filter duplicate masks between different crops.
          
        #   crop_overlap_ratio (float): Sets the degree to which crops overlap.
        #     In the first crop layer, crops will overlap by this fraction of
        #     the image length. Later layers with more crops scale down this overlap.
          
        #   crop_n_points_downscale_factor (int): The number of points-per-side
        #     sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          
        #   point_grids (list(np.ndarray) or None): A list over explicit grids
        #     of points used for sampling, normalized to [0,1]. The nth grid in the
        #     list is used in the nth crop layer. Exclusive with points_per_side.
          
        #   min_mask_region_area (int): If >0, postprocessing will be applied
        #     to remove disconnected regions and holes in masks with area smaller
        #     than min_mask_region_area. Requires opencv.
          
        #   output_mode (str): The form masks are returned in. Can be 'binary_mask',
        #     'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
        #     For large resolutions, 'binary_mask' may consume large amounts of
        #     memory.
        # )
        
        
        
        # sam_result is a list of dictionaries
        # each dictionary (mask) has the following keys:
            # segmentation - [np.ndarray] - the mask with (W, H) shape, and bool type
            # area - [int] - the area of the mask in pixels
            # bbox - [List[int]] - the boundary box of the mask in xywh format
            # predicted_iou - [float] - the model's own prediction for the quality of the mask
            # point_coords - [List[List[float]]] - the sampled input point that generated this mask
            # stability_score - [float] - an additional measure of mask quality
            # crop_box - List[int] - the crop of the image used to generate this mask in xywh format
            
        sam_result = self.mask_generator.generate(image)
        shapes = mathOps.OURnms_areaBased_fromSAM(sam_result, iou_threshold=iou_threshold) # with AREA not score
        
        return shapes
    