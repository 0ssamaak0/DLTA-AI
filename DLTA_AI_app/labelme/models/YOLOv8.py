from labelme.DLTA_Model import DLTA_Model
import cv2
import numpy as np
from supervision.detection.core import Detections
from labelme.utils.helpers import mathOps

YOLOv8 = DLTA_Model(
    model_name="YOLOv8",
    model_family="YOLOv8",
    config="",
    checkpoint="",
    task="object detection",
    classes="coco",
    inference_function=None
)

# prepare imports
def imports(type = "import"):
    if type == "import":
        from ultralytics import YOLO
        return YOLO
    elif type == "install":
        import subprocess
        subprocess.run(["pip", "install", "ultralytics==8.0.61"])

YOLOv8.imports = imports


# model initialization
def initialize(checkpoint_path, config_path = None):
    YOLO = YOLOv8.install()
    YOLOv8.model = YOLO(checkpoint_path)
    YOLOv8.model.fuse()

YOLOv8.initialize = initialize

def inference(img, threshold, classdict):
    if isinstance(img, str):
            img = cv2.imread(img)

    # get image size
    img_resized = cv2.resize(img , (640, 640))
    # default yolo arguments from yolov8 tracking repo
        # imgsz=(640, 640),  # inference size (height, width)
        # conf_thres=0.25,  # confidence threshold
        # iou_thres=0.45,  # NMS IOU threshold
        # max_det=1000,  # maximum detections per image
    results = YOLOv8.model(img_resized , conf = 0.25 , iou=  0.45 , verbose = False)
    results = results[0]
    # if len results is 0 then return empty dict
    if len(results) == 0:
        return []

    # get masks and convert them to numpy array
    masks = results.masks.cpu().numpy().masks
    masks = masks > 0.0

    # used to convert boxes and masks to original image size
    org_size = img.shape[:2]
    out_size = masks.shape[1:]

    # print(f'org_size : {org_size} , out_size : {out_size}')

    # convert boxes to original image size same as the masks (coords = coords * org_size / out_size)
    boxes = results.boxes.xyxy.cpu().numpy()
    boxes = boxes * np.array([org_size[1] / out_size[1], org_size[0] /
                            out_size[0], org_size[1] / out_size[1], org_size[0] / out_size[0]])
    
    
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    
    
    results = []

    for detection in zip(masks, boxes, confidences, class_ids):
        mask, bbox, confidence, class_id = detection
        result = {}
        result["mask"] = mask
        result["class"] = classdict.get(class_id)
        result["confidence"] = str(round(confidence, 2))
        result["bbox"] = bbox.astype(int)
        if result["class"] == None:
            continue
        results.append(result)


    resize_factors = [org_size[0] / out_size[0] , org_size[1] / out_size[1]]

    print("Inference done via DLTA_Model")

    return results, resize_factors
    
    # detections = Detections(
    #     xyxy=boxes,
    #     confidence=results.boxes.conf.cpu().numpy(),
    #     class_id=results.boxes.cls.cpu().numpy().astype(int)
    # )

    # polygons = []
    # result_dict = {}

    # resize_factors = [org_size[0] / out_size[0] , org_size[1] / out_size[1]]
    
    # if len(masks) == 0:
    #     return {"results":{}}
    # for mask in masks:
    #     polygon = mathOps.mask_to_polygons(
    #         mask, resize_factors=resize_factors)
    #     polygons.append(polygon)

    # # detection is a tuple of  (box, confidence, class_id, tracker_id)
    # ind = 0
    # res_list = []
    # for detection in detections:
    #     if round(detection[1], 2) < float(threshold):
    #         continue
    #     result = {}
    #     result["class"] = classdict.get(int(detection[2]))
    #     result["confidence"] = str(round(detection[1], 2))
    #     result["bbox"] = detection[0].astype(int)
    #     result["seg"] = polygons[ind]
    #     ind += 1
    #     if result["class"] == None:
    #         continue
    #     if len(result["seg"]) < 3:
    #         continue

    #     res_list.append(result)
    # result_dict["results"] = res_list
    # print("Inference done via DLTA_Model")
    # return result_dict["results"]

YOLOv8.inference = inference
