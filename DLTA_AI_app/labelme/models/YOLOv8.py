from labelme.DLTA_Model import DLTA_Model
import cv2
import numpy as np
import time


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
def imports(type="import"):
    if type == "import":
        from ultralytics import YOLO
        return YOLO
    elif type == "install":
        import subprocess
        subprocess.run(["pip", "install", "ultralytics==8.0.227"])


# model initialization
def initialize(checkpoint  = None , config = None):
    YOLO = YOLOv8.install()
    YOLOv8.model = YOLO(checkpoint)
    # YOLOv8.model.fuse()
    # run the model on a sample image to make sure that it is loaded correctly
    YOLOv8.model([np.zeros((640, 640, 3), dtype=np.uint8)], retina_masks=True, verbose=False, conf=0.05)
    
    


def inference(img):

    start_time = time.time()
    
    inference_results = YOLOv8.model(
        img, retina_masks=True, verbose=False, conf=0.05)[0]
    
    end_time = time.time()
    execution_time = end_time - start_time
    print("1---- Execution time for only model inference:",
          int(execution_time * 1000), "milliseconds")

    return postprocess(inference_results=inference_results)



def postprocess(inference_results):
    start_time = time.time()

    if not inference_results:
        return []

    # parsing the yoloV8 outputs
    classes = inference_results.boxes.cls.cpu().numpy().astype(int)
    # class_names = np.vectorize(classdict.get)(classes)
    confidences = inference_results.boxes.conf.cpu().numpy()
    masks = inference_results.masks.data.cpu().numpy()

    end_time = time.time()
    execution_time = end_time - start_time
    print("2---- Execution time for parsing inside the model:",
          int(execution_time * 1000), "milliseconds")
    return [classes, confidences, masks]



YOLOv8.imports = imports
YOLOv8.initialize = initialize
YOLOv8.inference = inference
