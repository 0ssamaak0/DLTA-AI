from labelme.DLTA_Model import DLTA_Model
import numpy as np
import time

# should be later imported in the imports function
from mmdet.apis import init_detector, inference_detector

mmdetection = DLTA_Model(
    model_name="mmdetection",
    model_family="mmdetection",
    config="",
    checkpoint="",
    task="object detection",
    classes="coco",
    inference_function=None
)


# prepare imports ( this function isn't yet tested thoroughly )
def imports(type="import"):
    if type == "import":
        from mmdet.apis import init_detector, inference_detector
        return init_detector, inference_detector
    elif type == "install":
        import subprocess
        # uncomment the following after testing and debugging
        # subprocess.run(["pip", "install", "-U", "openmim"])
        # subprocess.run(["mim", "install", "mmengine>=0.7.0"])
        # subprocess.run(["mim", "install", "mmcv>=2.0.0rc4"])
        # subprocess.run(["mim", "install", "mmdet"])


# model initialization
def initialize(checkpoint=None, config=None):
    mmdetection.model = init_detector(config, checkpoint,
                        device='cuda:0')  

def inference(img):
    
    start_time = time.time()
    inference_results = inference_detector(mmdetection.model, img)

    end_time = time.time()
    execution_time = end_time - start_time
    print("1---- Execution time for only model inference:",
          int(execution_time * 1000), "milliseconds")

    start_time = time.time()

    if not inference_results:
        return []
    # bboxes = inference_results.pred_instances.bboxes.cpu().numpy()
    masks = inference_results.pred_instances.masks.cpu().numpy()
    classes = inference_results.pred_instances.labels.cpu().numpy()
    confidences = inference_results.pred_instances.scores.cpu().numpy()
    

    end_time = time.time()
    execution_time = end_time - start_time
    print("2---- Execution time for parsing inside the model:",
          int(execution_time * 1000), "milliseconds")
    
    return [classes, confidences, masks]

mmdetection.imports = imports
mmdetection.initialize = initialize
mmdetection.inference = inference
