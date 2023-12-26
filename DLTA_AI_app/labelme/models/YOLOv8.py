from labelme.DLTA_Model import DLTA_Model
import numpy as np


YOLOv8 = DLTA_Model(
    model_name="YOLOv8",
    model_family="YOLOv8",
    config="",
    checkpoint="",
    task="object detection",
    classes="coco",
    inference_function=None
)


# verify installation and import necessary libraries
# if you only want to test/inference a model in dlta (not for devoloping DLTA-AI):
# This function is not necessary if you have already installed the required libraries in your environment.
# It is provided for integrating the models inside DLTA-AI applications.
def verify_installation():
    try:
        from ultralytics import YOLO
        print("Import YOLOv8 successful")
    except ImportError:
        print("Libraries not installed, installing...")
        import subprocess
        subprocess.run(["pip", "install", "ultralytics==8.0.227"])
        print("Installation complete")




# model initialization 
# this function is nessasary alongside the inference function with the same names to use them successfully in DLTA-AI 
# note that inside each function you should import needed modules and libraries
def initialize(checkpoint  = None , config = None):
    from ultralytics import YOLO

    # YOLO = YOLOv8.install()
    YOLOv8.model = YOLO(checkpoint)
    # YOLOv8.model.fuse()
    # run the model on a sample image to make sure that it is loaded correctly
    YOLOv8.model([np.zeros((640, 640, 3), dtype=np.uint8)], retina_masks=True, verbose=False, conf=0.05)

def inference(img):
    inference_results = YOLOv8.model.predict(
        img, retina_masks=True, verbose=False, conf=0.05 , device = YOLOv8.device)[0]
    
    if not inference_results:
        return []
    # parsing the yoloV8 outputs
    classes = inference_results.boxes.cls.cpu().numpy().astype(int)
    # class_names = np.vectorize(classdict.get)(classes)
    confidences = inference_results.boxes.conf.cpu().numpy()
    masks = inference_results.masks.data.cpu().numpy()
    
    return [classes, confidences, masks]





YOLOv8.verify_installation = verify_installation
YOLOv8.initialize = initialize
YOLOv8.inference = inference
