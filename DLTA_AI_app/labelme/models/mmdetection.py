from labelme.DLTA_Model import DLTA_Model

mmdetection = DLTA_Model(
    model_name="mmdetection",
    model_family="mmdetection",
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
        from mmdet.apis import init_detector, inference_detector
        print("Import mmdetection successful")
    except ImportError:
        print("Libraries not installed, installing...")
        import subprocess
        subprocess.run(["pip", "install", "-U", "openmim"])
        subprocess.run(["mim", "install", "mmengine>=0.7.0"])
        subprocess.run(["mim", "install", "mmcv>=2.0.0rc4"])
        subprocess.run(["mim", "install", "mmdet"])
        print("Installation complete")



# model initialization
def initialize(checkpoint=None, config=None):
    from mmdet.apis import init_detector
    mmdetection.model = init_detector(config, checkpoint,
                        device= mmdetection.device)  

def inference(img):
    from mmdet.apis import inference_detector

    inference_results = inference_detector(mmdetection.model, img)

    if not inference_results:
        return []
    # bboxes = inference_results.pred_instances.bboxes.cpu().numpy()
    masks = inference_results.pred_instances.masks.cpu().numpy()
    classes = inference_results.pred_instances.labels.cpu().numpy()
    confidences = inference_results.pred_instances.scores.cpu().numpy()
    return [classes, confidences, masks]


mmdetection.verify_installation = verify_installation
mmdetection.initialize = initialize
mmdetection.inference = inference
