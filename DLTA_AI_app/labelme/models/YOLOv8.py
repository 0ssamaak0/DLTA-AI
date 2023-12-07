from labelme.DLTA_Model import DLTA_Model


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
def initialize(checkpoint_path):
    YOLO = YOLOv8.install()
    model = YOLO(checkpoint_path)
    model.fuse()
    return model

YOLOv8.initialize = initialize

def xxxx():
    print("xxxx successfully called xxxx")

YOLOv8.inference_function = xxxx