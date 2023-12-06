from labelme.BIGMODEL import DLTA_Model


YOLOv8 = DLTA_Model(
    model_name="YOLOv8",
    model_family="YOLOv8",
    config="",
    checkpoint="",
    task="object detection",
    classes="coco",
    inference_function=None
)

def xxxx():
    print("xxxx successfully called xxxx")

YOLOv8.inference_function = xxxx
