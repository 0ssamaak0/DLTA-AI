import datetime
import glob
import json
import os


def get_bbox(segmentation):
    x = []
    y = []
    for i in range(len(segmentation)):
        if i % 2 == 0:
            x.append(segmentation[i])
        else:
            y.append(segmentation[i])
    return [min(x), min(y), max(x) - min(x), max(y) - min(y)]


def exportCOCO(target_directory, save_path):
    '''
    Exports the annotations in the Labelmm format to COCO format
    input:
        target_directory: directory of the annotations
        save_path: path of the image
    '''
    # if directory
    file_path = target_directory

    # if one file
    # TODO support \

    if file_path == "":
        file_path = save_path
        file_path = file_path.split("/")[:-1]
        file_path = "/".join(file_path)
    output_name = "coco"

    file = {}
    coco_categories = {
        "person": 1,
        "bicycle": 2,
        "car": 3,
        "motorcycle": 4,
        "bus": 6,
        "truck": 8,
    }
    # write the info header
    file["info"] = {
        "description": "Exported from Labelmm",
        # "url": "n/a",
        # "version": "n/a",
        "year": datetime.datetime.now().year,
        # "contributor": "n/a",
        "date_created": datetime.date.today().strftime("%Y/%m/%d")
    }

    # write list of COCO classes
    coco_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    used_classes = set()
    json_paths = glob.glob(f"{file_path}/*.json")
    annotations = []
    images = []
    for i in range(len(json_paths)):
        try:
            with open(json_paths[i]) as f:
                # image data
                data = json.load(f)
                images.append({
                    "id": i,
                    "width": data["imageWidth"],
                    "height": data["imageHeight"],
                    "file_name": json_paths[i].split("/")[-1].replace(".json", ".jpg"),
                })
            for j in range(len(data["shapes"])):
                if len(data["shapes"][j]["points"],) == 0:
                    continue
                if data["shapes"][j]["label"].lower() not in coco_classes:
                    coco_classes.append((data["shapes"][j]["label"].lower()))
                annotations.append({
                    "id": len(annotations),
                    "image_id": i,
                    "category_id": coco_classes.index(data["shapes"][j]["label"].lower()) + 1,
                    "segmentation": data["shapes"][j]["points"],
                    "bbox": get_bbox(data["shapes"][j]["points"]),

                })
                try:
                    annotations[-1]["confidence"]: float(
                        data["shapes"][j]["content"])
                except:
                    pass
                used_classes.add(coco_classes.index(
                    data["shapes"][j]["label"].lower()) + 1)
        except Exception as e:
            print(f"Error with {json_paths[i]}")
            print(e)
            continue
    used_classes = sorted(used_classes)
    print(f"Used classes: {used_classes}")
    file["categories"] = [{"id": i, "name": coco_classes[i - 1]}
                          for i in used_classes]
    file["images"] = images
    file["annotations"] = annotations

    # make a directory under file_path called Annotations (if it doesn't exist)
    if not os.path.exists(f"{file_path}/Annotations"):
        os.makedirs(f"{file_path}/Annotations")

    # write in the output file in json, format the output to be pretty
    with open(f"{file_path}/Annotations/{output_name}.json", 'w') as outfile:
        json.dump(file, outfile, indent=4)

    print(f"Exported to {file_path}/Annotations/{output_name}.json")
