import datetime
import glob
import json
import os


def get_bbox(segmentation):
    try:
        x = []
        y = []
        for i in range(len(segmentation)):
            if i % 2 == 0:
                x.append(segmentation[i])
            else:
                y.append(segmentation[i])
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]
    except:
        print("conversion required")
        # convert segmentation into 1D array
        segmentation = [item for sublist in segmentation for item in sublist]
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
                    "bbox": get_bbox(data["shapes"][j]["points"]),

                })
                try:
                    annotations[-1]["segmentation"] = data["shapes"][j]["points"]
                except:
                    pass
                try:
                    annotations[-1]["confidence"] = float(
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

    return (f"{file_path}/Annotations/{output_name}.json")


def exportCOCOvid(results_file, save_path, vid_width, vid_height, output_name="coco_vid"):
    file = {}
    file["info"] = {
        "description": "Exported from Labelmm",
        # "url": "n/a",
        # "version": "n/a",
        "year": datetime.datetime.now().year,
        # "contributor": "n/a",
        "date_created": datetime.date.today().strftime("%Y/%m/%d")
    }

    annotations = []
    images = []
    coco_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    used_classes = set()
    with open(results_file) as f:
        data = json.load(f)
        for frame in data:
            images.append({
                "id": frame["frame_idx"] + 1,
                "width": vid_width,
                "height": vid_height,
                "file_name": f"frame {frame['frame_idx'] + 1}",
            })
            for object in frame["frame_data"]:
                annotations.append({
                    "id": frame["frame_idx"],
                    "image_id": frame["frame_idx"] + 1,
                    "category_id": object["class_id"] + 1,
                })
                try:
                    annotations[-1]["bbox"] = get_bbox(object["segment"])
                except:
                    annotations[-1]["bbox"] = object["bbox"]
                try:
                    annotations[-1]["segmentation"] = object["segment"]
                except:
                    pass
                try:
                    annotations[-1]["confidence"] = float(object["confidence"])
                except:
                    pass
                used_classes.add(object["class_id"])

    used_classes = sorted(list(used_classes))
    file["categories"] = [{"id": i + 1, "name": coco_classes[i]}
                          for i in used_classes]
    file["images"] = images
    file["annotations"] = annotations

    # make a directory under save_path called Annotations (if it doesn't exist)
    if not os.path.exists(f"{save_path}/Annotations"):
        os.makedirs(f"{save_path}/Annotations")

    # write in the output file in json, format the output to be pretty
    with open(f"{save_path}/Annotations/{output_name}.json", 'w') as outfile:
        json.dump(file, outfile, indent=4)

    return (f"{save_path}/Annotations/{output_name}.json")
