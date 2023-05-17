import datetime
import glob
import json
import os
import csv
import numpy as np
from PyQt5.QtWidgets import QFileDialog

def center_of_polygon(polygon):
    """
    Calculates the center of a polygon defined by a list of consecutive pairs of vertices.

    Args:
        polygon (list): A list of consecutive pairs of vertices.

    Returns:
        tuple: The center point of the polygon as a tuple of two integers.
    """
    # Extract x and y coordinates from the list of polygon vertices
    x_coords = polygon[::2]
    y_coords = polygon[1::2]

    # Calculate the center point of the polygon
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    center = [center_x, center_y]

    # Find the center of the bounding box of the polygon
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    centers_rec = [(xmin + xmax) / 2, (ymin + ymax) / 2]

    # Calculate the final center point as a weighted average of the polygon center and the bounding box center
    (xp, yp) = centers_rec
    (xn, yn) = center
    r = 0.5
    x = r * xn + (1 - r) * xp
    y = r * yn + (1 - r) * yp
    center = (int(x), int(y))
    return center

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


def get_area_from_polygon(polygon, mode="segmentation"):
    if mode == "segmentation":
        # Convert the list to a numpy array of shape (n, 2) where n is the number of vertices
        polygon = np.array(polygon).reshape(-1, 2)
        # Use the shoelace formula to calculate the area of the polygon
        area = 0.5 * np.abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) -
                            np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
        # Return the area
        return area

    elif mode == "bbox":
        # Unpack the list into variables
        x_min, y_min, width, height = polygon
        # Calculate the area by multiplying the width and height
        area = width * height
        # Return the area
        return area

    else:
        raise ValueError("mode must be either 'segmentation' or 'bbox'")


def exportCOCO(target_directory, save_path, annotation_path):
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

    if len(json_paths) == 0:
        raise ValueError("No json files found in the directory")
    
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
                    "iscrowd": 0

                })
                try:
                    annotations[-1]["segmentation"] = [data["shapes"]
                                                       [j]["points"]]
                    annotations[-1]["area"] = get_area_from_polygon(
                        annotations[-1]["segmentation"][0], mode="segmentation")
                except:
                    annotations[-1]["area"] = get_area_from_polygon(
                        annotations[-1]["bbox"], mode="bbox")
                try:
                    annotations[-1]["score"] = float(
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

    # write in the output file in json, format the output to be pretty
    with open(annotation_path, 'w') as outfile:
        json.dump(file, outfile, indent=4)

    return annotation_path


def exportCOCOvid(results_file, vid_width, vid_height, annotation_path):
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
            if len(frame["frame_data"]) == 0:
                continue
            images.append({
                "id": frame["frame_idx"],
                "width": vid_width,
                "height": vid_height,
                "file_name": f"frame {frame['frame_idx']}",
            })
            for object in frame["frame_data"]:
                annotations.append({
                    "id": len(annotations),
                    "image_id": frame["frame_idx"],
                    "category_id": object["class_id"] + 1,
                    "iscrowd": 0
                })
                if annotations[-1]["category_id"] == 0:
                    coco_classes.append(object["class_name"].lower())
                    annotations[-1]["category_id"] = coco_classes.index(object["class_name"].lower()) + 1
                try:
                    annotations[-1]["bbox"] = get_bbox(object["segment"])
                except:
                    annotations[-1]["bbox"] = object["bbox"]
                try:
                    annotations[-1]["segmentation"] = [
                        [val for sublist in object["segment"] for val in sublist]]

                    annotations[-1]["area"] = get_area_from_polygon(
                        annotations[-1]["segmentation"][0], mode="segmentation")
                except:
                    annotations[-1]["area"] = get_area_from_polygon(
                        annotations[-1]["bbox"], mode="bbox")
                try:
                    annotations[-1]["score"] = float(object["confidence"])
                except:
                    pass
                used_classes.add(annotations[-1]["category_id"])

    used_classes = sorted(list(used_classes))
    file["categories"] = [{"id": i, "name": coco_classes[i - 1]}
                          for i in used_classes]
    file["images"] = images
    file["annotations"] = annotations

    # write in the output file in json, format the output to be pretty
    with open(annotation_path, 'w') as outfile:
        json.dump(file, outfile, indent=4)

    return annotation_path


def exportMOT(results_file, annotation_path):
    rows = []
    with open(results_file) as f:
        data = json.load(f)
        for frame in data:
            for object in frame["frame_data"]:
                # bbox = get_bbox(object["segment"])
                rows.append(
                    f'{frame["frame_idx"]}, {object["tracker_id"]},  {object["bbox"][0]},  {object["bbox"][1]},  {object["bbox"][2]},  {object["bbox"][3]},  {object["confidence"]}, {object["class_id"] + 1}, 1')

    with open(annotation_path, 'w') as outfile:
        # write each row in the file as a new line
        outfile.write("\n".join(rows))

    return annotation_path




# Define a class that inherits from QFileDialog
class FolderDialog(QFileDialog):
    # Override the __init__ method to accept parameters
    def __init__(self, default_file_name, default_format):
        # Call the parent constructor
        super().__init__()
        # Set the mode to save a file
        self.setAcceptMode(QFileDialog.AcceptSave)
        # Set the default file name
        self.selectFile(default_file_name)
        # Set the default format
        self.setNameFilters(
            [f"{default_format.upper()} (*.{default_format.lower()})", "All Files (*)"])
        self.selectNameFilter(
            f"{default_format.upper()} (*.{default_format.lower()})")
        # Set dialog title
        self.setWindowTitle("Save Annotations")
