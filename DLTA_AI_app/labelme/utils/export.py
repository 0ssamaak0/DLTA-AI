import datetime
import glob
import json
import os
import csv
import numpy as np
from PyQt5.QtWidgets import QFileDialog

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def center_of_polygon(polygon):
    """
    Calculates the center of a polygon defined by a list of consecutive pairs of vertices.

    Args:
        polygon (list): A list of consecutive pairs of vertices.

    Returns:
        tuple: The center point of the polygon as a tuple of two integers.
    """
    # Extract x and y coordinates from the list of polygon vertices
    x_coords = polygon[::2]  # Get every other element starting from the first (x-coordinates)
    y_coords = polygon[1::2]  # Get every other element starting from the second (y-coordinates)

    # Calculate the center point of the polygon
    center_x = sum(x_coords) / len(x_coords)  # Calculate the average x-coordinate
    center_y = sum(y_coords) / len(y_coords)  # Calculate the average y-coordinate
    center = [center_x, center_y]  # Store the center point as a list

    # Find the center of the bounding box of the polygon
    xmin = min(x_coords)  # Get the minimum x-coordinate
    xmax = max(x_coords)  # Get the maximum x-coordinate
    ymin = min(y_coords)  # Get the minimum y-coordinate
    ymax = max(y_coords)  # Get the maximum y-coordinate
    centers_rec = [(xmin + xmax) / 2, (ymin + ymax) / 2]  # Calculate the center of the bounding box

    # Calculate the final center point as a weighted average of the polygon center and the bounding box center
    (xp, yp) = centers_rec  # Unpack the bounding box center coordinates
    (xn, yn) = center  # Unpack the polygon center coordinates
    r = 0.5  # Set the weight for the polygon center
    x = r * xn + (1 - r) * xp  # Calculate the weighted average x-coordinate
    y = r * yn + (1 - r) * yp  # Calculate the weighted average y-coordinate
    center = (int(x), int(y))  # Store the final center point as a tuple of integers
    return center  # Return the final center point

def get_bbox(segmentation):
    """
    Calculates the bounding box of a polygon defined by a list of consecutive pairs of x-y coordinates.

    Args:
        segmentation (list): A list of consecutive pairs of x-y coordinates that define a polygon.

    Returns:
        list: A list of four values: the minimum x and y values, and the width and height of the bounding box that encloses the polygon.
    """
    try:
        x = []
        y = []
        # Extract x and y coordinates from the segmentation list
        for i in range(len(segmentation)):
            if i % 2 == 0:
                x.append(segmentation[i])
            else:
                y.append(segmentation[i])
        # Calculate the minimum x and y values, and the width and height of the bounding box
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]
    except:
        # If an exception occurs (e.g. if the segmentation list is empty), convert it into a 1D array
        segmentation = [item for sublist in segmentation for item in sublist]
        x = []
        y = []
        # Extract x and y coordinates from the 1D segmentation array
        for i in range(len(segmentation)):
            if i % 2 == 0:
                x.append(segmentation[i])
            else:
                y.append(segmentation[i])
        # Calculate the minimum x and y values, and the width and height of the bounding box
        return [min(x), min(y), max(x) - min(x), max(y) - min(y)]


def get_area_from_polygon(polygon, mode="segmentation"):
    """
    Calculates the area of a polygon defined by a list of consecutive pairs of x-y coordinates.

    Args:
        polygon (list): A list of consecutive pairs of x-y coordinates that define a polygon.
        mode (str): The mode to use for calculating the area. Can be "segmentation" (default) or "bbox".

    Returns:
        float: The area of the polygon.
    """
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
    """
    Export annotations in COCO format from a directory of JSON files for image and dir modes

    Args:
        target_directory (str): The directory containing the JSON files (dir)
        save_path (str): The path to save the output file (image mode)
        annotation_path (str): The path to the output file.

    Returns:
        str: The path to the output file.

    Raises:
        ValueError: If no JSON files are found in the directory.

    """
    # If the target is not a directory, set the file path to the save path
    if target_directory == "":
        image_mode = True
    else:
        image_mode = False


    # Create a dictionary to store the file info
    file = {}

    # Write the info header
    file["info"] = {
        "description": "Exported from DLTA-AI",
        # "url": "n/a",
        # "version": "n/a",
        "year": datetime.datetime.now().year,
        # "contributor": "n/a",
        "date_created": datetime.date.today().strftime("%Y/%m/%d")
    }

    # Create an empty set to store the used classes
    used_classes = set()

    # Get all the JSON files in the specified directory
    json_paths = glob.glob(f"{target_directory}/*.json")

    if image_mode:
        json_paths = [save_path]

    # Create empty lists to store annotations and images
    annotations = []
    images = []

    # Raise an error if no JSON files are found in the directory
    if len(json_paths) == 0:
        raise ValueError("No json files found in the directory")

    # Loop through each JSON file
    for i in range(len(json_paths)):
        try:
            with open(json_paths[i]) as f:
                # Load the JSON data
                data = json.load(f)

                # Add image data to the images list
                images.append({
                    "id": i,
                    "width": data["imageWidth"],
                    "height": data["imageHeight"],
                    "file_name": json_paths[i].split("/")[-1].replace(".json", ".jpg"),
                })

                # Loop through each shape in the JSON data
                for j in range(len(data["shapes"])):
                    # Skip shapes with no points
                    if len(data["shapes"][j]["points"],) == 0:
                        continue

                    # Add the class to the used_classes set if it hasn't been added yet
                    if data["shapes"][j]["label"].lower() not in coco_classes:
                        print(f"{data['shapes'][j]['label']} is not a valid COCO class.. Adding it to the list.")
                        coco_classes.append((data["shapes"][j]["label"].lower()))

                    # Add annotation data to the annotations list
                    annotations.append({
                        "id": len(annotations),
                        "image_id": i,
                        "category_id": coco_classes.index(data["shapes"][j]["label"].lower()) + 1,
                        "bbox": get_bbox(data["shapes"][j]["points"]),
                        "iscrowd": 0
                    })

                    # Try to add segmentation and area data to the annotation
                    try:
                        annotations[-1]["segmentation"] = [data["shapes"][j]["points"]]
                        annotations[-1]["area"] = get_area_from_polygon(
                            annotations[-1]["segmentation"][0], mode="segmentation")
                    except:
                        annotations[-1]["area"] = get_area_from_polygon(
                            annotations[-1]["bbox"], mode="bbox")

                    # Try to add score data to the annotation
                    try:
                        annotations[-1]["score"] = float(data["shapes"][j]["content"])
                    except:
                        pass

                    # Add the class to the used_classes set
                    used_classes.add(coco_classes.index(data["shapes"][j]["label"].lower()) + 1)

        # If there's an error with the JSON file, print the error and continue to the next file
        except Exception as e:
            print(f"Error with {json_paths[i]}")
            print(e)
            continue

    # Sort the used_classes set and add the categories to the file dictionary
    used_classes = sorted(used_classes)
    file["categories"] = [{"id": i, "name": coco_classes[i - 1]} for i in used_classes]

    # Add the images and annotations to the file dictionary
    file["images"] = images
    file["annotations"] = annotations

    # Write the file dictionary to the output file in JSON format
    with open(annotation_path, 'w') as outfile:
        json.dump(file, outfile, indent=4)

    # Return the path to the output file
    return annotation_path


def exportCOCOvid(results_file, vid_width, vid_height, annotation_path):
    """
    Export object detection results in COCO format for a video.

    Args:
        results_file (str): Path to the JSON file containing the object detection results.
        vid_width (int): Width of the video frames.
        vid_height (int): Height of the video frames.
        annotation_path (str): Path to the output COCO annotation file.

    Returns:
        str: Path to the output COCO annotation file.

    Raises:
        ValueError: If no object detection results are found in the JSON file.

    """
    file = {}
    file["info"] = {
        "description": "Exported from DLTA-AI",
        # "url": "n/a",
        # "version": "n/a",
        "year": datetime.datetime.now().year,
        # "contributor": "n/a",
        "date_created": datetime.date.today().strftime("%Y/%m/%d")
    }

    annotations = []
    images = []

    # Create an empty set to store the used classes
    used_classes = set()

    # Open the results file and load the JSON data
    with open(results_file) as f:
        data = json.load(f)

        # Loop through each frame in the JSON data
        for frame in data:
            # Skip frames with no object detection results
            if len(frame["frame_data"]) == 0:
                continue

            # Add image data to the images list
            images.append({
                "id": frame["frame_idx"],
                "width": vid_width,
                "height": vid_height,
                "file_name": f"frame {frame['frame_idx']}",
            })

            # Loop through each object in the frame
            for object in frame["frame_data"]:
                # Add annotation data to the annotations list
                annotations.append({
                    "id": len(annotations),
                    "image_id": frame["frame_idx"],
                    "category_id": object["class_id"] + 1,
                    "iscrowd": 0
                })

                # If the category ID is 0, add the class to the coco_classes list and update the category ID
                if annotations[-1]["category_id"] == 0:
                    coco_classes.append(object["class_name"].lower())
                    annotations[-1]["category_id"] = coco_classes.index(object["class_name"].lower()) + 1

                # Try to add the object's segmentation data to the annotation
                try:
                    annotations[-1]["bbox"] = get_bbox(object["segment"])
                    annotations[-1]["segmentation"] = [
                        [val for sublist in object["segment"] for val in sublist]]
                    annotations[-1]["area"] = get_area_from_polygon(
                        annotations[-1]["segmentation"][0], mode="segmentation")
                except:
                    # If the segmentation data is not available, use the object's bounding box data instead
                    annotations[-1]["bbox"] = object["bbox"]
                    annotations[-1]["area"] = get_area_from_polygon(
                        annotations[-1]["bbox"], mode="bbox")

                # Try to add the object's confidence score to the annotation
                try:
                    annotations[-1]["score"] = float(object["confidence"])
                except:
                    pass

                # Add the category ID to the used_classes set
                used_classes.add(annotations[-1]["category_id"])

    # Sort the used_classes set and add the categories to the file dictionary
    used_classes = sorted(list(used_classes))
    file["categories"] = [{"id": i, "name": coco_classes[i - 1]} for i in used_classes]

    # Add the images and annotations to the file dictionary
    file["images"] = images
    file["annotations"] = annotations

    # Write the file dictionary to the output file in JSON format
    with open(annotation_path, 'w') as outfile:
        json.dump(file, outfile, indent=4)

    # Return the path to the output file
    return annotation_path


def exportMOT(results_file, annotation_path):
    """
    Export object tracking results in MOT format.

    Args:
        results_file (str): Path to the JSON file containing the object tracking results.
        annotation_path (str): Path to the output MOT annotation file.

    Returns:
        str: Path to the output MOT annotation file.

    """

    # Open the results file and load the JSON data
    with open(results_file) as f, open(annotation_path, 'w') as outfile:
        # Loop through each frame in the JSON data
        for frame in json.load(f):
            for object in frame["frame_data"]:
                # Write the object tracking data to the output file
                outfile.write(f'{frame["frame_idx"]}, {object["tracker_id"]},  {object["bbox"][0]},  {object["bbox"][1]},  {object["bbox"][2]},  {object["bbox"][3]},  {object["confidence"]}, {object["class_id"] + 1}, 1\n')

    # Return the path to the output file
    return annotation_path


class FolderDialog(QFileDialog):
    """
    A custom file dialog that allows the user to save a file with a default file name and format.

    Args:
        default_file_name (str): The default file name to use.
        default_format (str): The default file format to use.

    """

    def __init__(self, default_file_name, default_format):
        """
        Initializes the FolderDialog object.

        Args:
            default_file_name (str): The default file name to use.
            default_format (str): The default file format to use.

        """

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
