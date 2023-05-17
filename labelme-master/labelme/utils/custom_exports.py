# Don't Modify These Lines
# =========================================
custom_exports_list = []

# custom export class blueprint
class CustomExport:
    def __init__(self, file_name, button_name, format, function):
        self.file_name = file_name
        self.button_name = button_name
        self.format = format
        self.function = function
    
    def __call__(self, *args):
        return self.function(*args)
    
# =========================================

# Add your functions here
# =========================================
import csv
import json
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

# define custom functions
def exportTrajectory(results_file, vid_width, vid_height, annotation_path, rle = False, use_seg = True):
    rows = []
    headers = ["id", "frame", "class", "conf", "center_point", "segmentation"]

    with open(results_file) as f:
        data = json.load(f)
        for frame in data:
            for object in frame["frame_data"]:
                tracker_id = object["tracker_id"]
                class_id = object["class_id"] + 1
                confidence = object["confidence"]
                frame_idx = frame["frame_idx"]
                segmentation = [val for sublist in object["segment"] for val in sublist]
                center_point = center_of_polygon(segmentation)
                if rle:
                    from pycocotools import mask as maskUtils
                    segmentation = [segmentation]
                    segmentation = maskUtils.frPyObjects(segmentation, vid_height, vid_width)[0]["counts"]
                if use_seg:
                    rows.append([tracker_id, frame_idx, class_id, confidence, center_point, segmentation])
                else:
                    rows.append([tracker_id, frame_idx, class_id, confidence, center_point])

    # sort rows by tracker id then by frame idx
    rows.sort(key=lambda x: (x[0], x[3]))
    with open(annotation_path, 'w', newline = "") as outfile:
        # write the header using the csv writer
        writer = csv.writer(outfile)
        writer.writerow(headers)
        # write each row in the file using the csv writer
        writer.writerows(rows)
    
    return annotation_path

def exportTrajectoryCompressed(results_file, vid_width, vid_height, annotation_path):
    return exportTrajectory(results_file, vid_width, vid_height, annotation_path, rle = True)

def exportTrajectoryNoSegmentation(results_file, vid_width, vid_height, annotation_path):
    return exportTrajectory(results_file, vid_width, vid_height, annotation_path, use_seg = False)

# =========================================

# append the custom exports to the list as new CustomExport objects
 
custom_exports_list.append(CustomExport("traj_full", "Customized Trajectory Format (Tracking)", "csv", exportTrajectory))
custom_exports_list.append(CustomExport("traj_compressed", "Compressed Customized Trajectory Format (Tracking)", "csv", exportTrajectoryCompressed))
custom_exports_list.append(CustomExport("traj_no_seg", "Customized Trajectory Format No Segmentation (Tracking)", "csv", exportTrajectoryNoSegmentation))
