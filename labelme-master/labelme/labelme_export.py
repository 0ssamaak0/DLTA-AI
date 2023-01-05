import json
import datetime
import cv2
import glob
import os

# Get bounding box from segmentation
def get_bbox(segmentation):
    x = []
    y = []
    for i in range(len(segmentation)):
        if i % 2 == 0:
            x.append(segmentation[i])
        else:
            y.append(segmentation[i])
    return [min(x), min(y), max(x) - min(x), max(y) - min(y)]

# Run length encoding
def RLE(segmentation):
    '''
    Converts a list of points into a run length encoding
    '''
    rle = []
    for i in range(len(segmentation)):
        if i % 2 == 0:
            rle.append(segmentation[i])
        else:
            rle.append(segmentation[i] - segmentation[i-1])
    return rle

# Export from labelme to COCO format
def export_coco(file_path, output_name):
    '''
    Exports the annotations in the Labelmm format to COCO format
    file_path: path to the directory containing the images and json files
    output_name: name of the output file
    '''
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
    file["info"] =  {
            "description": "Exported from Labelmm",
            # "url": "n/a",
            # "version": "n/a",
            "year": datetime.datetime.now().year,
            # "contributor": "n/a",
            "date_created": datetime.date.today().strftime("%Y/%m/%d")
        }


    # write list of COCO (custmn) categories
    file["categories"] = [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person"
            },
            {
                "id": 2,
                "name": "bicycle",
                "supercategory": "vehicle"
            },
            {
                "id": 3,
                "name": "car",
                "supercategory": "vehicle"
            },
            {
                "id": 4,
                "name": "motorcycle",
                "supercategory": "vehicle"
            },
            {
                "id": 6,
                "name": "bus",
                "supercategory": "vehicle"
            },
            {
                "id": 8,
                "name": "truck",
                "supercategory": "vehicle"
            },
    ]

    json_paths = glob.glob(f"{file_path}/*.json")
    annotations = []
    images = []
    for i in range(len(json_paths)):
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
            # annotation data
            if len(data["shapes"][j]["points"],) == 0:
                continue
            annotations.append({
                "id": len(annotations),
                "image_id": i,
                "category_id": coco_categories[data["shapes"][j]["label"]],
                "segmentation": data["shapes"][j]["points"],
                "bbox": get_bbox(data["shapes"][j]["points"]),
                "confidence": float(data["shapes"][j]["content"])
            })

    file["images"] = images
    file["annotations"] = annotations
    
    # make a directory under file_path called Annotations (if it doesn't exist)
    if not os.path.exists(f"{file_path}/Annotations"):
        os.makedirs(f"{file_path}/Annotations")



    # write in the output file in json, format the output to be pretty
    with open(f"{file_path}/Annotations/{output_name}.json", 'w') as outfile:
        json.dump(file, outfile, indent=4)