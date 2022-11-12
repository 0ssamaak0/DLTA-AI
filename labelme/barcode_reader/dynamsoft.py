from dbr import *

import torch
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def interpolate_polygon(polygon, n_points):
    # interpolate polygon to get less points
    polygon = np.array(polygon)
    if len(polygon) < 20:
        return polygon
    x = polygon[:, 0]
    y = polygon[:, 1]
    tck, u = splprep([x, y], s=0, per=1)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)
    return np.array([x_new, y_new]).T

# function to masks into polygons
def mask_to_polygons(mask, n_points=20):
    # Find contours
    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # Convert contours to polygons
    polygon = []
    for contour in contours:
        contour = contour.flatten().tolist()
        # Remove last point if it is the same as the first
        if contour[-2:] == contour[:2]:
            contour = contour[:-2]
        polygon.append(contour)
    polygon = [(polygon[0][i], polygon[0][i + 1]) for i in np.arange(0, len(polygon[0]), 2)]
    polygon = interpolate_polygon(polygon, n_points)
    return polygon

    

class DynamsoftBarcodeReader():
    def __init__(self):
        self.dbr = BarcodeReader()
        self.dbr.init_license("t0068MgAAABaPdihgo0ura46bBvXa/K+sCfupbVhYdDSY3AlEooBX/7ZSvLQVJmCnYzaJ8Xblhwt1G3hrI9hrklQDGgzvFp0=")

    def decode_file(self, img_path, engine="", threshold=0.5):
        config = "/mnt/c/Graduation Project/Auto Annotation Tool/mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py"
        checkpoint = "/mnt/c/Graduation Project/Auto Annotation Tool/mmdetection/checkpoints/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth"

        model = init_detector(config, checkpoint, device = torch.device("cuda"))

        torch.cuda.empty_cache()

        results = inference_detector(model, plt.imread(img_path))
        results0 = [results[0][0], results[0][2],results[0][3],results[0][5],results[0][7]]
        results1 = [results[1][0], results[1][2],results[1][3],results[1][5],results[1][7]]
        result_dict = {}
        res_list = []

        # def full_points(bbox):
        #     return np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])

        classdict = {0:"person", 1:"car", 2:"motorcycle", 3:"bus", 4:"truck"}
        for classno in range(len(results0)):
            for instance in range(len(results0[classno])):
                if float(results0[classno][instance][-1]) < float(threshold):
                    continue
                result = {}
                result["class"] = classdict[classno]
                # Confidence
                result["confidence"] = str(results0[classno][instance][-1])
                result["bbox"] = results0[classno][instance][:-1]
                result["seg"] = mask_to_polygons(results1[classno][instance].astype(np.uint8))

                # points = full_points(result["bbox"])
                # result["x1"] = points[0][0]
                # result["y1"] = points[0][1]
                # result["x2"] = points[1][0]
                # result["y2"] = points[1][1]
                # result["x3"] = points[2][0]
                # result["y3"] = points[2][1]
                # result["x4"] = points[3][0]
                # result["y4"] = points[3][1]
                res_list.append(result)

        result_dict["results"] = res_list
        return result_dict
        
if __name__ == '__main__':
    import time
    reader = DynamsoftBarcodeReader()
    start_time = time.time()
    results = reader.decode_file("image045.jpg")
    end_time = time.time()
    elapsedTime = int((end_time - start_time) * 1000)

    