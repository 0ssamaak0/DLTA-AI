from dbr import *

import torch
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import cv2
import numpy as np
import matplotlib.pyplot as plt

class DynamsoftBarcodeReader():
    def __init__(self):
        self.dbr = BarcodeReader()
        self.dbr.init_license("t0068MgAAABaPdihgo0ura46bBvXa/K+sCfupbVhYdDSY3AlEooBX/7ZSvLQVJmCnYzaJ8Xblhwt1G3hrI9hrklQDGgzvFp0=")

    def decode_file(self, img_path, engine=""):
        config = "/mnt/c/Graduation Project/Auto Annotation Tool/mmdetection/configs/detectors/htc_r50_sac_1x_coco.py"
        checkpoint = "/mnt/c/Graduation Project/Auto Annotation Tool/mmdetection/checkpoints/htc_r50_sac_1x_coco-bfa60c54.pth"

        model = init_detector(config, checkpoint, device = torch.device("cuda"))

        torch.cuda.empty_cache()

        results = inference_detector(model, plt.imread(img_path))
        results = results[0]
        results = [results[0], results[2],results[3],results[5],results[7]]
        result_dict = {}
        res_list = []

        def full_points(bbox):
            return np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])

        classdict = {0:"person", 1:"car", 2:"motorcycle", 3:"bus", 4:"truck"}
        for classno in range(len(results)):
            for instance in range(len(results[classno])):
                result = {}
                result["class"] = classdict[classno]
                result["confidence"] = results[classno][instance][-1]
                result["bbox"] = results[classno][instance][:-1]
                points = full_points(result["bbox"])
                result["x1"] = points[0][0]
                result["y1"] = points[0][1]
                result["x2"] = points[1][0]
                result["y2"] = points[1][1]
                result["x3"] = points[2][0]
                result["y3"] = points[2][1]
                result["x4"] = points[3][0]
                result["y4"] = points[3][1]
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

    