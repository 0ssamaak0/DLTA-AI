
from supervision.detection.core import Detections
import supervision as sv
from typing import List
from time import time
import sys
import torch
from mmdet.apis import inference_detector, init_detector , async_inference_detector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import json
import warnings
warnings.filterwarnings("ignore")


class models_inference():

    def interpolate_polygon(self , polygon, n_points):
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
    def mask_to_polygons(self , mask, n_points=20):
        # Find contours
        contours = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        # Convert contours to polygons
        polygon = []
        for contour in contours:
            contour = contour.flatten().tolist()
            # Remove last point if it is the same as the first
            if contour[-2:] == contour[:2]:
                contour = contour[:-2]
            polygon.append(contour)
        polygon = [(polygon[0][i], polygon[0][i + 1])
                   for i in np.arange(0, len(polygon[0]), 2)]
        polygon = self.interpolate_polygon(polygon, n_points)
        return polygon

    # experimental


<< << << < HEAD
    def full_points(bbox):
        return np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])

    def decode_file(self, img , model, classdict, threshold=0.3, img_array_flag=False , show_bbox_flag=False):

        if model.__class__.__name__ == "YOLO":
            results = model(img , conf=float(threshold))
            results = results[0]
            w , h = results.orig_img.shape[1] , results.orig_img.shape[0]
            segments = results.masks.segments
            detections = Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int)
            )
            polygons = []
            result_dict = {}

            for segment in segments:
                # segment is a array of points that make up the polygon of the mask and are set relative to the image size so we need to scale them to the image size
                for i in range(len(segment)):
                    segment[i][0] = segment[i][0] * w
                    segment[i][1] = segment[i][1] * h
                segment_points = self.interpolate_polygon(segment , 20)
                polygons.append(segment_points)

            # detection is a tuple of  (box, confidence, class_id, tracker_id)

            ind = 0
            res_list = []

            for detection in detections:

                result = {}
                result["class"] = classdict.get(int(detection[2]))
                result["confidence"] = str(detection[1])
                result["bbox"] = detection[0].astype(np.uint8)
                result["seg"] = polygons[ind]
                ind += 1
                res_list.append(result)
            result_dict["results"] = res_list
            return result_dict


== == == =
    def decode_file(self, img , model, classdict, threshold=0.3, img_array_flag=False):
        # def decode_file(self, img_path, threshold=0.3 , selected_model_name=""):
        # if img_path is none img_path = "test_img_1.webp"

        # with open("saved_models.json") as json_file:
        #     data = json.load(json_file)
        #     if selected_model_name == "":
        #     # read the saved_models.json file and import the config and checkpoint files from the first model
        #         selected_model_name = list(data.keys())[0]
        #         config = data[selected_model_name]["config"]
        #         checkpoint = data[selected_model_name]["checkpoint"]
        #     else:
        #         config = data[selected_model_name]["config"]
        #         checkpoint = data[selected_model_name]["checkpoint"]
        #     # print(f'selected model : {selected_model_name} \n config : {config} \n\
        #     #     checkpoint : {checkpoint} \n')
        print(classdict)

        # torch.cuda.empty_cache()

        # model = init_detector(config, checkpoint, device = torch.device("cuda"))
>>>>>> > e6bbd74e815d8f12d6fe18fe5c16b6059fb5c832
        if img_array_flag:
            results = inference_detector(model, img)
        else:
            results = inference_detector(model, plt.imread(img))
        # results = async_inference_detector(model, plt.imread(img_path))
<< << << < HEAD
        torch.cuda.empty_cache()


== == == =

        # del model
        torch.cuda.empty_cache()

        # print(f'ram used : {torch.cuda.memory_allocated() / 1024 ** 3} GB')
        # print(f'ram cached : {torch.cuda.memory_cached() / 1024 ** 3} GB')
        # print(f'what is using the ram : {torch.cuda.memory_summary()}')

        # results0 = [results[0][0], results[0][2],results[0][3],results[0][5],results[0][7]]
        # results1 = [results[1][0], results[1][2],results[1][3],results[1][5],results[1][7]]

>>>>>> > e6bbd74e815d8f12d6fe18fe5c16b6059fb5c832
        results0 = []
        results1 = []
        for i in classdict.keys():
            results0.append(results[0][i])
            results1.append(results[1][i])

        result_dict = {}
        res_list = []
<< << << < HEAD


== == == =

        def full_points(bbox):
            return np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])

>>>>>> > e6bbd74e815d8f12d6fe18fe5c16b6059fb5c832
        # classdict = {0:"person", 1:"car", 2:"motorcycle", 3:"bus", 4:"truck"}
        classes_numbering = [keyno for keyno in classdict.keys()]
        for classno in range(len(results0)):
            for instance in range(len(results0[classno])):
                if float(results0[classno][instance][-1]) < float(threshold):
                    continue
                result = {}
                result["class"] = classdict.get(classes_numbering[classno])
                # Confidence
                result["confidence"] = str(results0[classno][instance][-1])
                result["bbox"] = results0[classno][instance][:-
                                                             1].astype(np.uint8)
                if classno == 0:
<< << << < HEAD
                    result["seg"] = self.mask_to_polygons(
                        results1[classno][instance].astype(np.uint8) , 10)
                else : 
                    result["seg"] = self.mask_to_polygons(results1[classno][instance].astype(np.uint8) , 20)
                if show_bbox_flag:
                    # result["bbox"] = full_points(result["bbox"]).tolist()
                    # points = full_points(result["bbox"])
                    # result["x1"] = points[0][0]
                    # result["y1"] = points[0][1]
                    # result["x2"] = points[1][0]
                    # result["y2"] = points[1][1]
                    # result["x3"] = points[2][0]
                    # result["y3"] = points[2][1]
                    # result["x4"] = points[3][0]
                    # result["y4"] = points[3][1]
                    x = 30  # nothing
=======
                    result["seg"] = self.mask_to_polygons(
                        results1[classno][instance].astype(np.uint8) , 10)
                else :
                    result["seg"] = self.mask_to_polygons(
                        results1[classno][instance].astype(np.uint8))
                # result["bbox"] = full_points(result["bbox"]).tolist()
                # points = full_points(result["bbox"])
                # result["x1"] = points[0][0]
                # result["y1"] = points[0][1]
                # result["x2"] = points[1][0]
                # result["y2"] = points[1][1]
                # result["x3"] = points[2][0]
                # result["y3"] = points[2][1]
                # result["x4"] = points[3][0]
                # result["y4"] = points[3][1]
>>>>>>> e6bbd74e815d8f12d6fe18fe5c16b6059fb5c832
                res_list.append(result)

        result_dict["results"] = res_list
        return result_dict


# result will have ---> bbox , confidence , class_id , tracker_id , segment  
# result of the detection phase only should be (bbox , confidence , class_id , segment)