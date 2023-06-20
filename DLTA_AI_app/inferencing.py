
import copy
from supervision.detection.core import Detections
from time import time
import torch
from mmdet.apis import inference_detector, init_detector, async_inference_detector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import warnings
import random
from ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
import skimage.measure

warnings.filterwarnings("ignore")


class models_inference():
    def __init__(self):
        self.annotating_models = {}

    def get_bbox(self, segmentation):
        x = []
        y = []
        for i in range(len(segmentation)):
            x.append(segmentation[i][0])
            y.append(segmentation[i][1])
        # get the bbox in xyxy format
        bbox = [min(x), min(y), max(x), max(y)]
        return bbox

    # def addPoints(self, shape, n):
    #     res = shape.copy()
    #     sub = 1.0 * n / (len(shape) - 1)
    #     if sub == 0:
    #         return res
    #     if sub < 1:
    #         res = []
    #         res.append(shape[0])
    #         flag = True
    #         for i in range(len(shape) - 1):
    #             dif = [shape[i + 1][0] - shape[i][0],
    #                    shape[i + 1][1] - shape[i][1]]
    #             newPoint = [shape[i][0] + dif[0] *
    #                         0.5, shape[i][1] + dif[1] * 0.5]
    #             if flag:
    #                 res.append(newPoint)
    #             res.append(shape[i + 1])
    #             n -= 1
    #             if n == 0:
    #                 flag = False
    #         return res
    #     else:
    #         now = int(sub) + 1
    #         res = []
    #         res.append(shape[0])
    #         for i in range(len(shape) - 1):
    #             dif = [shape[i + 1][0] - shape[i][0],
    #                    shape[i + 1][1] - shape[i][1]]
    #             for j in range(1, now):
    #                 newPoint = [shape[i][0] + dif[0] * j /
    #                             now, shape[i][1] + dif[1] * j / now]
    #                 res.append(newPoint)
    #             res.append(shape[i + 1])
    #         return self.addPoints(res, n + len(shape) - len(res))

    # def reducePoints(self, polygon, n):
    #     if n >= len(polygon):
    #         return polygon
    #     distances = polygon.copy()
    #     for i in range(len(polygon)):
    #         mid = (np.array(polygon[i - 1]) +
    #                np.array(polygon[(i + 1) % len(polygon)])) / 2
    #         dif = np.array(polygon[i]) - mid
    #         dist_mid = np.sqrt(dif[0] * dif[0] + dif[1] * dif[1])

    #         dif_right = np.array(
    #             polygon[(i + 1) % len(polygon)]) - np.array(polygon[i])
    #         dist_right = np.sqrt(
    #             dif_right[0] * dif_right[0] + dif_right[1] * dif_right[1])

    #         dif_left = np.array(polygon[i - 1]) - np.array(polygon[i])
    #         dist_left = np.sqrt(
    #             dif_left[0] * dif_left[0] + dif_left[1] * dif_left[1])

    #         distances[i] = min(dist_mid, dist_right, dist_left)
    #     distances = [distances[i] + random.random()
    #                  for i in range(len(distances))]
    #     ratio = 1.0 * n / len(polygon)
    #     threshold = np.percentile(distances, 100 - ratio * 100)

    #     i = 0
    #     while i < len(polygon):
    #         if distances[i] < threshold:
    #             polygon[i] = None
    #             i += 1
    #         i += 1
    #     res = [x for x in polygon if x is not None]
    #     return self.reducePoints(res, n)

    # def handlePoints(self, polygon, n):
    #     if n == len(polygon):
    #         return polygon
    #     elif n > len(polygon):
    #         return self.addPoints(polygon, n - len(polygon))
    #     else:
    #         return self.reducePoints(polygon, n)

    # def interpolate_polygon(self , polygon, n_points):
    #     # interpolate polygon to get less points
    #     polygon = np.array(polygon)
    #     if len(polygon) < 25:
    #         return polygon
    #     return np.array(self.reducePoints(polygon.tolist() , n_points))
    #     x = polygon[:, 0]
    #     y = polygon[:, 1]
    #     tck, u = splprep([x, y], s=0, per=1)
    #     u_new = np.linspace(u.min(), u.max(), n_points)
    #     x_new, y_new = splev(u_new, tck, der=0)
    #     return np.array([x_new, y_new]).T

    # # function to masks into polygons
    # def mask_to_polygons(self , mask, n_points=25):
    #     # Find contours
    #     contours = cv2.findContours(
    #         mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    #     # Convert contours to polygons
    #     polygon = []
    #     for contour in contours:
    #         contour = contour.flatten().tolist()
    #         # Remove last point if it is the same as the first
    #         if contour[-2:] == contour[:2]:
    #             contour = contour[:-2]
    #         polygon.append(contour)
    #     polygon = [(polygon[0][i], polygon[0][i + 1])
    #                for i in np.arange(0, len(polygon[0]), 2)]
    #     polygon = self.interpolate_polygon(polygon, n_points)
    #     polygon = [[int(sublist[0]) , int(sublist[1])] for sublist in polygon ]

    #     return polygon

    def get_contour_length(self, contour):
        contour_start = contour
        contour_end = np.r_[contour[1:], contour[0:1]]
        return np.linalg.norm(contour_end - contour_start, axis=1).sum()

    def mask_to_polygons(self, mask, n_points=25, resize_factors=[1.0, 1.0]):
        mask = mask > 0.0
        contours = skimage.measure.find_contours(mask)
        if len(contours) == 0:
            return []
        contour = max(contours, key=self.get_contour_length)
        coords = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() / 100,
        )

        coords = coords * resize_factors
        # convert coords from x y to y x
        coords = np.fliplr(coords)

        # segment_points are a list of coords
        segment_points = coords.astype(int)
        polygon = segment_points
        return polygon

    def full_points(bbox):
        return np.array([[bbox[0], bbox[1]], [bbox[0], bbox[3]], [bbox[2], bbox[3]], [bbox[2], bbox[1]]])

    @torch.no_grad()
    def decode_file(self, img, model, classdict, threshold=0.3, img_array_flag=False):

        if model.__class__.__name__ == "YOLO":  
            if isinstance(img, str):
                img = cv2.imread(img)

            # get image size
            img_resized = cv2.resize (img , (640, 640))
            # default yolo arguments from yolov8 tracking repo
                # imgsz=(640, 640),  # inference size (height, width)
                # conf_thres=0.25,  # confidence threshold
                # iou_thres=0.45,  # NMS IOU threshold
                # max_det=1000,  # maximum detections per image
            results = model(img_resized , conf = 0.25 , iou=  0.45 , verbose = False)
            results = results[0]
            # if len results is 0 then return empty dict
            if results.masks is None:
                return {"results": {}}

            masks = results.masks.cpu().numpy().masks
            masks = masks > 0.0
            org_size = img.shape[:2]
            out_size = masks.shape[1:]

            # print(f'org_size : {org_size} , out_size : {out_size}')

            # convert boxes to original image size same as the masks (coords = coords * org_size / out_size)
            boxes = results.boxes.xyxy.cpu().numpy()
            boxes = boxes * np.array([org_size[1] / out_size[1], org_size[0] /
                                     out_size[0], org_size[1] / out_size[1], org_size[0] / out_size[0]])

            detections = Detections(
                xyxy=boxes,
                confidence=results.boxes.conf.cpu().numpy(),
                class_id=results.boxes.cls.cpu().numpy().astype(int)
            )

            polygons = []
            result_dict = {}

            resize_factors = [org_size[0] / out_size[0] , org_size[1] / out_size[1]]
            if len(masks) == 0:
                return {"results":{}}
            for mask in masks:
                polygon = self.mask_to_polygons(
                    mask, resize_factors=resize_factors)
                polygons.append(polygon)

            # detection is a tuple of  (box, confidence, class_id, tracker_id)
            ind = 0
            res_list = []
            for detection in detections:
                if round(detection[1], 2) < float(threshold):
                    continue
                result = {}
                result["class"] = classdict.get(int(detection[2]))
                result["confidence"] = str(round(detection[1], 2))
                result["bbox"] = detection[0].astype(int)
                result["seg"] = polygons[ind]
                ind += 1
                if result["class"] == None:
                    continue
                if len(result["seg"]) < 3:
                    continue

                res_list.append(result)
            result_dict["results"] = res_list
            return result_dict

        if img_array_flag:
            results = inference_detector(model, img)
        else:
            results = inference_detector(model, plt.imread(img))
        # results = async_inference_detector(model, plt.imread(img_path))
        torch.cuda.empty_cache()

        results0 = []
        results1 = []
        for i in classdict.keys():
            mask = results[0][i][:, 4] >= float(threshold)
            results0.append(results[0][i][mask])
            results1.append(list(np.array(results[1][i])[mask]))

        # for i in classdict.keys():
        #     results0.append(results[0][i])
        #     results1.append(results[1][i])

        # self.annotating_models[model.__class__.__name__] = [results0 , results1]
        # print(self.annotating_models.keys())

        # # if the length of the annotating_models is greater than 1 we need to merge the masks
        # if len(self.annotating_models.keys()) > 1:
        #     print("merging masks")
        #     results0,results1 =  self.merge_masks()

        #     assert len(results0) == len(results1)
        #     for i in range(len(results0)):
        #         assert len(results0[i]) == len(results1[i])
        return results0, results1

    def polegonise(self, results0, results1, classdict, threshold=0.3, show_bbox_flag=False):
        result_dict = {}
        res_list = []

        self.classes_numbering = [keyno for keyno in classdict.keys()]
        # print(self.classes_numbering)
        for classno in range(len(results0)):
            for instance in range(len(results0[classno])):
                if float(results0[classno][instance][-1]) < float(threshold):
                    continue
                result = {}
                result["class"] = classdict.get(
                    self.classes_numbering[classno])
                # Confidence
                result["confidence"] = str(
                    round(results0[classno][instance][-1], 2))
                if classno == 0:
                    result["seg"] = self.mask_to_polygons(
                        results1[classno][instance].astype(np.uint8), 10)
                else:
                    result["seg"] = self.mask_to_polygons(
                        results1[classno][instance].astype(np.uint8), 25)

                # result["bbox"] = self.get_bbox(result["seg"])
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
                    pass

                if result["class"] == None:
                    continue
                if len(result["seg"]) < 3:
                    continue
                res_list.append(result)

        result_dict["results"] = res_list
        return result_dict

    def merge_masks(self):
        tic = time()
        result0 = []
        result1 = []

        # Counting for debugging purposes
        # count the number of instances in each model
        counts = count_instances(self.annotating_models)
        # print the counts of each model
        for model in counts.keys():
            print("model {} has {} instances".format(model, counts[model]))

        # the following lines can be used if we use models with different number of classes
        # classnos = []
        # for model in self.annotating_models.keys():
        #     classnos.append(len(self.annotating_models[model][1]))
        # print(classnos)

        # instead the following line of code will be used if we use models with the same number of classes
        classnos = len(self.annotating_models[list(
            self.annotating_models.keys())[0]][1])

        merged_counts = 0
        # initialize the result list with the same number of classes as the model with the most classes
        for i in range(classnos):
            result1.append([])
            result0.append([])

        # deep copy the annotating_models dict to pop all the masks we have merged (try delete it for future optimisation)
        annotating_models_copy = copy.deepcopy(self.annotating_models)
        # merge masks of the same class
        for idx1, model in enumerate(self.annotating_models.keys()):
            for classno in range(len(self.annotating_models[model][1])):
                # check if an instance exists in the model in this class
                if len(self.annotating_models[model][1][classno]) > 0:
                    for instance in range(len(self.annotating_models[model][1][classno])):
                        for idx2, model2 in enumerate(self.annotating_models.keys()):
                            if model != model2 and idx2 > idx1:
                                # print(type(annotating_models_copy[model][0][classno]),type(annotating_models_copy[model2][0][classno]))
                                # check if the class exists in the other model
                                if classno in range(len(self.annotating_models[model2][1])):
                                    # check if an instance exists in the other model
                                    if len(self.annotating_models[model2][1][classno]) > 0:
                                        for instance2 in range(len(self.annotating_models[model2][1][classno])):
                                            dirty = False
                                            # print('checking class ' + str(classno)  ' of models ' + model + str(idx1) +  ' and ' + model2 + str(idx2))
                                            # get the intersection percentage of the two masks
                                            intersection = np.logical_and(
                                                self.annotating_models[model][1][classno][instance], self.annotating_models[model2][1][classno][instance2])
                                            intersection = np.sum(intersection)
                                            union = np.logical_or(
                                                self.annotating_models[model][1][classno][instance], self.annotating_models[model2][1][classno][instance2])
                                            union = np.sum(union)
                                            iou = intersection / union
                                            # print('iou of class ' + str(classno) + ' instance ' + str(instance) + ' and instance ' + str(instance2) + ' is ' + str(iou))
                                            if iou > 0.5:
                                                if (annotating_models_copy[model][1][classno][instance] is None) or (annotating_models_copy[model2][1][classno][instance2] is None):
                                                    dirty = True
                                                if dirty == False:
                                                    # merge their bboxes and store the result in result0
                                                    bbox1 = self.annotating_models[model][0][classno][instance]
                                                    bbox2 = self.annotating_models[model2][0][classno][instance2]
                                                    bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]), max(
                                                        bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3]), max(bbox1[4], bbox2[4])]
                                                    result0[classno].append(
                                                        bbox)
                                                    # store the merged mask in result1
                                                    result1[classno].append(np.logical_or(
                                                        self.annotating_models[model][1][classno][instance], self.annotating_models[model2][1][classno][instance2]))
                                                    # print('merging masks of class ' + str(classno) + ' instance ' + str(instance) + ' and instance ' + str(instance2) + ' of models ' + model + ' and ' + model2)
                                                    merged_counts += 1
                                                # remove the mask from both models
                                                annotating_models_copy[model][1][classno][instance] = None
                                                annotating_models_copy[model2][1][classno][instance2] = None
                                                annotating_models_copy[model][0][classno][instance] = None
                                                annotating_models_copy[model2][0][classno][instance2] = None
                                                # continue to the next instance of the first model
                                                break

        counts_here = {}
        # add the remaining masks to the result
        for model in annotating_models_copy.keys():
            counts_here[model] = 0
            for classno in range(len(annotating_models_copy[model][1])):
                for instance in range(len(annotating_models_copy[model][1][classno])):
                    if annotating_models_copy[model][1][classno][instance] is not None:
                        counts_here[model] += 1
                        # print('adding mask of class ' + str(classno) + ' instance ' + str(instance) + ' of model ' + model)
                        result1[classno].append(
                            annotating_models_copy[model][1][classno][instance])
                        result0[classno].append(
                            annotating_models_copy[model][0][classno][instance])
        # clear the annotating_models and add the result to it
        self.annotating_models = {}
        # self.annotating_models["merged"] = [result0 , result1]
        for model in counts_here.keys():
            print("model {} has {} instances".format(
                model, counts_here[model]))
        print("merged {} instances".format(merged_counts))
        tac = time()
        print("merging took {} ms".format((tac - tic) * 1000))
        return result0, result1


# result will have ---> bbox , confidence , class_id , tracker_id , segment
# result of the detection phase only should be (bbox , confidence , class_id , segment)
def count_instances(annotating_models):
    # separate the counts for each model
    counts = {}
    for model in annotating_models.keys():
        counts[model] = 0
        for classno in range(len(annotating_models[model][1])):
            counts[model] += len(annotating_models[model][1][classno])
    return counts
