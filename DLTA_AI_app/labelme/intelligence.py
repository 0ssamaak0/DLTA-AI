from ultralytics import YOLO
import json
import time
try:
    from inferencing import models_inference
except ModuleNotFoundError:
    import subprocess
    print("The required package 'mmcv-full' is not currently installed. It will now be installed. This process may take some time. Note that this package will only be installed the first time you use DLTA-AI.")
    # subprocess.run(["mim", "install", "mmcv-full==1.7.0"])
    from inferencing import models_inference
from labelme.label_file import LabelFile
from labelme import PY2
from PyQt6.QtCore import QThread
from PyQt6.QtCore import pyqtSignal as pyqtSignal
from PyQt6 import QtGui
from PyQt6 import QtWidgets
import os
import os.path as osp
import warnings
import yaml

import torch
# from mmdet.apis import init_detector
warnings.filterwarnings("ignore")

from .widgets.MsgBox import OKmsgBox
from .utils.helpers import mathOps

init_detector = None
# YOLO = None
# Model imports
from .DLTA_Model import DLTA_Model_list
# get all files under models directory
models_dir = os.path.dirname(__file__) + '/models'
model_files = [f for f in os.listdir(models_dir)]
# import all of them (e.g, from .models import YOLOv8)
for model_file in model_files:
    if model_file.endswith(".py") and model_file != "__init__.py":
        exec(f"from .models import {model_file[:-3]}")
        print(f"imported {model_file[:-3]}")
print(f'models list : {DLTA_Model_list}')



coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# make a list of 12 unique colors as we will use them to draw bounding boxes of different classes in different colors
# so the calor palette will be used to draw bounding boxes of different classes in different colors
# the color pallette should have the famous 12 colors as red, green, blue, yellow, cyan, magenta, white, black, gray, brown, pink, and orange in bgr format
color_palette = [(75, 25, 230),
                 (75, 180, 60),
                 (25, 225, 255),
                 (200, 130, 0),
                 (49, 130, 245),
                 (180, 30, 145),
                 (240, 240, 70),
                 (230, 50, 240),
                 (60, 245, 210),
                 (190, 190, 250),
                 (128, 128, 0),
                 (255, 190, 230),
                 (40, 110, 170),
                 (200, 250, 255),
                 (0, 0, 128),
                 (195, 255, 170)]


class IntelligenceWorker(QThread):
    sinOut = pyqtSignal(int, int)

    def __init__(self, parent, images, source,multi_model_flag=False):
        super(IntelligenceWorker, self).__init__(parent)
        self.parent = parent
        self.source = source
        self.images = images
        self.multi_model_flag = multi_model_flag
        self.notif = []

    def run(self):
        index = 0
        total = len(self.images)
        for filename in self.images:

            if self.parent.isVisible == False:
                return
            if self.source.operationCanceled == True:
                return
            index = index + 1
            json_name = osp.splitext(filename)[0] + ".json"
            # if os.path.exists(json_name)==False:

            if os.path.isdir(json_name):
                os.remove(json_name)

            try:
                print("Decoding "+filename)
                if self.multi_model_flag:
                    s = self.source.get_shapes_of_one(filename, multi_model_flag=True)
                else:
                    s = self.source.get_shapes_of_one(filename)
                s = mathOps.convert_shapes_to_qt_shapes(s)
                self.source.saveLabelFile(filename, s)
            except Exception as e:
                print(e)
            self.sinOut.emit(index, total)



class Intelligence():
    def __init__(self, parent):
        self.reader = models_inference()
        self.parent = parent
        self.conf_threshold = 0.3
        self.segmentation_accuracy = 3.0
        self.iou_threshold = 0.5
        with open ("labelme/config/default_config.yaml") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.default_classes = self.config["default_classes"]
        try:
            self.selectedclasses = {}
            for class_ in self.default_classes:
                if class_ in coco_classes:
                    index = coco_classes.index(class_)
                    self.selectedclasses[index] = class_
        except:
            self.selectedclasses = {i:class_ for i,class_ in enumerate(coco_classes)}
            print("error in loading the default classes from the config file, so we will use all the coco classes")
        # print(f"selected classes are {self.selectedclasses}")
        self.selectedmodels = []
        # self.current_model_name, self.current_mm_model = self.make_mm_model("")
        self.current_DLTA_model = None


    @torch.no_grad()
    def make_mm_model(self, selected_model_name):
        try:
            with open("saved_models.json") as json_file:
                data = json.load(json_file)
                if selected_model_name == "":
                    # read the saved_models.json file and import the config and checkpoint files from the first model
                    selected_model_name = list(data.keys())[0]
                    config = data[selected_model_name]["config"]
                    checkpoint = data[selected_model_name]["checkpoint"]
                else:
                    config = data[selected_model_name]["config"]
                    checkpoint = data[selected_model_name]["checkpoint"]
                print(
                    f'selected model : {selected_model_name} \nconfig : {config}\ncheckpoint : {checkpoint} \n')
        except Exception as e:
            OKmsgBox("Error", f"Error in loading the model\n{e}", "critical")
            return


        torch.cuda.empty_cache()
        if "YOLOv8" in selected_model_name:
            model = YOLO(checkpoint)
            model.fuse()
            return selected_model_name, model

        try:
            print(f"From the working one: {config}")
            model = init_detector(config,
                                  checkpoint,
                                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        except:
            print(
                "Error in loading the model, please check if the config and checkpoint files do exist")

            #    cfg_options= dict(iou_threshold=0.2))

        # "C:\Users\Shehab\Desktop\l001\ANNOTATION_TOOL\mmdetection\mmdetection\configs\yolact\yolact_r50_1x8_coco.py"
        # model = init_detector("C:/Users/Shehab/Desktop/mmdetection/mmdetection/configs/detectors/htc_r50_sac_1x_coco.py",
            # "C:/Users/Shehab/Desktop/mmdetection/models_checkpoints/htc_r50_sac_1x_coco-bfa60c54.pth", device = torch.device("cuda"))
        return selected_model_name, model

    @torch.no_grad()
    def make_DLTA_model(self, selected_model_name, model_family, config, checkpoint):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # this is do get the family of the model to initialize the correct model accordingly
        model_idx = -1
        for index, model_instance in enumerate(DLTA_Model_list):
            if model_instance.model_family == model_family:
                print(f"Match found at index {index} | {model_family}")
                model_idx = index
                break
        if model_idx == -1:
            print(f"Model family {model_family} not found")
            return
        
        self.current_DLTA_model = DLTA_Model_list[model_idx]
        
        self.current_DLTA_model.initialize(checkpoint = checkpoint , config = config )
        
        print("Running through DLTA")

        return selected_model_name

    # @ torch.no_grad()
    # def make_mm_model_more(self, selected_model_name, config, checkpoint):
    #     torch.cuda.empty_cache()
    #     print(
    #         f"Selected model is {selected_model_name}\n and config is {config}\n and checkpoint is {checkpoint}")

    #     # if YOLOv8
    #     if "YOLOv8" in selected_model_name:
    #         try:
    #             model = YOLO(checkpoint)
    #             model.fuse()
    #             return selected_model_name, model
    #         except Exception as e:
    #             OKmsgBox("Error", f"Error in loading the model\n{e}", "critical")
    #             return

    #     # It's a MMDetection model
    #     else:
    #         try:
    #             print(f"From the new one: {config}")
    #             model = init_detector(config, checkpoint, device=torch.device(
    #                 "cuda" if torch.cuda.is_available() else "cpu"))
    #         except Exception as e:
    #             OKmsgBox
    #             OKmsgBox("Error", f"Error in loading the model\n{e}", "critical")
    #             return
    #         return selected_model_name, model

    def get_shapes_of_one(self, image, img_array_flag=False, multi_model_flag=False):
        # print(f"Threshold is {self.conf_threshold}")
        # results = self.reader.decode_file(img_path = filename, threshold = self.conf_threshold , selected_model_name = self.current_model_name)["results"]
        start_time_all = time.time()
        # if img_array_flag is true then the image is a numpy array and not a path
        if multi_model_flag:
            # to handle the case of the user selecting no models
            if len(self.selectedmodels) == 0:
                return []
            self.reader.annotating_models.clear()
            for model_name in self.selectedmodels:
                self.current_model_name, self.current_mm_model = self.make_mm_model(
                    model_name)
                if img_array_flag:
                    results0, results1 = self.reader.decode_file(
                        img=image, model=self.current_mm_model, classdict=self.selectedclasses, threshold=self.conf_threshold, img_array_flag=True)
                else:
                    results0, results1 = self.reader.decode_file(
                        img=image, model=self.current_mm_model, classdict=self.selectedclasses, threshold=self.conf_threshold)
                self.reader.annotating_models[model_name] = [
                    results0, results1]
                end_time = time.time()
                print(
                    f"Time taken to annoatate img on {self.current_model_name}: {int((end_time - start_time)*1000)} ms" + "\n")
            print('merging masks')
            results0, results1 = self.reader.merge_masks()
            results = self.reader.polegonise(
                results0, results1, classdict=self.selectedclasses, threshold=self.conf_threshold)['results']

        else:
            if img_array_flag:
                # print("entered img_array_flag")
                results = self.current_DLTA_model.inference(image)
                
                # post process results to send to the gui 
                
                
                # results = self.reader.decode_file(
                #     img=image, model=self.current_mm_model, classdict=self.selectedclasses, threshold=self.conf_threshold, img_array_flag=True)
                # # print(type(results))
                # if isinstance(results, tuple):
                #     results = self.reader.polegonise(
                #         results[0], results[1], classdict=self.selectedclasses, threshold=self.conf_threshold)['results']
                # else:
                #     results = results['results']
                
            else:
                results = self.reader.decode_file(
                    img=image, model=self.current_mm_model, classdict=self.selectedclasses, threshold=self.conf_threshold)
                if isinstance(results, tuple):
                    results = self.reader.polegonise(
                        results[0], results[1], classdict=self.selectedclasses, threshold=self.conf_threshold)['results']
                else:
                    results = results['results']
            start_time = time.time()

            final_shapes = self.parse_dlta_model_results_to_shapes(results)
            end_time = time.time()
            execution_time = end_time - start_time
            print("3---- Execution time for parsing to gui shapes(&masks2polygons):",
             int(execution_time * 1000), "milliseconds")
            
            end_time_all = time.time()
            print(
                f"total Time taken to annoatate img on {self.current_model_name}: {int((end_time_all - start_time_all)*1000)} ms")
            print(f'number of shapes: {len(final_shapes)}')


        return final_shapes


    def get_shapes_of_batch(self, images, multi_model_flag=False, notif = []):
        self.pd = self.startOperationDialog()
        self.thread = IntelligenceWorker(self.parent, images, self, multi_model_flag)
        self.thread.sinOut.connect(self.updateDialog)
        self.thread.start()
        self.notif = notif

    def updateDialog(self, completed, total):
        progress = int(completed/total*100)
        self.pd.setLabelText(str(completed) + "/" + str(total))
        self.pd.setValue(progress)
        if completed == total:
            self.onProgressDialogCanceledOrCompleted()


    def startOperationDialog(self):
        self.operationCanceled = False
        pd1 = QtWidgets.QProgressDialog(
            'Progress', 'Cancel', 0, 100, self.parent)
        pd1.setLabelText('Progress')
        pd1.setCancelButtonText('Cancel')
        pd1.setRange(0, 100)
        pd1.setValue(0)
        pd1.setMinimumDuration(0)
        pd1.show()
        pd1.canceled.connect(self.onProgressDialogCanceledOrCompleted)
        return pd1

    def onProgressDialogCanceledOrCompleted(self):
        try:
            if not self.notif[0] and not self.notif[1].isActiveWindow():
                self.notif[2]("Batch Annotation Completed")
        except:
            print("Error in batch mode notification")
        self.operationCanceled = True
        if self.parent.lastOpenDir and osp.exists(self.parent.lastOpenDir):
            self.parent.importDirImages(self.parent.lastOpenDir)
        else:
            self.parent.loadFile(self.parent.filename)


    def clear_annotating_models(self):
        self.reader.annotating_models.clear()

    def saveLabelFile(self, filename, detectedShapes):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=mathOps.flattener(s.points),
                    bbox=s.bbox,
                    group_id=s.group_id,
                    content=s.content,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item) for item in detectedShapes]

        imageData = LabelFile.load_image_file(filename)
        image = QtGui.QImage.fromData(imageData)
        if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
            os.makedirs(osp.dirname(filename))
        json_name = osp.splitext(filename)[0] + ".json"
        imagePath = osp.relpath(filename, osp.dirname(json_name))
        lf.save(
            filename=json_name,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=image.height(),
            imageWidth=image.width(),
            otherData={},
            flags={},
        )


    def parse_dlta_model_results_to_shapes(self, results):
        """
        This function converts the results of the model to shapes to be displayed in the GUI,
        while filtering the results based on the following criteria:
        1. Selected classes by the user
        2. Confidence threshold
        3. IOU NM Suppression threshold
        4. Segmentation accuracy (threshold for the epsilon parameter in the mask to polygon function)

        Args:
            results (list): A list containing the classes, confidences, and masks.

        Returns:
            list: A list of shapes representing the filtered and converted results.

        """
        if not results:
            return []
        classes, confidences, masks = results
        
        shapes = []
        
        for class_idx, confidence, mask in zip(classes, confidences, masks):
            if float(confidence) < self.conf_threshold:
                continue
            
            class_name = self.selectedclasses.get(class_idx)
            if class_name is None:
                continue
            
            shape = {   
                "label": class_name,
                "content": str(round(confidence, 2)),
                "points": [],
                "bbox": None,
                "shape_type": "polygon",
                "group_id": None,
                "flags": {},
                "other_data": {}
            }
            
            segmentation_points = mathOps.mask_to_polygons(mask, epsilon=self.segmentation_accuracy)
            
            if len(segmentation_points) < 3:
                continue
            
            shape["points"] = [item for sublist in segmentation_points for item in sublist]
            shape["bbox"] = mathOps.get_bbox_xyxy(segmentation_points)

            shapes.append(shape)
        
        # Apply non-maximum suppression based on confidence
        shapes = mathOps.OURnms_confidenceBased(shapes, self.iou_threshold)[0]

        return shapes
