import time
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
warnings.filterwarnings("ignore")

from .utils.helpers import mathOps



# Model imports
from .DLTA_Model import DLTA_Model_list
# get all files under models directory
models_dir = os.path.dirname(__file__) + '/models'
model_files = [f for f in os.listdir(models_dir)]
# import all of them (e.g, from .models import YOLOv8)
print ("Model families found in DLTA-AI:")
for model_file in model_files:
    if model_file.endswith(".py") and model_file != "__init__.py":
        exec(f"from .models import {model_file[:-3]}")
        print(f"{model_file[:-3]}")
# print(f'models list : {DLTA_Model_list}')



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


    def make_mm_model(self, selected_model_name):
        return selected_model_name, None

    @torch.no_grad()
    def make_DLTA_model(self, selected_model_name, model_family, config, checkpoint):
        
        # this is do get the family of the model to initialize the correct model accordingly
        model_idx = -1
        for index, model_instance in enumerate(DLTA_Model_list):
            if model_instance.model_family == model_family:
                # print(f"Match found at index {index} | {model_family}")
                model_idx = index
                break
        if model_idx == -1:
            print(f"Model family {model_family} not found")
            return
        
        self.current_DLTA_model = DLTA_Model_list[model_idx]
        self.current_DLTA_model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_DLTA_model.verify_installation()
        self.current_DLTA_model.initialize(checkpoint = checkpoint , config = config )


        return selected_model_name


    def get_shapes_of_one(self, image, multi_model_flag=False):
        
        if multi_model_flag:
            
            if len(self.selectedmodels) == 0:
                return []

            models_results = []  # Store results from each model
            for model in self.selectedmodels:
                print (f"Running inference on {model['Model Name']}")
                self.make_DLTA_model(model["Model Name"],
                    model["Family Name"],
                    model["Config"],
                    model["Checkpoint"])
                results = self.current_DLTA_model.inference(image)  # Run inference
                models_results.append(results)

            merged_results = self.merge_models_results(models_results)  # Merge results
            final_shapes = self.parse_dlta_model_results_to_shapes(merged_results)  # Parse shapes

        else:
            start_time_all = time.time()
            results = self.current_DLTA_model.inference(image)
            final_shapes = self.parse_dlta_model_results_to_shapes(results)
            end_time_all = time.time()
            print(
                f"total Time taken to annoatate img on {self.current_model_name}: {int((end_time_all - start_time_all)*1000)} ms")
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



    def merge_models_results(self, models_results):
        """
        Merge the results from multiple models.

        Args:
            models_results (list): A list of model results, where each result is a tuple
                containing the class indices, confidences, and masks.

        Returns:
            list: A list containing the merged class indices, confidences, and masks.
        """
        classes, confidences, masks = [], [], []
        for res in models_results:
            if not res:
                continue
            for class_idx, confidence, mask in zip(res[0], res[1], res[2]):
                classes.append(class_idx)
                confidences.append(confidence)
                masks.append(mask)
        return [classes, confidences, masks]

    def parse_dlta_model_results_to_shapes(self, results):
        """
        This function converts the results of the model to shapes to be displayed in the GUI,
        while filtering the results based on the following criteria:
        1. Selected classes by the user
        2. Confidence threshold
        3. Segmentation accuracy (threshold for the epsilon parameter in the mask to polygon function)
        4. IOU NM Suppression threshold

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
