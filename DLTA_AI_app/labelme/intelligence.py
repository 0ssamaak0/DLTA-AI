from ultralytics import YOLO
import json
import time
try:
    from inferencing import models_inference
except ModuleNotFoundError:
    import subprocess
    print("The required package 'mmcv-full' is not currently installed. It will now be installed. This process may take some time. Note that this package will only be installed the first time you use DLTA-AI.")
    subprocess.run(["mim", "install", "mmcv-full==1.7.0"])
    from inferencing import models_inference
from labelme.label_file import LabelFile
from labelme import PY2
from qtpy import QtCore
from qtpy.QtCore import QThread
from qtpy.QtCore import Signal as pyqtSignal
from qtpy import QtGui
from qtpy import QtWidgets
import os
import os.path as osp
import warnings
import numpy as np
import urllib.request
from .shape import Shape
from labelme.utils.model_explorer import ModelExplorerDialog
import yaml

import torch
from mmdet.apis import inference_detector, init_detector
warnings.filterwarnings("ignore")

from .utils import helpers


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
                s = convert_shapes_to_qt_shapes(s)
                self.source.saveLabelFile(filename, s)
            except Exception as e:
                print(e)
            self.sinOut.emit(index, total)


def convert_shapes_to_qt_shapes(shapes):
    qt_shapes = []
    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        bbox = shape["bbox"]
        shape_type = shape["shape_type"]
        # flags = shape["flags"]
        content = shape["content"]
        group_id = shape["group_id"]
        # other_data = shape["other_data"]

        if not points:
            # skip point-empty shape
            continue

        shape = Shape(
            label=label,
            shape_type=shape_type,
            group_id=group_id,
            content=content,
        )
        for i in range(0, len(points), 2):
            shape.addPoint(QtCore.QPointF(points[i], points[i + 1]))
        shape.close()
        qt_shapes.append(shape)
    return qt_shapes


class Intelligence():
    def __init__(self, parent):
        self.reader = models_inference()
        self.parent = parent
        self.conf_threshold = 0.3
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
        self.selectedmodels = []
        self.current_model_name, self.current_mm_model = self.make_mm_model("")

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
            helpers.OKmsgBox("Error", f"Error in loading the model\n{e}", "critical")
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
            # "C:/Users/Shehab/Desktop/mmdetection/mmdetection/checkpoints/htc_r50_sac_1x_coco-bfa60c54.pth", device = torch.device("cuda"))
        return selected_model_name, model

    @ torch.no_grad()
    def make_mm_model_more(self, selected_model_name, config, checkpoint):
        torch.cuda.empty_cache()
        print(
            f"Selected model is {selected_model_name}\n and config is {config}\n and checkpoint is {checkpoint}")

        # if YOLOv8
        if "YOLOv8" in selected_model_name:
            try:
                model = YOLO(checkpoint)
                model.fuse()
                return selected_model_name, model
            except Exception as e:
                helpers.OKmsgBox("Error", f"Error in loading the model\n{e}", "critical")
                return

        # It's a MMDetection model
        else:
            try:
                print(f"From the new one: {config}")
                model = init_detector(config, checkpoint, device=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"))
            except Exception as e:
                helpers.OKmsgBox
                helpers.OKmsgBox("Error", f"Error in loading the model\n{e}", "critical")
                return
            return selected_model_name, model

    def get_bbox(self, segmentation):
        x = []
        y = []
        for i in range(len(segmentation)):
            x.append(segmentation[i][0])
            y.append(segmentation[i][1])
        # get the bbox in xyxy format
        if len(x) == 0 or len(y) == 0:
            return []
        bbox = [min(x), min(y), max(x), max(y)]
        return bbox

    def get_shapes_of_one(self, image, img_array_flag=False, multi_model_flag=False):
        # print(f"Threshold is {self.conf_threshold}")
        # results = self.reader.decode_file(img_path = filename, threshold = self.conf_threshold , selected_model_name = self.current_model_name)["results"]
        start_time = time.time()
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
                results = self.reader.decode_file(
                    img=image, model=self.current_mm_model, classdict=self.selectedclasses, threshold=self.conf_threshold, img_array_flag=True)
                # print(type(results))
                if isinstance(results, tuple):
                    results = self.reader.polegonise(
                        results[0], results[1], classdict=self.selectedclasses, threshold=self.conf_threshold)['results']
                else:
                    results = results['results']
            else:
                results = self.reader.decode_file(
                    img=image, model=self.current_mm_model, classdict=self.selectedclasses, threshold=self.conf_threshold)
                if isinstance(results, tuple):
                    results = self.reader.polegonise(
                        results[0], results[1], classdict=self.selectedclasses, threshold=self.conf_threshold)['results']
                else:
                    results = results['results']
            end_time = time.time()
            print(
                f"Time taken to annoatate img on {self.current_model_name}: {int((end_time - start_time)*1000)} ms")

        shapes = []
        for result in results:
            shape = {}
            shape["label"] = result["class"]
            shape["content"] = result["confidence"]
            shape["group_id"] = None
            shape["shape_type"] = "polygon"
            shape["bbox"] = self.get_bbox(result["seg"])

            shape["flags"] = {}
            shape["other_data"] = {}

            # shape_points is result["seg"] flattened
            shape["points"] = [item for sublist in result["seg"]
                               for item in sublist]

            shapes.append(shape)
            shapes, boxes, confidences, class_ids, segments = self.OURnms(
                shapes, self.iou_threshold)
            # self.addLabel(shape)
        return shapes

    def get_boxes_conf_classids_segments(self, shapes):
        boxes = []
        confidences = []
        class_ids = []
        segments = []
        for s in shapes:
            label = s["label"]
            points = s["points"]
            # points are one dimensional array of x1,y1,x2,y2,x3,y3,x4,y4
            # we will convert it to a 2 dimensional array of points (segment)
            segment = []
            for j in range(0, len(points), 2):
                segment.append([int(points[j]), int(points[j + 1])])
            # if points is empty pass
            # if len(points) == 0:
            #     continue
            segments.append(segment)

            boxes.append(self.get_bbox(segment))
            confidences.append(float(s["content"]))
            class_ids.append(coco_classes.index(
                label)if label in coco_classes else -1)

        return boxes, confidences, class_ids, segments

    def compute_iou(self, box1, box2):
        """
        Computes IOU between two bounding boxes.

        Args:
            box1 (list): List of 4 coordinates (xmin, ymin, xmax, ymax) of the first box.
            box2 (list): List of 4 coordinates (xmin, ymin, xmax, ymax) of the second box.

        Returns:
            iou (float): IOU between the two boxes.
        """
        # Compute intersection coordinates
        xmin = max(box1[0], box2[0])
        ymin = max(box1[1], box2[1])
        xmax = min(box1[2], box2[2])
        ymax = min(box1[3], box2[3])

        # Compute intersection area
        if xmin < xmax and ymin < ymax:
            intersection_area = (xmax - xmin) * (ymax - ymin)
        else:
            intersection_area = 0

        # Compute union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Compute IOU
        iou = intersection_area / union_area if union_area > 0 else 0

        return iou

    def OURnms(self, shapes, iou_threshold=0.5):
        """
        Perform non-maximum suppression on a list of shapes based on their bounding boxes using IOU threshold.

        Args:
            shapes (list): List of shapes, each shape is a dictionary with keys (bbox, confidence, class_id)
            iou_threshold (float): IOU threshold for non-maximum suppression.

        Returns:
            list: List of shapes after performing non-maximum suppression, each shape is a dictionary with keys (bbox, confidence, class_id)
        """
        iou_threshold = float(iou_threshold)
        for shape in shapes:
            if shape['content'] is None:
                shape['content'] = 1.0

        # Sort shapes by their confidence
        shapes.sort(key=lambda x: x['content'], reverse=True)

        boxes, confidences, class_ids, segments = self.get_boxes_conf_classids_segments(
            shapes)

        toBeRemoved = []

        # Loop through each shape
        for i in range(len(shapes)):
            shape_bbox = boxes[i]
            # Loop through each remaining shape
            for j in range(i + 1, len(shapes)):
                remaining_shape_bbox = boxes[j]

                # Compute IOU between shape and remaining_shape
                iou = self.compute_iou(shape_bbox, remaining_shape_bbox)

                # If IOU is greater than threshold, remove remaining_shape from shapes list
                if iou > iou_threshold:
                    toBeRemoved.append(j)

        shapesFinal = []
        boxesFinal = []
        confidencesFinal = []
        class_idsFinal = []
        segmentsFinal = []
        for i in range(len(shapes)):
            if i in toBeRemoved:
                continue
            shapesFinal.append(shapes[i])
        boxesFinal, confidencesFinal, class_idsFinal, segmentsFinal = self.get_boxes_conf_classids_segments(
            shapesFinal)

        return shapesFinal, boxesFinal, confidencesFinal, class_idsFinal, segmentsFinal

    # print the labels of the selected classes in the dialog
    # def updatlabellist(self):
    #     for selectedclass in self.selectedclasses.values():
    #         shape = Shape()
    #         shape.label = selectedclass
    #         shape.content = ""
    #         shape.shape_type="polygon"
    #         shape.flags = {}
    #         shape.other_data = {}
    #         mainwindow = self.parent
    #         mainwindow.addLabel(shape)

    def get_shapes_of_batch(self, images, multi_model_flag=False, notif = []):
        self.pd = self.startOperationDialog()
        self.thread = IntelligenceWorker(self.parent, images, self, multi_model_flag)
        self.thread.sinOut.connect(self.updateDialog)
        self.thread.start()
        self.notif = notif

    # get the thresold as input from the user

    def setConfThreshold(self, prev_threshold=0.3):
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Threshold Selector')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel('Enter Confidence Threshold')
        layout.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(int(prev_threshold * 100))

        text_input = QtWidgets.QLineEdit(str(prev_threshold))

        def on_slider_change(value):
            text_input.setText(str(value / 100))

        def on_text_change(text):
            try:
                value = float(text)
                slider.setValue(int(value * 100))
            except ValueError:
                pass

        slider.valueChanged.connect(on_slider_change)
        text_input.textChanged.connect(on_text_change)

        layout.addWidget(slider)
        layout.addWidget(text_input)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        def on_ok():
            threshold = float(text_input.text())
            dialog.accept()
            return threshold

        def on_cancel():
            dialog.reject()
            return prev_threshold

        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(on_cancel)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return slider.value() / 100
        else:
            return prev_threshold

    def setIOUThreshold(self, prev_threshold=0.5):
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Threshold Selector')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel('Enter IOU Threshold')
        layout.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setValue(int(prev_threshold * 100))

        text_input = QtWidgets.QLineEdit(str(prev_threshold))

        def on_slider_change(value):
            text_input.setText(str(value / 100))

        def on_text_change(text):
            try:
                value = float(text)
                slider.setValue(int(value * 100))
            except ValueError:
                pass

        slider.valueChanged.connect(on_slider_change)
        text_input.textChanged.connect(on_text_change)

        layout.addWidget(slider)
        layout.addWidget(text_input)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        def on_ok():
            threshold = float(text_input.text())
            dialog.accept()
            return threshold

        def on_cancel():
            dialog.reject()
            return prev_threshold

        button_box.accepted.connect(on_ok)
        button_box.rejected.connect(on_cancel)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            return slider.value() / 100
        else:
            return prev_threshold

    # add a resizable and scrollable dialog that contains all coco classes and allow the user to select among them using checkboxes

    def selectClasses(self):
        """
        Display a dialog box that allows the user to select which classes to annotate.

        The function creates a QDialog object and adds various widgets to it, including a QScrollArea that contains QCheckBox
        widgets for each class. The function sets the state of each QCheckBox based on whether the class is in the
        self.selectedclasses dictionary. The function also adds "Select All", "Deselect All", "Select Classes", "Set as Default",
        and "Cancel" buttons to the dialog box. When the user clicks the "Select Classes" button, the function saves the selected
        classes to the self.selectedclasses dictionary and returns it.

        :return: A dictionary that maps class indices to class names for the selected classes.
        """
        # Create a new dialog box
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Classes')
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.resize(500, 500)
        dialog.setMinimumSize(QtCore.QSize(500, 500))
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

        # Create a vertical layout for the dialog box
        verticalLayout = QtWidgets.QVBoxLayout(dialog)
        verticalLayout.setObjectName("verticalLayout")

        # Create a horizontal layout for the "Select All" and "Deselect All" buttons
        horizontalLayout = QtWidgets.QHBoxLayout()
        selectAllButton = QtWidgets.QPushButton("Select All", dialog)
        deselectAllButton = QtWidgets.QPushButton("Deselect All", dialog)
        horizontalLayout.addWidget(selectAllButton)
        horizontalLayout.addWidget(deselectAllButton)
        verticalLayout.addLayout(horizontalLayout)

        # Create a scroll area for the class checkboxes
        scrollArea = QtWidgets.QScrollArea(dialog)
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName("scrollArea")
        scrollAreaWidgetContents = QtWidgets.QWidget()
        scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 478, 478))
        scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        gridLayout = QtWidgets.QGridLayout(scrollAreaWidgetContents)
        gridLayout.setObjectName("gridLayout")
        self.scrollAreaWidgetContents = scrollAreaWidgetContents
        scrollArea.setWidget(scrollAreaWidgetContents)
        verticalLayout.addWidget(scrollArea)

        # Create a button box for the "Select Classes", "Set as Default", and "Cancel" buttons
        buttonBox = QtWidgets.QDialogButtonBox(dialog)
        buttonBox.setOrientation(QtCore.Qt.Horizontal)
        buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")
        buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setText("Select Classes")
        defaultButton = QtWidgets.QPushButton("Set as Default", dialog)
        buttonBox.addButton(defaultButton, QtWidgets.QDialogButtonBox.ActionRole)

        # Add the buttons to a QHBoxLayout
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(buttonBox.button(QtWidgets.QDialogButtonBox.Ok))
        buttonLayout.addWidget(defaultButton)
        buttonLayout.addWidget(buttonBox.button(QtWidgets.QDialogButtonBox.Cancel))

        # Add the QHBoxLayout to the QVBoxLayout
        verticalLayout.addLayout(buttonLayout)

        # Connect the button signals to their respective slots
        buttonBox.accepted.connect(lambda: self.saveClasses(dialog))
        buttonBox.rejected.connect(dialog.reject)
        defaultButton.clicked.connect(lambda: self.saveClasses(dialog, True))

        # Create a QCheckBox for each class and add it to the grid layout
        self.classes = []
        for i in range(len(coco_classes)):
            self.classes.append(QtWidgets.QCheckBox(coco_classes[i], dialog))
            row = i // 3
            col = i % 3
            gridLayout.addWidget(self.classes[i], row, col)

        # Set the state of each QCheckBox based on whether the class is in the self.selectedclasses dictionary
        for value in self.selectedclasses.values():
            if value != None:
                indx = coco_classes.index(value)
                self.classes[indx].setChecked(True)

        # Connect the "Select All" and "Deselect All" buttons to their respective slots
        selectAllButton.clicked.connect(lambda: self.selectAll())
        deselectAllButton.clicked.connect(lambda: self.deselectAll())

        # Show the dialog box and wait for the user to close it
        dialog.show()
        dialog.exec_()

        # Save the selected classes to the self.selectedclasses dictionary and return it
        self.selectedclasses.clear()
        for i in range(len(self.classes)):
            if self.classes[i].isChecked():
                indx = coco_classes.index(self.classes[i].text())
                self.selectedclasses[indx] = self.classes[i].text()
        return self.selectedclasses

    def saveClasses(self, dialog, is_default=False):
        """
        Save the selected classes to the self.selectedclasses dictionary.

        The function clears the self.selectedclasses dictionary and then iterates over the QCheckBox widgets for each class.
        If a QCheckBox is checked, the function adds the corresponding class name to the self.selectedclasses dictionary. If the
        is_default parameter is True, the function also updates the default_config.yaml file with the selected classes.

        :param dialog: The QDialog object that contains the class selection dialog.
        :param is_default: A boolean that indicates whether to update the default_config.yaml file with the selected classes.
        """
        # Clear the self.selectedclasses dictionary
        self.selectedclasses.clear()

        # Iterate over the QCheckBox widgets for each class
        for i in range(len(self.classes)):
            if self.classes[i].isChecked():
                indx = coco_classes.index(self.classes[i].text())
                self.selectedclasses[indx] = self.classes[i].text()

        # If is_default is True, update the default_config.yaml file with the selected classes
        if is_default:
            with open("labelme/config/default_config.yaml", 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            config['default_classes'] = list(self.selectedclasses.values())
            with open("labelme/config/default_config.yaml", 'w') as f:
                yaml.dump(config, f)

        # Accept the dialog box
        dialog.accept()

    def selectAll(self):
        """
        Select all classes in the class selection dialog.

        The function iterates over the QCheckBox widgets for each class and sets their checked state to True.
        """
        # Iterate over the QCheckBox widgets for each class and set their checked state to True
        for checkbox in self.classes:
            checkbox.setChecked(True)

    def deselectAll(self):
        """
        Deselect all classes in the class selection dialog.

        The function iterates over the QCheckBox widgets for each class and sets their checked state to False.
        """
        # Iterate over the QCheckBox widgets for each class and set their checked state to False
        for checkbox in self.classes:
            checkbox.setChecked(False)

    def mergeSegModels(self):
        # add a resizable and scrollable dialog that contains all the models and allow the user to select among them using checkboxes
        models = []
        with open("saved_models.json") as json_file:
            data = json.load(json_file)
            for model in data.keys():
                if "YOLOv8" not in model:
                    models.append(model)
        # ExplorerMerge = ModelExplorerDialog(merge=True)
        # ExplorerMerge.adjustSize()
        # ExplorerMerge.resize(
        #     int(ExplorerMerge.width() * 2), int(ExplorerMerge.height() * 1.5))
        # ExplorerMerge.exec_()

        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Models')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.resize(200, 250)
        dialog.setMinimumSize(QtCore.QSize(200, 200))
        verticalLayout = QtWidgets.QVBoxLayout(dialog)
        verticalLayout.setObjectName("verticalLayout")
        scrollArea = QtWidgets.QScrollArea(dialog)
        scrollArea.setWidgetResizable(True)
        scrollArea.setObjectName("scrollArea")
        scrollAreaWidgetContents = QtWidgets.QWidget()
        scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 478, 478))
        scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        verticalLayout_2 = QtWidgets.QVBoxLayout(scrollAreaWidgetContents)
        verticalLayout_2.setObjectName("verticalLayout_2")
        self.scrollAreaWidgetContents = scrollAreaWidgetContents
        scrollArea.setWidget(scrollAreaWidgetContents)
        verticalLayout.addWidget(scrollArea)
        buttonBox = QtWidgets.QDialogButtonBox(dialog)
        buttonBox.setOrientation(QtCore.Qt.Horizontal)
        buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")
        verticalLayout.addWidget(buttonBox)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        self.models = []
        for i in range(len(models)):
            self.models.append(QtWidgets.QCheckBox(models[i], dialog))
            verticalLayout_2.addWidget(self.models[i])
        dialog.show()
        dialog.exec_()
        self.selectedmodels.clear()
        for i in range(len(self.models)):
            if self.models[i].isChecked():
                self.selectedmodels.append(self.models[i].text())
        print(self.selectedmodels)
        return self.selectedmodels

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


    def flattener(self, list_2d):
        points = [(p.x(), p.y()) for p in list_2d]
        points = np.array(points, np.uint16).flatten().tolist()
        return points

    def clear_annotating_models(self):
        self.reader.annotating_models.clear()

    def saveLabelFile(self, filename, detectedShapes):
        lf = LabelFile()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=self.flattener(s.points),
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
