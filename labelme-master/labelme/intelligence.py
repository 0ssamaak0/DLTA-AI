import json
import time
from inferencing import models_inference
from labelme.label_file import LabelFile
from labelme.shape import Shape
from labelme import PY2
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtCore import QThread
from qtpy.QtCore import Signal as pyqtSignal
from qtpy import QtGui
from qtpy import QtWidgets
import threading
import os
import os.path as osp
import warnings
import numpy as np

import torch
from mmdet.apis import inference_detector, init_detector
warnings.filterwarnings("ignore")


from ultralytics import YOLO

coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
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
    sinOut = pyqtSignal(int,int)
    def __init__(self, parent, images, source):
        super(IntelligenceWorker, self).__init__(parent)
        self.parent = parent
        self.source = source
        self.images = images

    def run(self):
        index = 0
        total = len(self.images)
        for filename in self.images:
            
            if self.parent.isVisible==False:
                return
            if self.source.operationCanceled==True:
                return
            index = index + 1
            json_name = osp.splitext(filename)[0] + ".json"
            # if os.path.exists(json_name)==False:
            
            if os.path.isdir(json_name):
                os.remove(json_name)
            
            try:
                print("Decoding "+filename)
                s = self.source.get_shapes_of_one(filename)
                self.source.saveLabelFile(filename, s)
            except Exception as e:
                print(e)
            self.sinOut.emit(index,total)

class Intelligence():
    def __init__(self,parent):
        self.reader = models_inference()
        self.parent = parent
        self.threshold = 0.3
        self.selectedclasses = {0:"person", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
        self.current_model_name , self.current_mm_model = self.make_mm_model("")
        
        
    def make_mm_model(self, selected_model_name):
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
            print(f'selected model : {selected_model_name} \nconfig : {config}\ncheckpoint : {checkpoint} \n')
            
        torch.cuda.empty_cache()
        # model = init_detector("C:/Users/Shehab/Desktop/l001/ANNOTATION_TOOL/mmdetection/mmdetection/configs/detectors/htc_r50_sac_1x_coco.py", 
        #                     "C:/Users/Shehab/Desktop/l001/ANNOTATION_TOOL/mmdetection/mmdetection/checkpoints/htc_r50_sac_1x_coco-bfa60c54.pth", device = torch.device("cuda"))
        if selected_model_name == "YOLOv8x":
            model = YOLO(checkpoint)
            model.fuse()
            return selected_model_name, model
        
        model = init_detector(config, 
                            checkpoint, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # "C:\Users\Shehab\Desktop\l001\ANNOTATION_TOOL\mmdetection\mmdetection\configs\yolact\yolact_r50_1x8_coco.py"
        # model = init_detector("C:/Users/Shehab/Desktop/mmdetection/mmdetection/configs/detectors/htc_r50_sac_1x_coco.py",
                            # "C:/Users/Shehab/Desktop/mmdetection/mmdetection/checkpoints/htc_r50_sac_1x_coco-bfa60c54.pth", device = torch.device("cuda"))
        return selected_model_name, model   
        
        
        
    def get_bbox(self,segmentation):
        x = []
        y = []
        for i in range(len(segmentation)):
            x.append(segmentation[i][0])
            y.append(segmentation[i][1])
        # get the bbox in xyxy format
        if len(x)==0 or len(y)==0:
            return []
        bbox = [min(x),min(y),max(x) ,max(y)]
        return bbox
        
        
    def get_shapes_of_one(self,image,img_array_flag = False):
        # print(f"Threshold is {self.threshold}")
        # results = self.reader.decode_file(img_path = filename, threshold = self.threshold , selected_model_name = self.current_model_name)["results"]
        start_time = time.time()
        # if img_array_flag is true then the image is a numpy array and not a path
        if img_array_flag:
            results = self.reader.decode_file(img = image, model = self.current_mm_model,classdict = self.selectedclasses ,threshold = self.threshold, img_array_flag=True )["results"]
        else:
            results = self.reader.decode_file(img = image, model = self.current_mm_model,classdict = self.selectedclasses ,threshold = self.threshold )["results"]
        end_time = time.time()
        print(f"Time taken to annoatate img on {self.current_model_name}: {end_time - start_time}")

        shapes = []
        for result in results:
            
  
            shape = {}
            shape["label"] = result["class"]
            shape["content"] = result["confidence"]
            shape["group_id"] = None
            shape["shape_type"]= "polygon"
            shape["bbox"] = self.get_bbox(result["seg"])
            
            
            shape["flags"] = {}
            shape["other_data"] = {}
            
            # shape_points is result["seg"] flattened
            shape["points"] = [item for sublist in result["seg"] for item in sublist]

            shapes.append(shape)
            
            #self.addLabel(shape)
        return shapes
    
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


    def get_shapes_of_batch(self,images):
        self.pd = self.startOperationDialog()
        self.thread = IntelligenceWorker(self.parent,images,self)
        self.thread.sinOut.connect(self.updateDialog)
        self.thread.start()



    # get the thresold as input from the user
    def setThreshold(self):
        text, ok = QtWidgets.QInputDialog.getText(self.parent, 'Threshold Selector', 'Enter the threshold:')
        if ok:
            return text
        else:
            return 0.3
    # add a resizable and scrollable dialog that contains all coco classes and allow the user to select among them using checkboxes
    def selectClasses(self):
        #self.selectedclasses.clear()
        # return the selected classes
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Classes')
        dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        dialog.resize(500, 500)
        dialog.setMinimumSize(QtCore.QSize(500, 500))
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
        buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        buttonBox.setObjectName("buttonBox")
        verticalLayout.addWidget(buttonBox)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        self.classes = []
        for i in range(len(coco_classes)):
            self.classes.append(QtWidgets.QCheckBox(coco_classes[i], dialog))
            verticalLayout_2.addWidget(self.classes[i])
        # if the class is in self.selectedclasses then check the checkbox by default 
        for value in self.selectedclasses.values():
            if value != None:
                indx = coco_classes.index(value)
                self.classes[indx].setChecked(True)
        dialog.show()
        dialog.exec_()
        self.selectedclasses.clear()
        for i in range(len(self.classes)):
            if self.classes[i].isChecked():
                indx = coco_classes.index(self.classes[i].text())
                self.selectedclasses[indx] = self.classes[i].text()
        #print(self.selectedclasses)
        #self.updatlabellist()
        return self.selectedclasses


    def updateDialog(self, completed, total):
        progress = int(completed/total*100)
        self.pd.setLabelText(str(completed) +"/"+ str(total))
        self.pd.setValue(progress)
        if completed==total:
            self.onProgressDialogCanceledOrCompleted()
            
    def startOperationDialog(self):
        self.operationCanceled = False
        pd1 =  QtWidgets.QProgressDialog('Progress','Cancel',0,100,self.parent)
        pd1.setLabelText('Progress')
        pd1.setCancelButtonText('Cancel')
        pd1.setRange(0, 100)
        pd1.setValue(0)
        pd1.setMinimumDuration(0)
        pd1.show()
        pd1.canceled.connect(self.onProgressDialogCanceledOrCompleted)
        return pd1
        
        
    def onProgressDialogCanceledOrCompleted(self):
        self.operationCanceled = True
        if self.parent.lastOpenDir and osp.exists(self.parent.lastOpenDir):
            self.parent.importDirImages(self.parent.lastOpenDir)
        else:
            self.parent.loadFile(self.parent.filename)
        
    def flattener(self, list_2d):
        points=[(p.x(), p.y()) for p in list_2d]
        points= np.array(points, np.uint16).flatten().tolist()
        return points
    
    def saveLabelFile(self, filename, detectedShapes):
        lf = LabelFile()
        
        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points= self.flattener(s.points),
                    bbox = s.bbox,
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