
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

# add ClassWidget and allow the user to select among coco classes using a combobox
class Classeswidget(QtWidgets.QDialog):
    def __init__(self):
        super(Classeswidget, self).__init__()
        self.setModal(True)
        self.setWindowTitle("Select Class")
        self.class_name = "person"
        self.class_name = self._createQComboBox()
    
    def _createQComboBox(self):
        class_name = QtWidgets.QComboBox()
        class_name.addItems(["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"])
        class_name.currentIndexChanged.connect(self.onNewValue)
        return class_name

    def onNewValue(self, value):
        self.class_name = value

    def getValue(self):
        return self.class_name

    def setValue(self, value):
        self.class_name = value

    def exec_(self):
        super(Classeswidget, self).exec_()
        return self.class_name

