# relevant imports for the functions
from PyQt6 import QtCore
from PyQt6 import QtWidgets
import yaml


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

class SegmentationOptionsUI():
    def __init__(self, parent):
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
        
        

    # get the thresold as input from the user
    def setConfThreshold(self, prev_threshold=0.3):
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Threshold Selector')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)

        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel('Enter Confidence Threshold')
        layout.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
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

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
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

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return slider.value() / 100
        else:
            return prev_threshold
        

    def setIOUThreshold(self, prev_threshold=0.5):
        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Threshold Selector')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)

        layout = QtWidgets.QVBoxLayout(dialog)

        label = QtWidgets.QLabel('Enter IOU Threshold')
        layout.addWidget(label)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
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

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
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

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
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
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dialog.resize(500, 500)
        dialog.setMinimumSize(QtCore.QSize(500, 500))
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)

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
        buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
        buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttonBox.setObjectName("buttonBox")
        buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Select Classes")
        defaultButton = QtWidgets.QPushButton("Set as Default", dialog)
        buttonBox.addButton(defaultButton, QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)

        # Add the buttons to a QHBoxLayout
        buttonLayout = QtWidgets.QHBoxLayout()
        buttonLayout.addWidget(buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Ok))
        buttonLayout.addWidget(defaultButton)
        buttonLayout.addWidget(buttonBox.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel))

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
        dialog.exec()

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


    
