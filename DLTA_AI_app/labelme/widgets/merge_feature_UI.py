import json
from PyQt6 import QtWidgets
from PyQt6 import QtCore


# create an interface for merging features
class MergeFeatureUI():
    def __init__(self, parent):
        self.parent = parent
        self.selectedmodels = []

    # merge dialog 
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
        # ExplorerMerge.exec()

        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Models')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
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
        buttonBox.setOrientation(QtCore.Qt.Orientation.Horizontal)
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
        dialog.exec()
        self.selectedmodels.clear()
        for i in range(len(self.models)):
            if self.models[i].isChecked():
                self.selectedmodels.append(self.models[i].text())
        print(self.selectedmodels)
        return self.selectedmodels

