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

        try:
            with open("models_menu/models_metadata.json") as json_file:
                data = json.load(json_file)
            # Filter for downloaded models
            downloaded_models = [item for item in data if item["Downloaded"] == "YES"]
            excluded_families = ["SAM_basic"]  # add here any other families such as sam 
            self.models = [item for item in downloaded_models if item["Family Name"] not in excluded_families]
        except FileNotFoundError:
            print("Error: models_metadata.json file not found.")
        except json.JSONDecodeError:
            print("Error: Unable to load models_metadata.json file.")


        dialog = QtWidgets.QDialog(self.parent)
        dialog.setWindowTitle('Select Models')
        dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        dialog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dialog.resize(400, 500)
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
            QtWidgets.QDialogButtonBox.StandardButton.Cancel | QtWidgets.QDialogButtonBox.StandardButton.Ok)
        buttonBox.setObjectName("buttonBox")
        verticalLayout.addWidget(buttonBox)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        self.models_checkboxes = []
        for i in range(len(self.models)):
            name = f'{self.models[i]["Model Name"]} | {self.models[i]["Family Name"]}'
            checkbox = QtWidgets.QCheckBox(name, dialog)
            self.models_checkboxes.append(checkbox)
            verticalLayout_2.addWidget(checkbox)
            
        dialog.show()
        dialog.exec()
        self.selectedmodels.clear()
        for i in range(len(self.models)):
            if self.models_checkboxes[i].isChecked():
                self.selectedmodels.append(self.models[i])
        return self.selectedmodels

