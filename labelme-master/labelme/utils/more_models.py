from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QToolBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QPushButton, QProgressDialog, QApplication, QWidget
import json
import urllib.request
import requests
import os
import time

# store json file into list of dictionaries
cwd = os.getcwd()
with open(cwd + '/models_menu/models_json.json') as f:
    models_json = json.load(f)


class ModelExplorerDialog(QDialog):
    def __init__(self):
        # print current working directory
        super().__init__()
        self.setWindowTitle("Model Explorer")

        self.cols_labels = ["id", "Model Name", "Backbone", "Lr schd",
                            "Memory (GB)", "Inference Time (fps)", "box AP", "mask AP", "Checkpoint Size (MB)"]

        self.model_keys = sorted(
            list(set([model['Model'] for model in models_json])))

        # Set up the layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Set up the toolbar
        toolbar = QToolBar()
        layout.addWidget(toolbar)

        # Set up the model type dropdown menu
        self.model_type_dropdown = QComboBox()
        self.model_type_dropdown.addItems(["All"] + self.model_keys)
        toolbar.addWidget(self.model_type_dropdown)

        # Set up the checkboxes
        self.available_checkbox = QCheckBox("Downloaded")
        toolbar.addWidget(self.available_checkbox)
        self.not_available_checkbox = QCheckBox("Not Downloaded")
        toolbar.addWidget(self.not_available_checkbox)

        # Set up the search button
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.search)
        toolbar.addWidget(search_button)

        open_checkpoints_dir_button = QPushButton("Open Checkpoints Dir")
        # add icon to the button
        open_checkpoints_dir_button.setIcon(
            QtGui.QIcon(cwd + '/labelme/icons/downloads.png'))
        open_checkpoints_dir_button.setIconSize(QtCore.QSize(20, 20))

        open_checkpoints_dir_button.clicked.connect(
            self.open_checkpoints_dir)

        toolbar.addWidget(open_checkpoints_dir_button)

        # set spacing
        layout.setSpacing(10)

        # Set up the table
        self.table = QTableWidget()

        layout.addWidget(self.table)

        # Set up the number of rows and columns
        self.num_rows = 54
        self.num_cols = 9

        # make availability list
        self.check_availability()
        # Populate the table with default data
        self.populate_table()

        # Set up the submit and cancel buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        # add side padding to the button
        close_button.setFixedWidth(100)

        # make the button in the middle of the layout, don't stretch
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        # layout spacing
        layout.setSpacing(10)

        # Resize the dialog to fit the table
        self.resize(int(self.table.width() * 1.75),
                    int(self.table.height() * 1.5))

    def populate_table(self):
        # Clear the table (keep the header labels)
        self.table.clearContents()
        self.table.setRowCount(self.num_rows)
        # +2 for the available cell and select row button
        self.table.setColumnCount(self.num_cols + 2)

        # Set the header labels
        header = self.table.horizontalHeader()
        self.table.setHorizontalHeaderLabels(
            self.cols_labels + ["Status", "Select Model"])
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        # remove vertical header
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents)

        # Populate the table with data
        row_count = 0
        for model in models_json:
            col_count = 0
            for key in self.cols_labels:
                item = QTableWidgetItem(f"{model[key]}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row_count, col_count, item)
                col_count += 1

            # Select Model column
            self.selected_model = (-1, -1, -1)
            select_row_button = QPushButton("Select Model")
            select_row_button.clicked.connect(self.select_model)

            self.table.setContentsMargins(10, 10, 10, 10)
            self.table.setCellWidget(row_count, 10, select_row_button)

            # Downloaded column
            if model["Downloaded"]:
                available_item = QTableWidgetItem("Downloaded")
                # make the text color dark green
                available_item.setForeground(QtCore.Qt.darkGreen)
                self.table.setItem(row_count, 9, available_item)
                available_item.setTextAlignment(QtCore.Qt.AlignCenter)
            else:
                available_item = QPushButton("Requires Download")
                available_item.clicked.connect(
                    self.create_download_callback(model["id"]))
                # add padding to button
                available_item.setContentsMargins(10, 10, 10, 10)
                # maek the button text color red
                available_item.setStyleSheet("color: red")
                self.table.setCellWidget(row_count, 9, available_item)
                # make select_row_button disabled
                select_row_button.setEnabled(False)

            row_count += 1
        # hide the first row
        self.table.hideRow(0)

    def search(self):
        # Filter the table based on the selected model type and availability
        model_type = self.model_type_dropdown.currentText()
        available = self.available_checkbox.isChecked()
        not_available = self.not_available_checkbox.isChecked()

        for row in range(self.num_rows):
            show_row = True
            if model_type != "All":
                id = int(self.table.item(row, 0).text())
                if models_json[id]["Model"] != model_type:
                    show_row = False

            if available or not_available:
                available_text = self.table.item(row, 9)
                try:
                    available_text = available_text.text()
                except AttributeError:
                    pass
                if available and available_text != "Downloaded":
                    show_row = False
                if not_available and available_text == "Downloaded":
                    show_row = False

            self.table.setRowHidden(row, not show_row)

    def select_model(self):
        # Get the sender of the signal, which is the button that was clicked
        sender = self.sender()
        # Get the model index of the button in the table widget
        index = self.table.indexAt(sender.pos())
        # Get the row index from the model index
        row = index.row()
        model_id = int(self.table.item(row, 0).text())
        # select the model with this id from the models_json
        self.selected_model = models_json[model_id][
            "Model Name"], models_json[model_id]["Config"], models_json[model_id]["Checkpoint"],
        self.accept()

    def download_model(self, id):
        # get the checkpoints of the model with this id
        checkpoint_link = models_json[id]["Checkpoint_link"]
        model_name = models_json[id]["Model Name"]

        # create a progress dialog
        progress_dialog = QProgressDialog(
            f"Downloading {model_name}...", "Cancel", 0, 100, self)
        progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        progress_dialog.canceled.connect(self.cancel_download)
        progress_dialog.show()

        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_downloaded = 0
        self.download_canceled = False

        def handle_progress(block_num, block_size, total_size):
            # calculate the progress
            read_data = block_num * block_size
            if total_size > 0:
                download_percentage = read_data * 100 / total_size
                progress_dialog.setValue(download_percentage)
                progress_dialog.setLabelText(f"Downloading {model_name}... ")
                QApplication.processEvents()

        try:
            # download the file using requests
            response = requests.get(checkpoint_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            block_num = 0

            file_path = f"{cwd}/mmdetection/checkpoints/{checkpoint_link.split('/')[-1]}"
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if self.download_canceled:
                        break
                    f.write(data)
                    block_num += 1
                    handle_progress(block_num, block_size, total_size)

            if self.download_canceled:
                print("Download canceled by user")
        except Exception as e:
            print(f"Download error: {e}")

        # close the progress dialog
        progress_dialog.close()
        self.check_availability()
        self.populate_table()
        print("download finished")

    def cancel_download(self):
        self.download_canceled = True

    def create_download_callback(self, model_id):
        return lambda: self.download_model(model_id)

    def check_availability(self):
        checkpoints_dir = cwd + "/mmdetection/checkpoints/"
        for model in models_json:
            if model["Checkpoint"].split("/")[-1] in os.listdir(checkpoints_dir):
                model["Downloaded"] = True
            else:
                model["Downloaded"] = False

    def open_checkpoints_dir(self):
        url = QtCore.QUrl.fromLocalFile(cwd + "/mmdetection/checkpoints/")
        if not QtGui.QDesktopServices.openUrl(url):
            # print an error message if opening failed
            print("failed")


# if __name__ == "__main__":
#     app = QtWidgets.QApplication([])
#     window = ModelExplorerDialog()
#     window.show()
#     app.exec_()
