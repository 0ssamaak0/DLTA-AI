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
    """
    A dialog window for exploring available models and downloading them.

    Attributes:
        main_window (QMainWindow): The main window of the application.
        mute (bool): Whether to mute notifications or not.
        notification (function): A function for displaying notifications.
    """

    def __init__(self, main_window=None, mute=None, notification=None):
        """
        Initializes the ModelExplorerDialog.

        Args:
            main_window (QMainWindow): The main window of the application.
            mute (bool): Whether to mute notifications.
            notification (function): A function for displaying notifications.
        """
        super().__init__()
        self.main_window = main_window
        self.mute = mute
        self.notification = notification
        self.setWindowTitle("Model Explorer")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)


        # Define the columns of the table
        self.cols_labels = ["id", "Model Name", "Backbone", "Lr schd",
                            "Memory (GB)", "Inference Time (fps)", "box AP", "mask AP", "Checkpoint Size (MB)"]

        # Get the unique model names
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
        self.model_type_dropdown.currentIndexChanged.connect(self.search)
        toolbar.addWidget(self.model_type_dropdown)

        # Set up the checkboxes
        self.available_checkbox = QCheckBox("Downloaded")
        self.available_checkbox.clicked.connect(self.search)
        toolbar.addWidget(self.available_checkbox)
        self.not_available_checkbox = QCheckBox("Not Downloaded")
        self.not_available_checkbox.clicked.connect(self.search)
        toolbar.addWidget(self.not_available_checkbox)

        # Set up the search button
        # search_button = QPushButton("Search")
        # search_button.clicked.connect(self.search)
        # toolbar.addWidget(search_button)

        # Set up the button for opening the checkpoints directory
        open_checkpoints_dir_button = QPushButton("Open Checkpoints Dir")
        # add icon to the button
        open_checkpoints_dir_button.setIcon(
            QtGui.QIcon(cwd + '/labelme/icons/downloads.png'))
        open_checkpoints_dir_button.setIconSize(QtCore.QSize(20, 20))
        open_checkpoints_dir_button.clicked.connect(
            self.open_checkpoints_dir)
        toolbar.addWidget(open_checkpoints_dir_button)

        # Set spacing
        layout.setSpacing(10)

        # Set up the table
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Set up the number of rows and columns
        self.num_rows = len(models_json)
        self.num_cols = 9

        # Make availability list
        self.check_availability()

        # Populate the table with default data
        self.populate_table()

        # Set up the submit and cancel buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        close_button = QPushButton("Ok")
        close_button.clicked.connect(self.close)
        # add side padding to the button
        close_button.setFixedWidth(100)

    
        # make the button in the middle of the layout, don't stretch
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()

        # layout spacing
        layout.setSpacing(10)

    def populate_table(self):
        """
        Populates the table with data from models_json.

        Returns:
            None
        """
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

            # Disable SAM Selection
            if model["Model"] == "SAM":
                select_row_button.setEnabled(False)
                # change text
                select_row_button.setText("Select from SAM Toolbar")

            row_count += 1

    def search(self):
        """
        Filters the table based on the selected model type and availability.

        Returns:
            None
        """
        # Get the selected model type and availability
        model_type = self.model_type_dropdown.currentText()
        available = self.available_checkbox.isChecked()
        not_available = self.not_available_checkbox.isChecked()

        # Iterate over each row in the table
        for row in range(self.num_rows):
            show_row = True
            # Filter by model type
            if model_type != "All":
                id = int(self.table.item(row, 0).text())
                if models_json[id]["Model"] != model_type:
                    show_row = False

            # Filter by availability
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

            # Hide or show the row based on the filters
            self.table.setRowHidden(row, not show_row)


    def select_model(self):
        """
        Gets the selected model from the table and sets it as the selected model.

        Returns:
            None
        """
        # Get the button that was clicked
        sender = self.sender()
        # Get the row index of the button in the table
        index = self.table.indexAt(sender.pos())
        # Get the model id from the row index
        row = index.row()
        model_id = int(self.table.item(row, 0).text())
        # Set the selected model as the model with this id
        self.selected_model = models_json[model_id]["Model Name"], models_json[model_id]["Config"], models_json[model_id]["Checkpoint"],
        self.accept()

    def download_model(self, id):
        """
        Downloads the model with the given id and updates the progress dialog.

        Args:
            id (int): The id of the model to download.

        Returns:
            None
        """
        # Get the checkpoint link and model name for the model with this id
        checkpoint_link = models_json[id]["Checkpoint_link"]
        model_name = models_json[id]["Model Name"]

        # Create a progress dialog
        self.progress_dialog = QProgressDialog(
            f"Downloading {model_name}...", "Cancel", 0, 100, self)
        # Set the window title
        self.progress_dialog.setWindowTitle("Downloading Model")
        self.progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.cancel_download)
        self.progress_dialog.show()

        # Initialize variables for tracking download progress
        self.start_time = time.time()
        self.last_time = self.start_time
        self.last_downloaded = 0
        self.download_canceled = False

        def handle_progress(block_num, block_size, total_size):
            """
            Updates the progress dialog with the current download progress.

            Args:
                block_num (int): The number of blocks downloaded.
                block_size (int): The size of each block.
                total_size (int): The total size of the file being downloaded.

            Returns:
                None
            """
            # failed flag
            # Calculate the download progress
            read_data = block_num * block_size
            if total_size > 0:
                download_percentage = read_data * 100 / total_size
                self.progress_dialog.setValue(download_percentage)
                self.progress_dialog.setLabelText(f"Downloading {model_name}... ")
                QApplication.processEvents()

        failed = False
        try:
            # Download the file using requests
            response = requests.get(checkpoint_link, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            block_num = 0

            # Save the downloaded file to disk
            file_path = f"{cwd}/mmdetection/checkpoints/{checkpoint_link.split('/')[-1]}"
            with open(file_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    if self.download_canceled:
                        break
                    f.write(data)
                    block_num += 1
                    handle_progress(block_num, block_size, total_size)
            if self.download_canceled:
                # Delete the file if the download was canceled
                os.remove(file_path)
                print("Download canceled by user")
                failed = True
        except Exception as e:
            os.remove(file_path)
            print(f"Download error: {e}")
            failed = True

        # Close the progress dialog and update the table
        self.progress_dialog.close()
        self.check_availability()
        self.populate_table()
        print("Download finished")

        # Show a notification if the main window is not active
        try:
            if not self.mute:
                if not self.isActiveWindow():
                    if not failed:
                        self.notification(f"{model_name} has been downloaded successfully")
                    else:
                        self.notification(f"Failed to download {model_name}")
        except:
            pass


    def cancel_download(self):
        """
        Sets the download_canceled flag to True to cancel the download.

        Returns:
            None
        """
        self.download_canceled = True


    def create_download_callback(self, model_id):
        """
        Returns a lambda function that downloads the model with the given id.

        Args:
            model_id (int): The id of the model to download.

        Returns:
            function: A lambda function that downloads the model with the given id.
        """
        return lambda: self.download_model(model_id)


    def check_availability(self):
        """
        Checks the availability of each model in the table and updates the "Downloaded" column.

        Returns:
            None
        """
        checkpoints_dir = cwd + "/mmdetection/checkpoints/"
        for model in models_json:
            if model["Checkpoint"].split("/")[-1] in os.listdir(checkpoints_dir):
                model["Downloaded"] = True
            else:
                model["Downloaded"] = False


    def open_checkpoints_dir(self):
        """
        Opens the directory containing the downloaded checkpoints in the file explorer.

        Returns:
            None
        """
        url = QtCore.QUrl.fromLocalFile(cwd + "/mmdetection/checkpoints/")
        if not QtGui.QDesktopServices.openUrl(url):
            # Print an error message if opening failed
            print("Failed to open checkpoints directory")
