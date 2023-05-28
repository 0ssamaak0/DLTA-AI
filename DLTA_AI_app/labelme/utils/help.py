from .helpers import OKmsgBox

def show_runtime_data():
    """
    Function Name: show_runtime_data()

    Description:
    This function displays a dialog box with information about the runtime data of the system, including GPU and RAM stats.

    Parameters:
    This function takes no parameters.

    Returns:
    This function does not return anything.

    Libraries:
    This function requires the following libraries to be installed:
    - PyQt5
    - psutil
    - torch
    """
    # Import necessary modules
    from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout
    from PyQt5.QtGui import QFont
    import psutil
    import torch

    # Create a dialog box to display the runtime data
    dialog = QDialog()
    dialog.setWindowTitle("Runtime data")
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(10)

    # Set font styles for the title and normal text
    title_font = QFont()
    title_font.setPointSize(12)
    title_font.setBold(True)

    normal_font = QFont()
    normal_font.setPointSize(10)

    # If CUDA is available, display GPU stats
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_title_label = QLabel("Device Stats")
        gpu_title_label.setFont(title_font)
        layout.addWidget(gpu_title_label)

        gpu_name_label = QLabel(f"GPU Name: {device_name}")
        gpu_name_label.setFont(normal_font) 
        layout.addWidget(gpu_name_label)

        total_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        used_vram = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
        gpu_vram_label = QLabel(f"Total GPU VRAM: {total_vram} GB\nUsed: {used_vram} GB")
        gpu_vram_label.setFont(normal_font)
        layout.addWidget(gpu_vram_label)

    # If CUDA is not available, display CPU stats
    else:
        cpu_label = QLabel("DLTA-AI is Using CPU")
        cpu_label.setFont(title_font)
        layout.addWidget(cpu_label)

    # Display RAM stats
    ram_title_label = QLabel("RAM Stats")
    ram_title_label.setFont(title_font)
    layout.addWidget(ram_title_label)

    total_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    used_ram = round(psutil.virtual_memory().used / (1024 ** 3), 2)
    ram_label = QLabel(f"Total RAM: {total_ram} GB\nUsed: {used_ram} GB")
    ram_label.setFont(normal_font)
    layout.addWidget(ram_label)

    # Display the dialog box
    dialog.exec_()
             
def preferences():
    """
    Function Name: preferences()

    Description:
    This function displays a dialog box with preferences for the LabelMe application, including theme and notification settings.

    Parameters:
    This function takes no parameters.

    Returns:
    If the user clicks the OK button, this function writes the new theme and notification settings to the config file and returns `QtWidgets.QDialog.Accepted`. If the user clicks the Cancel button, this function does not write any changes to the config file and returns `QtWidgets.QDialog.Rejected`.

    Libraries:
    This function requires the following libraries to be installed:
    - yaml
    - PyQt5.QtWidgets
    - PyQt5.QtGui
    - PyQt5.QtCore
    """
    import yaml
    from PyQt5 import QtWidgets, QtGui, QtCore

    with open("labelme/config/default_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create the dialog
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Preferences")

    # Create the labels
    themeLabel = QtWidgets.QLabel("Theme Settings 🌓")
    themeLabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    theme_note_label = QtWidgets.QLabel("Requires app restart to take effect")

    notificationLabel = QtWidgets.QLabel("Notifications Settings 🔔")
    notificationLabel.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
    notification_note_label = QtWidgets.QLabel("Notifications works only for long tasks and if the app isn't focused")

    # Load the current theme from the config file
    current_theme = config["theme"]
    current_mute =  config["mute"]

    # Create the radio buttons
    autoButton = QtWidgets.QRadioButton("OS Default")
    lightButton = QtWidgets.QRadioButton("Light")
    darkButton = QtWidgets.QRadioButton("Dark")

    # Set the current theme as the default selection
    if current_theme == "auto":
        autoButton.setChecked(True)
    elif current_theme == "light":
        lightButton.setChecked(True)
    elif current_theme == "dark":
        darkButton.setChecked(True)

    # Create the images
    autoImage = QtGui.QPixmap("labelme/icons/auto-img.png").scaledToWidth(128)
    lightImage = QtGui.QPixmap("labelme/icons/light-img.png").scaledToWidth(128)
    darkImage = QtGui.QPixmap("labelme/icons/dark-img.png").scaledToWidth(128)

    # Create the image labels
    autoLabel = QtWidgets.QLabel()
    autoLabel.setPixmap(autoImage)
    lightLabel = QtWidgets.QLabel()
    lightLabel.setPixmap(lightImage)
    darkLabel = QtWidgets.QLabel()
    darkLabel.setPixmap(darkImage)

    # Create the layout
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(themeLabel)
    layout.addWidget(theme_note_label)
    buttonLayout = QtWidgets.QHBoxLayout()
    buttonLayout.addWidget(autoButton)
    buttonLayout.addWidget(lightButton)
    buttonLayout.addWidget(darkButton)
    layout.addLayout(buttonLayout)

    # Create the image layout
    imageLayout = QtWidgets.QHBoxLayout()
    imageLayout.addWidget(autoLabel)
    imageLayout.addWidget(lightLabel)
    imageLayout.addWidget(darkLabel)
    layout.addLayout(imageLayout)

    # Create the notification checkbox
    notificationCheckbox = QtWidgets.QCheckBox("Mute Notifications")
    notificationCheckbox.setChecked(current_mute)
    layout.addWidget(notificationLabel)
    layout.addWidget(notification_note_label)
    layout.addWidget(notificationCheckbox)

    dialog.setLayout(layout)

    # Create the OK and Cancel buttons
    okButton = QtWidgets.QPushButton("OK")
    cancelButton = QtWidgets.QPushButton("Cancel")

    # Add the buttons to a QHBoxLayout
    buttonLayout = QtWidgets.QHBoxLayout()
    buttonLayout.addWidget(okButton)
    buttonLayout.addWidget(cancelButton)

    # Add the QHBoxLayout to the QVBoxLayout
    layout.addLayout(buttonLayout)

    # Connect the OK and Cancel buttons to the accept and reject functions
    okButton.clicked.connect(dialog.accept)
    cancelButton.clicked.connect(dialog.reject)

    # Show the dialog
    if dialog.exec_() == QtWidgets.QDialog.Accepted:
        # Write the new theme and notification settings to the config file
        if autoButton.isChecked():
            theme = "auto"
        elif lightButton.isChecked():
            theme = "light"
        elif darkButton.isChecked():
            theme = "dark"
        mute = notificationCheckbox.isChecked()
        with open("labelme/config/default_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["theme"] = theme
        config["mute"] = mute
        with open("labelme/config/default_config.yaml", "w") as f:
            yaml.dump(config, f)

    

def shortcut_selector():
    """
    Displays a dialog box for selecting and editing keyboard shortcuts for the application.

    Parameters:
    None

    Returns:
    None
    """
    # Import necessary modules
    import yaml
    from PyQt5 import QtWidgets, QtGui, QtCore

    # Load the default shortcuts from the config file
    shortcuts = {}
    with open("labelme/config/default_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        shortcuts = config.get("shortcuts", {})

    # Encode the shortcut names for display in the table
    shortcuts_names_encode = {name: name.lower().capitalize().replace("_", " ").replace("Sam", "SAM").replace("sam", "SAM") for name in shortcuts.keys()}

    # Decode the shortcut names back to their original form
    shortcuts_names_decode = {value: key for key, value in shortcuts_names_encode.items()}

    # Change the keys of the shortcuts dictionary to use the encoded names
    shortcuts = {shortcuts_names_encode[key]: value for key, value in shortcuts.items()}

    # Create a table to display the shortcuts
    shortcut_table = QtWidgets.QTableWidget()
    shortcut_table.setColumnCount(2)
    shortcut_table.setHorizontalHeaderLabels(['Name', 'Shortcut'])
    shortcut_table.setRowCount(len(shortcuts))
    shortcut_table.verticalHeader().setVisible(False)

    # Populate the table with the shortcut names and keys
    row = 0
    for name, key in shortcuts.items():
        name_item = QtWidgets.QTableWidgetItem(name)
        shortcut_item = QtWidgets.QTableWidgetItem(key)
        shortcut_table.setItem(row, 0, name_item)
        shortcut_table.setItem(row, 1, shortcut_item)
        row += 1

    # Define a function to handle clicks on the shortcut table
    def on_shortcut_table_clicked(item):
        row = item.row()
        name_item = shortcut_table.item(row, 0)
        name = name_item.text()
        current_key = shortcuts[name]
        key_edit = QtWidgets.QKeySequenceEdit(QtGui.QKeySequence(current_key))
        key_edit.setWindowTitle(f"Edit Shortcut for {name}")
        key_edit_label = QtWidgets.QLabel("Enter new shortcut for " + name)
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Shortcut Selector")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(key_edit_label)
        layout.addWidget(key_edit)
        ok_button = QtWidgets.QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        null_hint_label = QtWidgets.QLabel("to remove shortcut, press 'Ctrl' only then click 'OK") 
        layout.addWidget(ok_button)
        layout.addWidget(null_hint_label)
        dialog.setLayout(layout)

        # If the user clicks OK, update the shortcut and table
        if dialog.exec_():
            key = key_edit.keySequence().toString(QtGui.QKeySequence.NativeText)
            if key in shortcuts.values() and list(shortcuts.keys())[list(shortcuts.values()).index(key)] != name:
                conflicting_shortcut = list(shortcuts.keys())[list(shortcuts.values()).index(key)]
                QtWidgets.QMessageBox.warning(None, "Error", f"{key} is already assigned to {conflicting_shortcut}.")
            else:
                if key == "":
                    key = None
                shortcuts[name] = key
                shortcut_table.item(row, 1).setText(key)

    # Connect the on_shortcut_table_clicked function to the itemClicked signal of the shortcut table
    shortcut_table.itemClicked.connect(on_shortcut_table_clicked)

    # Create a dialog box to display the shortcut table
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Shortcuts")
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(shortcut_table)
    ok_button = QtWidgets.QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)
    layout.addWidget(ok_button)
    note_label = QtWidgets.QLabel("Shortcuts will be updated after restarting the app.")
    layout.addWidget(note_label)
    dialog.setLayout(layout)

    # Set the size of the dialog box
    dialog.setFixedWidth(shortcut_table.sizeHintForColumn(0) + shortcut_table.sizeHintForColumn(1) + 50)
    dialog.setMinimumHeight(shortcut_table.rowHeight(0) * 10 + 50)

    # Set the size policy to allow vertical resizing
    dialog.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

    # Display the dialog box
    dialog.exec_()

    # Decode the shortcut names back to their original form
    shortcuts = {shortcuts_names_decode[key]: value for key, value in shortcuts.items()}

    # Write the updated shortcuts to the config file
    with open("labelme/config/default_config.yaml", "w") as f:
        config["shortcuts"] = shortcuts
        yaml.dump(config, f)



def git_hub_link():
    """
    Opens the GitHub repository for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """
    # Import necessary modules
    import webbrowser

    # Open the GitHub repository in the default web browser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI')  

def open_issue():
    """
    Opens the GitHub issues page for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """
    # Import necessary modules
    import webbrowser

    # Open the GitHub issues page in the default web browser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI/issues')

def open_license():
    """
    Opens the license file for the DLTA-AI project in the default web browser.

    Parameters:
    None

    Returns:
    None
    """
    # Import necessary modules
    import webbrowser

    # Open the license file in the default web browser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI/blob/master/LICENSE')

def feedback():
    """
    Displays a dialog box for providing feedback on the DLTA-AI project.

    Parameters:
    None

    Returns:
    None
    """
    # Define the text for the feedback dialog box
    text = "Found a bug? 🐞\nWant to suggest a feature? 🌟\n"

    # Import necessary modules
    from PyQt5.QtWidgets import QMessageBox
    from PyQt5.QtCore import Qt

    # Create the feedback dialog box
    msgBox = QMessageBox()
    msgBox.setWindowTitle("Feedback")
    msgBox.setText(text)

    # Add a button to open the GitHub issues page
    msgBox.addButton(QMessageBox.Yes)
    msgBox.button(QMessageBox.Yes).setText("Open an Issue")
    msgBox.button(QMessageBox.Yes).clicked.connect(open_issue)

    # Add a close button
    msgBox.addButton(QMessageBox.Close)
    msgBox.button(QMessageBox.Close).setText("Close")

    # Display the feedback dialog box
    msgBox.exec_()

def version():
    """
    Displays the version number of the DLTA-AI project.

    Parameters:
    None

    Returns:
    None
    """
    from labelme import __version__
    # Display the version number in a message box
    OKmsgBox("Version", f"DLTA-AI Version {__version__}")
