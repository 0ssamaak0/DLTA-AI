from .helpers import OKmsgBox

def showRuntimeData(self):
    import torch
    runtime = ""
    # if cuda is available, print cuda name
    if torch.cuda.is_available():
        runtime = ("DLTA-AI is Using GPU\n")
        runtime += ("GPU Name: " + torch.cuda.get_device_name(0))
        runtime += "\n" + "Total GPU VRAM: " + \
            str(round(torch.cuda.get_device_properties(
                0).total_memory/(1024 ** 3), 2)) + " GB"
        runtime += "\n" + "Used: " + \
            str(round(torch.cuda.memory_allocated(0)/(1024 ** 3), 2)) + " GB"

    else:
        # runtime = (f'DLTA-AI is Using CPU: {platform.processor()}')
        runtime = ('DLTA-AI is Using CPU')
    


    # show runtime in QMessageBox
    OKmsgBox("Runtime Data", runtime)
             
def preferences():
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

    
import yaml
from PyQt5 import QtWidgets, QtGui, QtCore

def shortcut_selector():
    shortcuts = {}
    with open("labelme/config/default_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        shortcuts = config.get("shortcuts", {})

    shortcut_table = QtWidgets.QTableWidget()
    shortcut_table.setColumnCount(2)
    shortcut_table.setHorizontalHeaderLabels(['Name', 'Shortcut'])
    shortcut_table.setRowCount(len(shortcuts))
    shortcut_table.verticalHeader().setVisible(False)

    row = 0
    for name, key in shortcuts.items():
        name_item = QtWidgets.QTableWidgetItem(name)
        shortcut_item = QtWidgets.QTableWidgetItem(key)
        shortcut_table.setItem(row, 0, name_item)
        shortcut_table.setItem(row, 1, shortcut_item)
        row += 1

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
        layout.addWidget(ok_button)
        dialog.setLayout(layout)
        if dialog.exec_():
            key = key_edit.keySequence().toString(QtGui.QKeySequence.NativeText)
            if key in shortcuts.values() and list(shortcuts.keys())[list(shortcuts.values()).index(key)] != name:
                conflicting_shortcut = list(shortcuts.keys())[list(shortcuts.values()).index(key)]
                QtWidgets.QMessageBox.warning(None, "Error", f"{key} is already assigned to {conflicting_shortcut}.")
            else:
                shortcuts[name] = key
                name_item.setText(name)
                shortcut_table.item(row, 1).setText(key)

    shortcut_table.itemClicked.connect(on_shortcut_table_clicked)

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Shortcuts")
    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(shortcut_table)
    ok_button = QtWidgets.QPushButton("OK")
    ok_button.clicked.connect(dialog.accept)
    layout.addWidget(ok_button)
    note_label = QtWidgets.QLabel("Tooltips will show the new shortcuts after restarting the app")
    layout.addWidget(note_label)
    dialog.setLayout(layout)
    dialog.setFixedWidth(shortcut_table.sizeHintForColumn(0) + shortcut_table.sizeHintForColumn(1) + 40)
    dialog.setMinimumHeight(shortcut_table.rowHeight(0) * 10 + 50)
    

    # Set the size policy to allow vertical resizing
    dialog.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

    
    
    dialog.exec_()

    with open("labelme/config/default_config.yaml", "w") as f:
        config["shortcuts"] = shortcuts
        yaml.dump(config, f)



def git_hub_link():
    import webbrowser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI')  
    return

def open_issue():
    import webbrowser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI/issues')
    return
                    
def open_license():
    import webbrowser
    webbrowser.open('https://github.com/0ssamaak0/DLTA-AI/blob/master/LICENSE')
    return

def feedback():
    text = "Found a bug? 🐞\nWant to suggest a feature? 🌟\n"

    from PyQt5.QtWidgets import QMessageBox
    from PyQt5.QtCore import Qt

    msgBox = QMessageBox()
    msgBox.setWindowTitle("Feedback")
    msgBox.setText(text)

    # add button when clicking on it will open github issues page
    msgBox.addButton(QMessageBox.Yes)
    msgBox.button(QMessageBox.Yes).setText("Open an Issue")
    msgBox.button(QMessageBox.Yes).clicked.connect(open_issue)

    # add close button
    msgBox.addButton(QMessageBox.Close)
    msgBox.button(QMessageBox.Close).setText("Close")

    msgBox.exec_()

def version():
    OKmsgBox("Version", "DLTA-AI Version 1.0.0 Pre-Release")
