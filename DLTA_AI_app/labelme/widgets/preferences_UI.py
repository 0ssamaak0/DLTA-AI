import yaml
from PyQt6 import QtWidgets, QtGui, QtCore





def PopUp():
    """

    Description:
    This function displays a dialog box with preferences for the LabelMe application, including theme and notification settings.

    Parameters:
    This function takes no parameters.

    Returns:
    If the user clicks the OK button, this function writes the new theme and notification settings to the config file and returns `QtWidgets.QDialog.DialogCode.Accepted`. If the user clicks the Cancel button, this function does not write any changes to the config file and returns `QtWidgets.QDialog.Rejected`.

    Libraries:
    This function requires the following libraries to be installed:
    - yaml
    - PyQt6.QtWidgets
    - PyQt6.QtGui
    - PyQt6.QtCore
    """
    

    with open("labelme/config/default_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create the dialog
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Preferences")
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)

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
    if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
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
