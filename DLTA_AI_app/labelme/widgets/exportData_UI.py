from PyQt6.QtCore import Qt
from PyQt6 import QtGui
from PyQt6 import QtWidgets
from labelme.widgets import open_file

try:
    from labelme.utils.custom_exports import custom_exports_list
except:
    custom_exports_list = []
    print("custom_exports file not found")




def PopUp(mode = "video"):
    """
    Displays a dialog box for choosing export options for annotations and videos.

    Args:
        mode (str): The mode of the export. Can be either "video" or "image". Defaults to "video".

    Returns:
        A tuple containing the result of the dialog box and the selected export options. If the dialog box is accepted, the first element of the tuple is `QtWidgets.QDialog.DialogCode.Accepted`. Otherwise, it is `QtWidgets.QDialog.Rejected`. The second element of the tuple is a boolean indicating whether to export annotations in COCO format. If `mode` is "video", the third element of the tuple is a boolean indicating whether to export annotations in MOT format, and the fourth element is a boolean indicating whether to export the video with the current visualization settings. If there are any custom export options available, the fifth element of the tuple is a list of booleans indicating whether to export using each custom export option.
    """

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Export Options")
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.resize(250, 100)

    layout = QtWidgets.QVBoxLayout()

    font = QtGui.QFont()
    font.setBold(True)
    font.setPointSize(10)

    if mode == "video":
        vid_label = QtWidgets.QLabel("Export Video")
        vid_label.setFont(font)
        vid_label.setMargin(7)

    std_label = QtWidgets.QLabel("Export Annotations (Standard Formats)")
    std_label.setFont(font)
    std_label.setMargin(7)

    custom_label = QtWidgets.QLabel("Export Annotations (Custom Formats)")
    custom_label.setFont(font)
    custom_label.setMargin(7)
    
    # Create a button group to hold the radio buttons
    button_group = QtWidgets.QButtonGroup()

    # Create the radio buttons and add them to the button group
    coco_radio = QtWidgets.QRadioButton(
        "COCO Format (Detection / Segmentation)")
    
    # make the video and mot radio buttons if the mode is video
    if mode == "video":
        video_radio = QtWidgets.QRadioButton("Export Video with current visualization settings")
        mot_radio = QtWidgets.QRadioButton("MOT Format (Tracking)")

    # make the custom exports radio buttons
    custom_exports_radio_list = []
    if len(custom_exports_list) != 0:
        for custom_exp in custom_exports_list:
            if custom_exp.mode == "video" and mode == "video":
                custom_radio = QtWidgets.QRadioButton(custom_exp.button_name)
                button_group.addButton(custom_radio)
                custom_exports_radio_list.append(custom_radio)
            if custom_exp.mode == "image" and mode == "image":
                custom_radio = QtWidgets.QRadioButton(custom_exp.button_name)
                button_group.addButton(custom_radio)
                custom_exports_radio_list.append(custom_radio)
            
    
    button_group.addButton(coco_radio)
    
    # add the video and mot radio buttons to the button group if the mode is video
    if mode == "video":
        button_group.addButton(video_radio)
        button_group.addButton(mot_radio)

    # Add custom radio buttons to the button group
    if len(custom_exports_list) != 0:
        for custom_radio in custom_exports_radio_list:
            button_group.addButton(custom_radio)

    # Add to the layout

    # video label and radio buttons
    if mode == "video":
        layout.addWidget(vid_label)
        layout.addWidget(video_radio)

    # standard label and radio buttons
    layout.addWidget(std_label)
    layout.addWidget(coco_radio)
    if mode == "video":
        layout.addWidget(mot_radio)

    # custom label and radio buttons
    layout.addWidget(custom_label)
    if len(custom_exports_radio_list) != 0:
        for custom_radio in custom_exports_radio_list:
            layout.addWidget(custom_radio)
    else:
        layout.addWidget(QtWidgets.QLabel("No Custom Exports Available, you can add them in utils.custom_exports.py"))

    # create button when clicking it open custom_exports.py file
    custom_exports_button = QtWidgets.QPushButton("Open Custom Exports")
    custom_exports_button.clicked.connect(open_file.PopUp)
    layout.addWidget(custom_exports_button)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttonBox.accepted.connect(dialog.accept)
    buttonBox.rejected.connect(dialog.reject)

    layout.addWidget(buttonBox)

    dialog.setLayout(layout)

    result = dialog.exec()

    # prepare the checked list of custom exports
    custom_exports_radio_checked_list = []
    if len(custom_exports_list) != 0:
        for custom_radio in custom_exports_radio_list:
            custom_exports_radio_checked_list.append(custom_radio.isChecked())
    
    if mode == "video":
        return result, coco_radio.isChecked(), mot_radio.isChecked(), video_radio.isChecked(), custom_exports_radio_checked_list
    else:
        return result, coco_radio.isChecked(), custom_exports_radio_checked_list
