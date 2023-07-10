from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets



def PopUp(config):
    
    """
    Summary:
        Show a dialog to choose the interpolation options.
        (   interpolate only missed frames between detected frames, 
            interpolate all frames between your KEY frames, 
            interpolate ALL frames with SAM (more precision, more time) )
            
    Args:
        config: a dictionary of configurations
        
    Returns:
        result: the result of the dialog
        config: the updated dictionary of configurations
    """
    
    def show_unshow_overwrite():
        if with_sam.isChecked():
            config.update({'interpolationDefMethod': 'SAM'})
            overwrite_checkBox.setEnabled(True)
        else:
            config.update({'interpolationDefMethod': 'Linear'})
            overwrite_checkBox.setEnabled(False)
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Interpolation Options")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(250, 100)
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("Choose Interpolation Options")
    label.setFont(QtGui.QFont("", 10))
    method_label = QtWidgets.QLabel("Interpolation Method")
    between_label = QtWidgets.QLabel("Interpolation Between")

    layout.addWidget(label)
    

    # Interpolation Method button group
    method_group = QtWidgets.QButtonGroup()
    with_linear = QtWidgets.QRadioButton("Linear Interpolation")
    with_sam = QtWidgets.QRadioButton("SAM Interpolation")
    method_group.addButton(with_linear)
    method_group.addButton(with_sam)


    with_linear.toggled.connect(show_unshow_overwrite)
    with_sam.toggled.connect(show_unshow_overwrite)

    layout.addWidget(method_label)
    method_layout = QtWidgets.QHBoxLayout()
    method_layout.addWidget(with_linear)
    method_layout.addWidget(with_sam)
    layout.addLayout(method_layout)

    # Keyframes button group
    between_group = QtWidgets.QButtonGroup()
    with_keyframes = QtWidgets.QRadioButton("Selected Keyframes")
    without_keyframes = QtWidgets.QRadioButton("Detected Frames")
    between_group.addButton(with_keyframes)
    between_group.addButton(without_keyframes)

    with_keyframes.toggled.connect(lambda: config.update({'interpolationDefType': 'key' * with_keyframes.isChecked()}))
    without_keyframes.toggled.connect(lambda: config.update({'interpolationDefType': 'all' * without_keyframes.isChecked()}))

    layout.addWidget(between_label)
    keyframes_layout = QtWidgets.QHBoxLayout()
    keyframes_layout.addWidget(with_keyframes)
    keyframes_layout.addWidget(without_keyframes)
    layout.addLayout(keyframes_layout)
    
    overwrite_checkBox = QtWidgets.QCheckBox("Overwrite used frames with SAM")
    overwrite_checkBox.setChecked(config['interpolationOverwrite'])
    overwrite_checkBox.toggled.connect(lambda: config.update({'interpolationOverwrite': overwrite_checkBox.isChecked()}))
    layout.addWidget(overwrite_checkBox)
    
    show_unshow_overwrite()
    # for some reason you must check linear then sam to make it work
    with_linear.setChecked(True)

    buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    layout.addWidget(buttonBox)

    dialog.setLayout(layout)
    result = dialog.exec_()
    return result, config

