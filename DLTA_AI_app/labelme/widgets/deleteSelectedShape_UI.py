from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets


def PopUp(TOTAL_VIDEO_FRAMES, INDEX_OF_CURRENT_FRAME, config):
    
    """
    Summary:
        Show a dialog to choose the deletion options.
        (   This Frame and All Previous Frames,
            This Frame and All Next Frames,
            All Frames,
            This Frame Only,
            Specific Range of Frames           )
            
    Args:
        TOTAL_VIDEO_FRAMES: the total number of frames
        config: a dictionary of configurations
        
    Returns:
        result: the result of the dialog
        config: the updated dictionary of configurations
        fromFrameVAL: the start frame of the deletion range
        toFrameVAL: the end frame of the deletion range
    """
    
    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Choose Deletion Options")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(500, 100)
    dialog.setWindowFlags(dialog.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel("Choose Deletion Options")
    layout.addWidget(label)

    prev = QtWidgets.QRadioButton("This Frame and All Previous Frames")
    next = QtWidgets.QRadioButton("This Frame and All Next Frames")
    all = QtWidgets.QRadioButton(
        "All Frames")
    only = QtWidgets.QRadioButton("This Frame Only")

    from_to = QtWidgets.QRadioButton(
        "Specific Range of Frames")
    from_frame = QtWidgets.QSpinBox()
    to_frame = QtWidgets.QSpinBox()
    from_frame.setRange(1, TOTAL_VIDEO_FRAMES)
    to_frame.setRange(1, TOTAL_VIDEO_FRAMES)
    from_frame.valueChanged.connect(lambda: from_to.toggle())
    to_frame.valueChanged.connect(lambda: from_to.toggle())

    from_label = QtWidgets.QLabel("From:")
    to_label = QtWidgets.QLabel("To:")

    if config['deleteDefault'] == 'This Frame and All Previous Frames':
        prev.toggle()
    if config['deleteDefault'] == 'This Frame and All Next Frames':
        next.toggle()
    if config['deleteDefault'] == 'All Frames':
        all.toggle()
    if config['deleteDefault'] == 'This Frame Only':
        only.toggle()
    if config['deleteDefault'] == 'Specific Range of Frames':
        from_to.toggle()

    prev.toggled.connect(lambda: config.update(
        {'deleteDefault': 'This Frame and All Previous Frames'}))
    next.toggled.connect(lambda: config.update(
        {'deleteDefault': 'This Frame and All Next Frames'}))
    all.toggled.connect(lambda: config.update(
        {'deleteDefault': 'All Frames'}))
    only.toggled.connect(lambda: config.update(
        {'deleteDefault': 'This Frame Only'}))
    from_to.toggled.connect(lambda: config.update(
        {'deleteDefault': 'Specific Range of Frames'}))


    button_layout = QtWidgets.QHBoxLayout()
    button_layout.addWidget(only)
    button_layout.addWidget(all)
    layout.addLayout(button_layout)

    button_layout = QtWidgets.QHBoxLayout()
    button_layout.addWidget(prev)
    button_layout.addWidget(next)
    layout.addLayout(button_layout)

    layout.addWidget(from_to)

    button_layout = QtWidgets.QHBoxLayout()
    button_layout.addWidget(from_label)
    button_layout.addWidget(from_frame)
    button_layout.addWidget(to_label)
    button_layout.addWidget(to_frame)
    layout.addLayout(button_layout)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    buttonBox.rejected.connect(dialog.reject)
    layout.addWidget(buttonBox)
    dialog.setLayout(layout)
    result = dialog.exec_()
    
    mode = config['deleteDefault']
    fromFrameVAL = from_frame.value() 
    toFrameVAL  = to_frame.value()
    
    if mode == 'This Frame and All Previous Frames':
        toFrameVAL = INDEX_OF_CURRENT_FRAME
        fromFrameVAL = 1
    elif mode == 'This Frame and All Next Frames':
        toFrameVAL = TOTAL_VIDEO_FRAMES
        fromFrameVAL = INDEX_OF_CURRENT_FRAME
    elif mode == 'This Frame Only':
        toFrameVAL = INDEX_OF_CURRENT_FRAME
        fromFrameVAL = INDEX_OF_CURRENT_FRAME
    elif mode == 'All Frames':
        toFrameVAL = TOTAL_VIDEO_FRAMES
        fromFrameVAL = 1
    
    return result, config, fromFrameVAL, toFrameVAL
