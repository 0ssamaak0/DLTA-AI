from qtpy.QtCore import Qt
from qtpy import QtWidgets
from .MsgBox import OKmsgBox
from labelme.utils.helpers.mathOps import is_id_repeated


def PopUp(self, group_id, text):
    
    """
    Summary:
        Show a dialog to get a new id from the user.
        check if the id is repeated.
        
    Args:
        self: the main window object to access the canvas
        group_id: the group id
        text: Class name
        
    Returns:
        group_id: the new group id
        text: Class name (False if the user-input id is repeated)
    """    
    
    mainTEXT = "A Shape with that ID already exists in this frame.\n\n"
    repeated = 0

    while is_id_repeated(self, group_id):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("ID already exists")
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.resize(450, 100)

        if repeated == 0:
            label = QtWidgets.QLabel(mainTEXT + f'Please try a new ID: ')
        if repeated == 1:
            label = QtWidgets.QLabel(
                mainTEXT + f'OH GOD.. AGAIN? I hpoe you are not doing this on purpose..')
        if repeated == 2:
            label = QtWidgets.QLabel(
                mainTEXT + f'AGAIN? REALLY? LAST time for you..')
        if repeated == 3:
            text = False
            return group_id, text

        properID = QtWidgets.QSpinBox()
        properID.setRange(1, 1000)

        buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok)
        buttonBox.accepted.connect(dialog.accept)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(properID)
        layout.addWidget(buttonBox)
        dialog.setLayout(layout)
        result = dialog.exec_()
        if result != QtWidgets.QDialog.Accepted:
            text = False
            return group_id, text

        group_id = properID.value()
        repeated += 1

    if repeated > 1:
        OKmsgBox("Finally..!", "OH, Finally..!")

    return group_id, text



