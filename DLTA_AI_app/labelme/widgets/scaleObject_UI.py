from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets
from labelme.utils.helpers.mathOps import scaleQTshape




def PopUp(self):
    
    """
    Summary:
        Show a dialog to scale a shape.
        
    Args:
        self: the main window object to access the canvas
        
    Returns:
        result: the result of the dialog
    """
    
    originalshape = self.canvas.selectedShapes[0].copy()
    xx = [originalshape.points[i].x()
            for i in range(len(originalshape.points))]
    yy = [originalshape.points[i].y()
            for i in range(len(originalshape.points))]
    center = [sum(xx) / len(xx), sum(yy) / len(yy)]

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Scaling")
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.resize(400, 400)

    layout = QtWidgets.QVBoxLayout()

    label = QtWidgets.QLabel(
        "Scaling object with ID: " + str(originalshape.group_id) + "\n ")
    label.setStyleSheet(
        "QLabel { font-weight: bold; }")
    layout.addWidget(label)

    xLabel = QtWidgets.QLabel()
    xLabel.setText("Width(x) factor is: " + "100" + "%")
    yLabel = QtWidgets.QLabel()
    yLabel.setText("Hight(y) factor is: " + "100" + "%")

    xSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
    xSlider.setMinimum(50)
    xSlider.setMaximum(150)
    xSlider.setValue(100)
    xSlider.setTickPosition(
        QtWidgets.QSlider.TickPosition.TicksBelow)
    xSlider.setTickInterval(1)
    xSlider.setMaximumWidth(750)
    xSlider.valueChanged.connect(lambda: xLabel.setText(
        "Width(x) factor is: " + str(xSlider.value()) + "%"))
    xSlider.valueChanged.connect(lambda: scaleQTshape(self,
        originalshape, center, xSlider.value(), ySlider.value()))

    ySlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
    ySlider.setMinimum(50)
    ySlider.setMaximum(150)
    ySlider.setValue(100)
    ySlider.setTickPosition(
        QtWidgets.QSlider.TickPosition.TicksBelow)
    ySlider.setTickInterval(1)
    ySlider.setMaximumWidth(750)
    ySlider.valueChanged.connect(lambda: yLabel.setText(
        "Hight(y) factor is: " + str(ySlider.value()) + "%"))
    ySlider.valueChanged.connect(lambda: scaleQTshape(self,
        originalshape, center, xSlider.value(), ySlider.value()))

    layout.addWidget(xLabel)
    layout.addWidget(yLabel)
    layout.addWidget(xSlider)
    layout.addWidget(ySlider)

    buttonBox = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok)
    buttonBox.accepted.connect(dialog.accept)
    layout.addWidget(buttonBox)
    dialog.setLayout(layout)
    result = dialog.exec()
    return result    

