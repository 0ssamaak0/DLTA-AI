from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets


class ThresholdWidget(QtWidgets.QDialog):
    def __init__(self):
        super(ThresholdWidget, self).__init__()
        self.setModal(True)
        self.setWindowTitle("Enter Threshold")
        self.threshold = 0.5
        self.threshold = self._createQLineEdit()
    
    def _createQLineEdit(self):
        threshold = QtWidgets.QLineEdit()
        threshold.setRange(0, 1)
        threshold.setValue(0.5)
        threshold.valueChanged.connect(self.onNewValue)
        return threshold
