from PyQt6 import QtWidgets


class ColorDialog(QtWidgets.QColorDialog):
    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        self.setOption(QtWidgets.QColorDialog.ColorDialogOption.ShowAlphaChannel)
        # The Mac native dialog does not support our restore button.
        self.setOption(QtWidgets.QColorDialog.ColorDialogOption.DontUseNativeDialog)
        # Add a restore defaults button.
        # The default is set at invocation time, so that it
        # works across dialogs for different elements.
        self.default = None
        self.bb = self.findChild(QtWidgets.QDialogButtonBox)
        self.bb.addButton(QtWidgets.QDialogButtonBox.StandardButton.RestoreDefaults)
        self.bb.clicked.connect(self.checkRestore)

    def getColor(self, value=None, title=None, default=None):
        self.default = default
        if title:
            self.setWindowTitle(title)
        if value:
            self.setCurrentColor(value)
        return self.currentColor() if self.exec() else None

    def checkRestore(self, button):
        if (
            self.bb.buttonRole(button) & QtWidgets.QDialogButtonBox.ButtonRole.ResetRole
            and self.default
        ):
            self.setCurrentColor(self.default)
