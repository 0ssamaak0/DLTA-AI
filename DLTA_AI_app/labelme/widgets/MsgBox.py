from PyQt6 import QtWidgets


def OKmsgBox(title, text, type = "info", turnResult = False):
    
    """
    Show a message box.

    Args:
        title (str): The title of the message box.
        text (str): The text of the message box.
        type (str, optional): The type of the message box. Can be "info", "warning", or "critical". Defaults to "info".

    Returns:
        int: The result of the message box. This will be the value of the button clicked by the user.
    """
    
    msgBox = QtWidgets.QMessageBox()
    if type == "info":
        msgBox.setIcon(QtWidgets.QMessageBox.Information)
    elif type == "warning":
        msgBox.setIcon(QtWidgets.QMessageBox.Warning)
    elif type == "critical":
        msgBox.setIcon(QtWidgets.QMessageBox.Critical)
    msgBox.setText(text)
    msgBox.setWindowTitle(title)
    if turnResult:
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        msgBox.setDefaultButton(QtWidgets.QMessageBox.Ok)
    else:
        msgBox.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msgBox.exec_()
    return msgBox.result()






