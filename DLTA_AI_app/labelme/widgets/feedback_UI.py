from labelme.widgets.links import open_issue
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import Qt


def PopUp():
    """
    Displays a dialog box for providing feedback on the DLTA-AI project.

    Parameters:
    None

    Returns:
    None
    """
    # Define the text for the feedback dialog box
    text = "Found a bug? 🐞\nWant to suggest a feature? 🌟\n"
    
    # Create the feedback dialog box
    msgBox = QMessageBox()
    msgBox.setWindowTitle("Feedback")
    msgBox.setText(text)

    # Add a button to open the GitHub issues page
    msgBox.addButton(QMessageBox.Yes)
    msgBox.button(QMessageBox.Yes).setText("Open an Issue")
    msgBox.button(QMessageBox.Yes).clicked.connect(open_issue)

    # Add a close button
    msgBox.addButton(QMessageBox.Close)
    msgBox.button(QMessageBox.Close).setText("Close")

    # Display the feedback dialog box
    msgBox.exec_()
