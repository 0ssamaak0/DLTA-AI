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
    msgBox.addButton(QMessageBox.StandardButton.Yes)
    msgBox.button(QMessageBox.StandardButton.Yes).setText("Open an Issue")
    msgBox.button(QMessageBox.StandardButton.Yes).clicked.connect(open_issue)

    # Add a close button
    msgBox.addButton(QMessageBox.StandardButton.Close)
    msgBox.button(QMessageBox.StandardButton.Close).setText("Close")

    # Display the feedback dialog box
    msgBox.exec()
