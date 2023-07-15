from labelme.widgets.links import open_release
from bs4 import BeautifulSoup
import requests
from PyQt6.QtWidgets import QMessageBox, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6 import QtWidgets
import time
    
    
def PopUp():
    """
    Check for updates of DLTA-AI and display a message box with the result.

    The function checks the latest release of DLTA-AI on GitHub and compares it with the current version.
    If the latest release is newer than the current version, a message box is displayed with a button to
    download the latest version. Otherwise, a message box is displayed indicating that the user is using
    the latest version.

    Args:
        None

    Returns:
        None
    """

    # Import the current version of DLTA-AI
    from labelme import __version__

    # Initialize variables
    updates = False
    tag = {}
    tag["href"] = None

    try:
        # Get the HTML content of the releases page on GitHub
        url = "https://github.com/0ssamaak0/DLTA-AI/releases"
        html = requests.get(url, timeout=5).text
        soup = BeautifulSoup(html, "html.parser")

        # Find the first <a> tag with class="Link--primary"
        tag = soup.find("a", class_="Link--primary")

        # Split the tag text on the first "v" to get the latest version number
        lastest_version = tag.text.lower().split("v")[1]

        # Compare the latest version with the current version
        if lastest_version != __version__:
            text = f"New version of DLTA-AI (v{lastest_version}) is available.\n You are currently using (v{__version__})\n"
            updates = True
        else:
            text = f"you are using the latest version of DLTA-AI (v{__version__})\n"
    except:
        text = f"You are using DLTA-AI (v{__version__})\n There was an error checking for updates.\n"

    # Create a message box with the result
    msgBox = QMessageBox()
    msgBox.setWindowTitle("Check for Updates")
    msgBox.setFont(QFont("Arial", 10))  # Set the font size to 10

    # Add the text label to the message box
    msgBox.setText(text)

    # If there are updates, add a button to download the latest version
    if updates:
        msgBox.addButton(QMessageBox.Yes)
        msgBox.button(QMessageBox.Yes).setText("Get the Latest Version")
        msgBox.button(QMessageBox.Yes).clicked.connect(lambda: open_release(tag["href"]))

    # Add a close button to the message box
    msgBox.addButton(QMessageBox.Close)
    msgBox.button(QMessageBox.Close).setText("Close")

    # Display the message box
    msgBox.exec_()
