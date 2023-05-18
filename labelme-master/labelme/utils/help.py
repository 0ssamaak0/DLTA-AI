from .helpers import OKmsgBox

def showRuntimeData(self):
    import torch
    runtime = ""
    # if cuda is available, print cuda name
    if torch.cuda.is_available():
        runtime = ("Labelmm is Using GPU\n")
        runtime += ("GPU Name: " + torch.cuda.get_device_name(0))
        runtime += "\n" + "Total GPU VRAM: " + \
            str(round(torch.cuda.get_device_properties(
                0).total_memory/(1024 ** 3), 2)) + " GB"
        runtime += "\n" + "Used: " + \
            str(round(torch.cuda.memory_allocated(0)/(1024 ** 3), 2)) + " GB"

    else:
        # runtime = (f'Labelmm is Using CPU: {platform.processor()}')
        runtime = ('Labelmm is Using CPU')
    


    # show runtime in QMessageBox
    OKmsgBox("Runtime Data", runtime)
             

def git_hub_link():
    import webbrowser
    webbrowser.open('https://github.com/0ssamaak0/labelmm')  
    return

def open_issue():
    import webbrowser
    webbrowser.open('https://github.com/0ssamaak0/labelmm/issues')
    return
                    
def open_license():
    import webbrowser
    webbrowser.open('https://github.com/0ssamaak0/labelmm/blob/master/LICENSE')
    return

def feedback():
    text = "Found a bug? 🐞\nWant to suggest a feature? 🌟\n"

    from PyQt5.QtWidgets import QMessageBox
    from PyQt5.QtCore import Qt

    msgBox = QMessageBox()
    msgBox.setWindowTitle("Feedback")
    msgBox.setText(text)

    # add button when clicking on it will open github issues page
    msgBox.addButton(QMessageBox.Yes)
    msgBox.button(QMessageBox.Yes).setText("Open an Issue")
    msgBox.button(QMessageBox.Yes).clicked.connect(open_issue)

    # add close button
    msgBox.addButton(QMessageBox.Close)
    msgBox.button(QMessageBox.Close).setText("Close")

    msgBox.exec_()

def version():
    OKmsgBox("Version", "Labelmm Version 1.0.0 Pre-Release")
