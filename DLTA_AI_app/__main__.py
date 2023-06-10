import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from qtpy import QtCore, QtGui, QtWidgets

from labelme import __appname__
from labelme import __version__
from labelme.utils import newIcon

import qdarktheme



def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("icon"))

    # create and show splash screen
    splash_pix = QtGui.QPixmap('labelme/icons/splash_screen.png')
    splash = QtWidgets.QSplashScreen(splash_pix)
    splash.show()

    qss = """
    QMenuBar::item {
        padding: 10px;
        margin: 0 5px
    }
    QMenu{
        border-radius: 5px;
    }
    QMenu::item{
        padding: 8px;
        margin: 5px;
        border-radius: 5px;
    }
    QToolTip {
            color: #111111;
            background-color: #EEEEEE;
            }
    QCheckBox{
        margin: 0 7px;
    }
    QComboBox{
        font-size: 10pt;
        font-weight: bold;
    }
    """
    try:
        import yaml
        with open ("labelme/config/default_config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        qdarktheme.setup_theme(theme = config["theme"], default_theme = "dark",  additional_qss=qss)
    except Exception as e:
        pass

    # create main window
    from labelme.app import MainWindow
    win = MainWindow()
    splash.finish(win)
    win.show()

    # close splash screen

    win.raise_()
    sys.exit(app.exec_())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()