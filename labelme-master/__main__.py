import os.path as osp
import sys

from qtpy import QtCore
from qtpy import QtWidgets

from labelme import __appname__
from labelme import __version__
from labelme.app import MainWindow

from labelme.utils import newIcon

try:
    import qdarktheme
except ImportError:
    pass


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(__appname__)

    # style
    app.setWindowIcon(newIcon("icon"))
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
        qdarktheme.setup_theme("auto", additional_qss=qss)
    except Exception as e:
        pass

    win = MainWindow()

    win.show()
    win.raise_()
    sys.exit(app.exec_())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
