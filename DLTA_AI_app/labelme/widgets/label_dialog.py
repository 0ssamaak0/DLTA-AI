import re

from qtpy import QT_VERSION
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from labelme.logger import logger
import labelme.utils


QT5 = QT_VERSION[0] == "5"


# TODO(unknown):
# - Calculate optimal position so as not to go out of screen area.


class LabelQLineEdit(QtWidgets.QLineEdit):
    def setListWidget(self, list_widget):
        self.list_widget = list_widget

    def keyPressEvent(self, e):
        if e.key() in [QtCore.Qt.Key_Up, QtCore.Qt.Key_Down]:
            self.list_widget.keyPressEvent(e)
        else:
            super(LabelQLineEdit, self).keyPressEvent(e)


class LabelDialog(QtWidgets.QDialog):
    def __init__(
        self,
        text="Enter object label",
        parent=None,
        labels=None,
        sort_labels=True,
        show_text_field=True,
        completion="startswith",
        fit_to_content=None,
        flags=None,
    ):
        if fit_to_content is None:
            fit_to_content = {"row": False, "column": True}
        self._fit_to_content = fit_to_content
        super(LabelDialog, self).__init__(parent)
        self.setWindowTitle("Edit Label")

        self.edit = LabelQLineEdit()
        self.edit.setPlaceholderText(text)
        self.edit.setValidator(labelme.utils.labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        if flags:
            self.edit.textChanged.connect(self.updateFlags)
        self.edit_group_id = QtWidgets.QLineEdit()
        self.edit_group_id.setPlaceholderText("Tracking ID")
        self.edit_group_id.setValidator(
            QtGui.QRegExpValidator(QtCore.QRegExp(r"\d*"), None)
        )

        self.edit_group_id_label = QtWidgets.QLabel()
        self.edit_group_id_label.setText("Tracking ID")

        self.select_class_label = QtWidgets.QLabel()
        self.select_class_label.setText("Class Name")

        # buttons
        self.buttonBox = bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        bb.button(bb.Ok).setIcon(labelme.utils.newIcon("done"))
        bb.button(bb.Cancel).setIcon(labelme.utils.newIcon("undo"))
        bb.setCenterButtons(True)  # center the buttons
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        
        # label_list
        self.labelList = QtWidgets.QListWidget()
        if self._fit_to_content["row"]:
            self.labelList.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        if self._fit_to_content["column"]:
            self.labelList.setVerticalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
        self._sort_labels = sort_labels
        if labels:
            self.labelList.addItems(labels)
        if self._sort_labels:
            self.labelList.sortItems()
        else:
            self.labelList.setDragDropMode(
                QtWidgets.QAbstractItemView.InternalMove
            )
        self.labelList.currentItemChanged.connect(self.labelSelected)
        self.labelList.itemDoubleClicked.connect(self.labelDoubleClicked)
        self.edit.setListWidget(self.labelList)

        self.labelListLabel = QtWidgets.QLabel()
        self.labelListLabel.setText("Select From Class List")
        # label_flags
        if flags is None:
            flags = {}
        self._flags = flags
        self.flagsLayout = QtWidgets.QVBoxLayout()
        self.resetFlags()
        self.edit.textChanged.connect(self.updateFlags)

        # confidence
        self.confidenceEdit = QtWidgets.QLineEdit()
        self.confidenceEdit.setPlaceholderText('Confidence')

        # Add a validator to accept only floats between 0 and 1
        validator = QtGui.QDoubleValidator(0, 1, 2, self.confidenceEdit)
        self.confidenceEdit.setValidator(validator)

        # add title before confidence
        self.confidenceEditLabel = QtWidgets.QLabel()
        self.confidenceEditLabel.setText('Confidence')
        


        layout = QtWidgets.QVBoxLayout()
        layout.addItem(self.flagsLayout)

        layout.addWidget(self.select_class_label)
        layout.addWidget(self.edit)
        
        # Create a vertical layout for the edit group ID label and edit
        edit_group_id_layout = QtWidgets.QVBoxLayout()
        edit_group_id_layout.addWidget(self.edit_group_id_label)
        edit_group_id_layout.addWidget(self.edit_group_id)

        # Create a vertical layout for the confidence label and edit
        confidence_layout = QtWidgets.QVBoxLayout()
        confidence_layout.addWidget(self.confidenceEditLabel)
        confidence_layout.addWidget(self.confidenceEdit)
        
        # add both vertical layouts to a horizontal layout
        horizontal_layout = QtWidgets.QHBoxLayout()
        horizontal_layout.addItem(edit_group_id_layout)
        horizontal_layout.addSpacing(10)  # add 10 pixels of space
        horizontal_layout.addItem(confidence_layout)

        # add the horizontal layout to the main layout
        layout.addItem(horizontal_layout)

        # add the label list and label list label to the main layout
        layout.addWidget(self.labelListLabel)
        layout.addWidget(self.labelList)
        layout.addWidget(bb)
        self.resize(300,200)
        self.setLayout(layout)


        # completion
        completer = QtWidgets.QCompleter()
        if not QT5 and completion != "startswith":
            logger.warn(
                "completion other than 'startswith' is only "
                "supported with Qt5. Using 'startswith'"
            )
            completion = "startswith"
        if completion == "startswith":
            completer.setCompletionMode(QtWidgets.QCompleter.InlineCompletion)
            # Default settings.
            # completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        elif completion == "contains":
            completer.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
            completer.setFilterMode(QtCore.Qt.MatchContains)
        else:
            raise ValueError("Unsupported completion: {}".format(completion))
        completer.setModel(self.labelList.model())
        self.edit.setCompleter(completer)

    def addLabelHistory(self, label):
        if self.labelList.findItems(label, QtCore.Qt.MatchExactly):
            return
        self.labelList.addItem(label)
        if self._sort_labels:
            self.labelList.sortItems()

    def labelSelected(self, item):
        self.edit.setText(item.text())

    def validate(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        if text:
            self.accept()

    def labelDoubleClicked(self, item):
        self.validate()

    def postProcess(self):
        text = self.edit.text()
        if hasattr(text, "strip"):
            text = text.strip()
        else:
            text = text.trimmed()
        self.edit.setText(text)

    def updateFlags(self, label_new):
        # keep state of shared flags
        flags_old = self.getFlags()

        flags_new = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label_new):
                for key in keys:
                    flags_new[key] = flags_old.get(key, False)
        self.setFlags(flags_new)

    def deleteFlags(self):
        for i in reversed(range(self.flagsLayout.count())):
            item = self.flagsLayout.itemAt(i).widget()
            self.flagsLayout.removeWidget(item)
            item.setParent(None)

    def resetFlags(self, label=""):
        flags = {}
        for pattern, keys in self._flags.items():
            if re.match(pattern, label):
                for key in keys:
                    flags[key] = False
        self.setFlags(flags)

    def setFlags(self, flags):
        self.deleteFlags()
        for key in flags:
            item = QtWidgets.QCheckBox(key, self)
            item.setChecked(flags[key])
            self.flagsLayout.addWidget(item)
            item.show()

    def getFlags(self):
        flags = {}
        for i in range(self.flagsLayout.count()):
            item = self.flagsLayout.itemAt(i).widget()
            flags[item.text()] = item.isChecked()
        return flags

    def getGroupId(self):
        group_id = self.edit_group_id.text()
        if group_id:
            return int(group_id)
        return None
        
    def getContent(self):
        content = self.confidenceEdit.text()
        if content:
            return content
        return None
        
    def setContent(self, content):
        if type(content) != str:
            content = str(content)
        self.confidenceEdit.setText(content)

    def popUp(self, text=None, move=True, flags=None, group_id=None, content=None, skip_flag=False):
        if self._fit_to_content["row"]:
            self.labelList.setMinimumHeight(
                self.labelList.sizeHintForRow(0) * self.labelList.count() + 2
            )
        if self._fit_to_content["column"]:
            self.labelList.setMinimumWidth(
                self.labelList.sizeHintForColumn(0) + 2
            )
        # if text is None, the previous label in self.edit is kept
        if text is None:
            text = self.edit.text()
        # if content is None, make the self.confidenceEdit empty
        if content is None:
            content=""
        self.setContent(content)
        if flags:
            self.setFlags(flags)
        else:
            self.resetFlags(text)
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        if group_id is None:
            self.edit_group_id.clear()
        else:
            self.edit_group_id.setText(str(group_id))
        items = self.labelList.findItems(text, QtCore.Qt.MatchFixedString)
        if items:
            if len(items) != 1:
                logger.warning("Label list has duplicate '{}'".format(text))
            self.labelList.setCurrentItem(items[0])
            row = self.labelList.row(items[0])
            self.edit.completer().setCurrentRow(row)
        self.edit.setFocus(QtCore.Qt.PopupFocusReason)
        if move:
            self.move(QtGui.QCursor.pos())
        if skip_flag:
            return self.edit.text(), self.getFlags(), self.getGroupId(), self.getContent()
        if self.exec_():
            return self.edit.text(), self.getFlags(), self.getGroupId(), self.getContent()
        else:
            return None, None, None, None
