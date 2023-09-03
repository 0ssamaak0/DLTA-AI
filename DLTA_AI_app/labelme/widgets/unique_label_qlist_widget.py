# -*- encoding: utf-8 -*-

from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets

from .escapable_qlist_widget import EscapableQListWidget


class UniqueLabelQListWidget(EscapableQListWidget):
    def mousePressEvent(self, event):
        super(UniqueLabelQListWidget, self).mousePressEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.clearSelection()

    def findItemsByLabel(self, label):
        items = []
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == label:
                items.append(item)
        return items

    def createItemFromLabel(self, label):
        item = QtWidgets.QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, label)
        return item

    def setItemLabel(self, item, label, color=None):
        qlabel = QtWidgets.QLabel()
        if color is None:
            qlabel.setText("{}".format(label))
        else:
            qlabel.setText(
                '{} <font color="#{:02x}{:02x}{:02x}">●</font>'.format(
                    label, *color
                )
            )
        qlabel.setAlignment(Qt.AlignmentFlag.AlignBottom)

        item.setSizeHint(qlabel.sizeHint())

        self.setItemWidget(item, qlabel)
