# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Code\HotSpotter-python\widgets\AlgoWidget.ui'
#
# Created: Fri Mar 29 22:42:29 2013
#      by: PyQt4 UI code generator 4.9.6
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_algoWidget(object):
    def setupUi(self, algoWidget):
        algoWidget.setObjectName(_fromUtf8("algoWidget"))
        algoWidget.resize(459, 321)
        self.verticalLayout = QtGui.QVBoxLayout(algoWidget)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.treeWidget = QtGui.QTreeWidget(algoWidget)
        self.treeWidget.setObjectName(_fromUtf8("treeWidget"))
        self.treeWidget.headerItem().setText(0, _fromUtf8("1"))
        self.verticalLayout.addWidget(self.treeWidget)

        self.retranslateUi(algoWidget)
        QtCore.QMetaObject.connectSlotsByName(algoWidget)

    def retranslateUi(self, algoWidget):
        algoWidget.setWindowTitle(_translate("algoWidget", "Algorithm Settings", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    algoWidget = QtGui.QWidget()
    ui = Ui_algoWidget()
    ui.setupUi(algoWidget)
    algoWidget.show()
    sys.exit(app.exec_())

