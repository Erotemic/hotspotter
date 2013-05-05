# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/joncrall/code/hotspotter/hotspotter/front/ResultDialog.ui'
#
# Created: Sun May  5 19:14:16 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_ResultDialog(object):
    def setupUi(self, ResultDialog):
        ResultDialog.setObjectName(_fromUtf8("ResultDialog"))
        ResultDialog.resize(609, 382)
        self.verticalLayout_2 = QtGui.QVBoxLayout(ResultDialog)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.buttonBox = QtGui.QDialogButtonBox(ResultDialog)
        self.buttonBox.setAutoFillBackground(False)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.No|QtGui.QDialogButtonBox.Yes)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout_2.addWidget(self.buttonBox)

        self.retranslateUi(ResultDialog)
        QtCore.QMetaObject.connectSlotsByName(ResultDialog)

    def retranslateUi(self, ResultDialog):
        ResultDialog.setWindowTitle(QtGui.QApplication.translate("ResultDialog", "ResultDialog", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    ResultDialog = QtGui.QDialog()
    ui = Ui_ResultDialog()
    ui.setupUi(ResultDialog)
    ResultDialog.show()
    sys.exit(app.exec_())

