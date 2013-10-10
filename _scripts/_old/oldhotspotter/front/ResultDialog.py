# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\jon.crall\code\hotspotter\hs_setup\../hotspotter/front\ResultDialog.ui'
#
# Created: Thu May 30 15:48:17 2013
#      by: PyQt4 UI code generator 4.10.1
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
        ResultDialog.setWindowTitle(_translate("ResultDialog", "ResultDialog", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    ResultDialog = QtGui.QDialog()
    ui = Ui_ResultDialog()
    ui.setupUi(ResultDialog)
    ResultDialog.show()
    sys.exit(app.exec_())

