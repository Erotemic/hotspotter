# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\joncrall\code\hotspotter\hsgui/_frontend\OpenDatabaseDialog.ui'
#
# Created: Thu Feb 06 19:54:16 2014
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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(387, 211)
        self.verticalLayout_2 = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label = QtGui.QLabel(Dialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.new_db_but = QtGui.QPushButton(Dialog)
        self.new_db_but.setMinimumSize(QtCore.QSize(0, 100))
        self.new_db_but.setObjectName(_fromUtf8("new_db_but"))
        self.horizontalLayout.addWidget(self.new_db_but)
        self.open_db_but = QtGui.QPushButton(Dialog)
        self.open_db_but.setMinimumSize(QtCore.QSize(0, 100))
        self.open_db_but.setObjectName(_fromUtf8("open_db_but"))
        self.horizontalLayout.addWidget(self.open_db_but)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.label.setText(_translate("Dialog", "HotSpotter - Animal Instance Recognition (dev version)", None))
        self.new_db_but.setText(_translate("Dialog", "New Database", None))
        self.open_db_but.setText(_translate("Dialog", "Open Database", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

