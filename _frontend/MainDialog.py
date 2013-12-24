# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\joncrall\code\hotspotter\_frontend\MainDialog.ui'
#
# Created: Tue Dec 24 02:26:11 2013
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
        Dialog.resize(454, 443)
        self.verticalLayout_2 = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.pushButton_2 = QtGui.QPushButton(Dialog)
        self.pushButton_2.setMinimumSize(QtCore.QSize(0, 62))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.verticalLayout_3.addWidget(self.pushButton_2)
        self.pushButton_5 = QtGui.QPushButton(Dialog)
        self.pushButton_5.setMinimumSize(QtCore.QSize(0, 62))
        self.pushButton_5.setObjectName(_fromUtf8("pushButton_5"))
        self.verticalLayout_3.addWidget(self.pushButton_5)
        self.pushButton_3 = QtGui.QPushButton(Dialog)
        self.pushButton_3.setMinimumSize(QtCore.QSize(0, 62))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.verticalLayout_3.addWidget(self.pushButton_3)
        self.pushButton_4 = QtGui.QPushButton(Dialog)
        self.pushButton_4.setMinimumSize(QtCore.QSize(0, 62))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.verticalLayout_3.addWidget(self.pushButton_4)
        self.pushButton = QtGui.QPushButton(Dialog)
        self.pushButton.setMinimumSize(QtCore.QSize(0, 62))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.verticalLayout_3.addWidget(self.pushButton)
        self.pushButton_8 = QtGui.QPushButton(Dialog)
        self.pushButton_8.setObjectName(_fromUtf8("pushButton_8"))
        self.verticalLayout_3.addWidget(self.pushButton_8)
        self.pushButton_6 = QtGui.QPushButton(Dialog)
        self.pushButton_6.setObjectName(_fromUtf8("pushButton_6"))
        self.verticalLayout_3.addWidget(self.pushButton_6)
        self.pushButton_7 = QtGui.QPushButton(Dialog)
        self.pushButton_7.setObjectName(_fromUtf8("pushButton_7"))
        self.verticalLayout_3.addWidget(self.pushButton_7)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_2.addLayout(self.verticalLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))
        self.pushButton_2.setText(_translate("Dialog", "Open Database\n"
" Currently open:\n"
"%{db_path}s", None))
        self.pushButton_5.setText(_translate("Dialog", "Peruse Database\n"
" Num Identified: %d{nNames} ; Num Unchecked Images: %d ;  Num Unrefined Rois: %d", None))
        self.pushButton_3.setText(_translate("Dialog", "1. Add Images\n"
"Current number of tracked images: \n"
" Internally: %{nImg_internal}d ; Externally: %{nImg_external}d", None))
        self.pushButton_4.setText(_translate("Dialog", "2. Mark Regions of Interest\n"
"Current number of tracked chips: %d{nChips}d", None))
        self.pushButton.setText(_translate("Dialog", "3. Run Queries\n"
"Current number of unidentified animals: %d{nUnidentified}", None))
        self.pushButton_8.setText(_translate("Dialog", "Preferences", None))
        self.pushButton_6.setText(_translate("Dialog", "Help", None))
        self.pushButton_7.setText(_translate("Dialog", "Quit", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Dialog = QtGui.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

