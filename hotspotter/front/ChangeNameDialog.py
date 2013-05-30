# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\jon.crall\code\hotspotter\hs_setup\../hotspotter/front\ChangeNameDialog.ui'
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

class Ui_changeNameDialog(object):
    def setupUi(self, changeNameDialog):
        changeNameDialog.setObjectName(_fromUtf8("changeNameDialog"))
        changeNameDialog.resize(441, 109)
        self.verticalLayout = QtGui.QVBoxLayout(changeNameDialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.formLayout = QtGui.QFormLayout()
        self.formLayout.setObjectName(_fromUtf8("formLayout"))
        self.label = QtGui.QLabel(changeNameDialog)
        self.label.setObjectName(_fromUtf8("label"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.LabelRole, self.label)
        self.label_2 = QtGui.QLabel(changeNameDialog)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.LabelRole, self.label_2)
        self.newNameEdit = QtGui.QLineEdit(changeNameDialog)
        self.newNameEdit.setObjectName(_fromUtf8("newNameEdit"))
        self.formLayout.setWidget(1, QtGui.QFormLayout.FieldRole, self.newNameEdit)
        self.oldNameEdit = QtGui.QLineEdit(changeNameDialog)
        self.oldNameEdit.setObjectName(_fromUtf8("oldNameEdit"))
        self.formLayout.setWidget(0, QtGui.QFormLayout.FieldRole, self.oldNameEdit)
        self.verticalLayout.addLayout(self.formLayout)
        self.buttonBox = QtGui.QDialogButtonBox(changeNameDialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtGui.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName(_fromUtf8("buttonBox"))
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(changeNameDialog)
        QtCore.QMetaObject.connectSlotsByName(changeNameDialog)

    def retranslateUi(self, changeNameDialog):
        changeNameDialog.setWindowTitle(_translate("changeNameDialog", "Change Name Dialog", None))
        self.label.setText(_translate("changeNameDialog", "Change all names matching:", None))
        self.label_2.setText(_translate("changeNameDialog", "To the new name:", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    changeNameDialog = QtGui.QDialog()
    ui = Ui_changeNameDialog()
    ui.setupUi(changeNameDialog)
    changeNameDialog.show()
    sys.exit(app.exec_())

