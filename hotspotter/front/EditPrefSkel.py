# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\jon.crall\code\hotspotter\hs_setup\../hotspotter/front\EditPrefSkel.ui'
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

class Ui_editPrefSkel(object):
    def setupUi(self, editPrefSkel):
        editPrefSkel.setObjectName(_fromUtf8("editPrefSkel"))
        editPrefSkel.resize(668, 530)
        self.verticalLayout = QtGui.QVBoxLayout(editPrefSkel)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.prefTreeView = QtGui.QTreeView(editPrefSkel)
        self.prefTreeView.setObjectName(_fromUtf8("prefTreeView"))
        self.verticalLayout.addWidget(self.prefTreeView)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.redrawBUT = QtGui.QPushButton(editPrefSkel)
        self.redrawBUT.setObjectName(_fromUtf8("redrawBUT"))
        self.horizontalLayout.addWidget(self.redrawBUT)
        self.unloadFeaturesAndModelsBUT = QtGui.QPushButton(editPrefSkel)
        self.unloadFeaturesAndModelsBUT.setObjectName(_fromUtf8("unloadFeaturesAndModelsBUT"))
        self.horizontalLayout.addWidget(self.unloadFeaturesAndModelsBUT)
        self.defaultPrefsBUT = QtGui.QPushButton(editPrefSkel)
        self.defaultPrefsBUT.setObjectName(_fromUtf8("defaultPrefsBUT"))
        self.horizontalLayout.addWidget(self.defaultPrefsBUT)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(editPrefSkel)
        QtCore.QMetaObject.connectSlotsByName(editPrefSkel)

    def retranslateUi(self, editPrefSkel):
        editPrefSkel.setWindowTitle(_translate("editPrefSkel", "Edit Preferences", None))
        self.redrawBUT.setText(_translate("editPrefSkel", "Redraw", None))
        self.unloadFeaturesAndModelsBUT.setText(_translate("editPrefSkel", "Unload Features and Models", None))
        self.defaultPrefsBUT.setText(_translate("editPrefSkel", "Defaults", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    editPrefSkel = QtGui.QWidget()
    ui = Ui_editPrefSkel()
    ui.setupUi(editPrefSkel)
    editPrefSkel.show()
    sys.exit(app.exec_())

