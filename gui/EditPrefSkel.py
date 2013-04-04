# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/joncrall/code/hotspotter/gui/EditPrefSkel.ui'
#
# Created: Thu Apr  4 14:15:34 2013
#      by: PyQt4 UI code generator 4.9.1
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_editPrefSkel(object):
    def setupUi(self, editPrefSkel):
        editPrefSkel.setObjectName(_fromUtf8("editPrefSkel"))
        editPrefSkel.resize(668, 530)
        self.verticalLayout = QtGui.QVBoxLayout(editPrefSkel)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.prefTreeView = QtGui.QTreeView(editPrefSkel)
        self.prefTreeView.setObjectName(_fromUtf8("prefTreeView"))
        self.verticalLayout.addWidget(self.prefTreeView)

        self.retranslateUi(editPrefSkel)
        QtCore.QMetaObject.connectSlotsByName(editPrefSkel)

    def retranslateUi(self, editPrefSkel):
        editPrefSkel.setWindowTitle(QtGui.QApplication.translate("editPrefSkel", "Edit Preferences", None, QtGui.QApplication.UnicodeUTF8))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    editPrefSkel = QtGui.QWidget()
    ui = Ui_editPrefSkel()
    ui.setupUi(editPrefSkel)
    editPrefSkel.show()
    sys.exit(app.exec_())

