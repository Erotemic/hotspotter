from PyQt4 import QtCore, QtGui
from MainSkel import Ui_mainSkel
import multiprocessing

class HotSpotterMainWindow(QtGui.QMainWindow):
    def __init__(self, hs=None):
        super(HotSpotterMainWindow, self).__init__()
        self.hs = None
        self.ui=Ui_mainSkel()
        self.ui.setupUi(self)
        self.show()
        if hs is None:
            self.connect_api(hs)
    def connect_api(self, hs):
        print('[win] connecting api')
        self.hs = hs

    def update_image_table(self):
        pass

if __name__ == '__main__':
    import sys
    multiprocessing.freeze_support()
    def test():
        app = QtGui.QApplication(sys.argv)
        main_win = HotSpotterMainWindow()
        app.setActiveWindow(main_win)
        sys.exit(app.exec_())
    test()

