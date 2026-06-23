#!/usr/bin/env python3
import os
import sys

os.environ.pop('QT_PLUGIN_PATH', None)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QLibraryInfo

pyqt_plugins = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
os.environ['QT_PLUGIN_PATH'] = pyqt_plugins

from engine import FaceRecognitionEngine
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Face Recognition System')
    engine = FaceRecognitionEngine()
    engine.load_model()
    win = MainWindow(engine)
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
