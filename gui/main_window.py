import time
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem,
    QRadioButton, QButtonGroup, QCheckBox, QTextEdit, QSplitter,
    QFrame, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from gui.camera_thread import CameraThread
from gui.collect_dialog import CollectDialog
from gui.train_dialog import TrainDialog


class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._mode_ref = ['recognize']
        self._collect_name_ref = ['']
        self._camera_index = 0
        self._camera_thread = None
        self._fps_count = 0
        self._fps_timer = time.time()
        self._build_ui()
        self._start_camera()

    def _build_ui(self):
        self.setWindowTitle('Face Recognition System')
        self.setMinimumSize(960, 640)
        central = QWidget()
        self.setCentralWidget(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        sidebar = self._build_sidebar()
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 0)
        right = self._build_right_panel()
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 740])

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setFrameStyle(QFrame.Shape.StyledPanel)
        sidebar.setMaximumWidth(240)
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(8)
        title = QLabel('<b>Face Recognition</b>')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('font-size: 14px; padding: 10px;')
        layout.addWidget(title)
        layout.addWidget(QLabel('Mode:'))
        self._mode_group = QButtonGroup(self)
        self._rb_recognize = QRadioButton('Recognize')
        self._rb_recognize.setChecked(True)
        self._rb_collect = QRadioButton('Collect')
        self._rb_train = QRadioButton('Train')
        for rb in (self._rb_recognize, self._rb_collect, self._rb_train):
            self._mode_group.addButton(rb)
            layout.addWidget(rb)
        self._mode_group.buttonToggled.connect(self._on_mode_changed)
        layout.addSpacing(10)
        layout.addWidget(QLabel('Known Persons:'))
        self._person_list = QListWidget()
        layout.addWidget(self._person_list)
        self._btn_add = QPushButton('+ Add Person')
        self._btn_add.clicked.connect(self._on_add_person)
        layout.addWidget(self._btn_add)
        self._btn_delete = QPushButton('Delete Person')
        self._btn_delete.clicked.connect(self._on_delete_person)
        layout.addWidget(self._btn_delete)
        self._btn_train = QPushButton('Train Model')
        self._btn_train.clicked.connect(self._on_train)
        layout.addWidget(self._btn_train)
        layout.addSpacing(10)
        self._cb_phone = QCheckBox('Use Phone Camera')
        self._cb_phone.toggled.connect(self._toggle_camera)
        layout.addWidget(self._cb_phone)
        layout.addStretch()
        return sidebar

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        self._video_label = QLabel()
        self._video_label.setMinimumSize(640, 480)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet('background-color: #1a1a1a; border: 1px solid #333;')
        self._video_label.setText('Starting camera...')
        layout.addWidget(self._video_label, 1)
        self._fps_label = QLabel('FPS: 0')
        self._fps_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._fps_label.setStyleSheet('font-size: 11px; color: #888; padding: 2px 8px;')
        layout.addWidget(self._fps_label)
        self._log_widget = QTextEdit()
        self._log_widget.setReadOnly(True)
        self._log_widget.setMaximumHeight(140)
        self._log_widget.setStyleSheet('font-family: monospace; font-size: 11px;')
        layout.addWidget(self._log_widget)
        return panel

    def _start_camera(self):
        if self._camera_thread and self._camera_thread.isRunning():
            self._camera_thread.stop()
        self._camera_thread = CameraThread(
            self.engine, self._mode_ref, self._collect_name_ref,
            camera_index=self._camera_index,
        )
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.result_ready.connect(self._on_results)
        self._camera_thread.collect_ready.connect(self._on_collect_result)
        self._camera_thread.error.connect(self._on_camera_error)
        self._camera_thread.start()
        self._log('Camera started.')

    def _on_frame(self, pixmap):
        scaled = pixmap.scaled(
            self._video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._video_label.setPixmap(scaled)
        self._fps_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps_label.setText(f'FPS: {self._fps_count / elapsed:.1f}')
            self._fps_count = 0
            self._fps_timer = time.time()

    def _on_results(self, results):
        for r in results:
            if r.get('is_unknown'):
                self._log(f'Unknown face detected ({r["confidence"]:.0%})')
            else:
                self._log(f'{r["name"]} recognized ({r["confidence"]:.0%})')

    def _on_collect_result(self, result):
        pass

    def _on_camera_error(self, msg):
        self._log(f'ERROR: {msg}')
        self._video_label.setText(f'Camera error: {msg}')

    def _toggle_camera(self, checked):
        self._camera_index = 2 if checked else 0
        label = 'phone' if checked else 'laptop'
        self._log(f'Switched to {label} camera (index {self._camera_index})')
        self._start_camera()

    def _on_mode_changed(self):
        btn = self._mode_group.checkedButton()
        if btn == self._rb_recognize:
            self._mode_ref[0] = 'recognize'
            self._log('Mode: Recognize')
        elif btn == self._rb_collect:
            self._mode_ref[0] = 'collect'
            self._log('Mode: Collect')
        elif btn == self._rb_train:
            self._mode_ref[0] = 'train'
            self._log('Mode: Train (idle — click Train Model)')
        self._refresh_person_list()

    def _refresh_person_list(self):
        self._person_list.clear()
        info = self.engine.get_dataset_info()
        for p in info['persons']:
            item = QListWidgetItem(f'{p["name"]}  ({p["count"]})')
            item.setData(Qt.ItemDataRole.UserRole, p['name'])
            self._person_list.addItem(item)
        if not info['persons']:
            self._person_list.addItem('(no data)')

    def _on_add_person(self):
        dlg = CollectDialog(self.engine, self._camera_thread,
                            self._mode_ref, self._collect_name_ref)
        dlg.exec()
        if dlg.collected_name:
            self._log(f'Collection complete: {dlg.collected_name}')
        self._mode_ref[0] = 'recognize'
        self._rb_recognize.setChecked(True)
        self._refresh_person_list()

    def _on_delete_person(self):
        item = self._person_list.currentItem()
        if not item:
            self._log('No person selected to delete.')
            return
        name = item.data(Qt.ItemDataRole.UserRole)
        if not name:
            return
        confirm = QMessageBox.question(
            self, 'Confirm Delete',
            f'Delete all data for "{name}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.engine.delete_person(name)
            self._log(f'Deleted: {name}')
            self._refresh_person_list()

    def _on_train(self):
        self._mode_ref[0] = 'train'
        dlg = TrainDialog(self.engine)
        dlg.exec()
        if dlg.result_data and dlg.result_data.get('success'):
            self._log(f'Training complete — acc: {dlg.result_data["train_acc"]:.1%} / '
                      f'val: {dlg.result_data["val_acc"]:.1%}')
            self._refresh_person_list()
        self._mode_ref[0] = 'recognize'
        self._rb_recognize.setChecked(True)

    def _log(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        self._log_widget.append(f'[{ts}] {msg}')
        sb = self._log_widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event):
        if self._camera_thread and self._camera_thread.isRunning():
            self._camera_thread.stop()
        event.accept()
