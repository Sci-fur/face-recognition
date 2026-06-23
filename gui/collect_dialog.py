from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QProgressBar,
)
from PyQt6.QtCore import Qt


class CollectDialog(QDialog):
    def __init__(self, engine, camera_thread, mode_ref, collect_name_ref):
        super().__init__()
        self.engine = engine
        self._camera_thread = camera_thread
        self._mode_ref = mode_ref
        self._collect_name_ref = collect_name_ref
        self.collected_name = None
        self._collect_name_ref[0] = ''
        self._build_ui()
        self._camera_thread.collect_ready.connect(self._on_collect_result)
        self._mode_ref[0] = 'collect'

    def _build_ui(self):
        self.setWindowTitle('Collect Face Data')
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Enter the person\'s name:'))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText('e.g. Alice')
        layout.addWidget(self._name_input)
        btn_row = QHBoxLayout()
        self._btn_start = QPushButton('Start Collection')
        self._btn_start.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_start)
        self._btn_cancel = QPushButton('Cancel')
        self._btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btn_cancel)
        layout.addLayout(btn_row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 150)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._stage_label = QLabel()
        self._stage_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stage_label.setVisible(False)
        layout.addWidget(self._stage_label)
        self._status_label = QLabel('Enter a name and click Start.')
        layout.addWidget(self._status_label)

    def _on_start(self):
        name = self._name_input.text().strip()
        if not name:
            self._status_label.setText('Please enter a name.')
            return
        self.collected_name = name
        self._collect_name_ref[0] = name
        self._btn_start.setEnabled(False)
        self._name_input.setEnabled(False)
        self._progress.setVisible(True)
        self._stage_label.setVisible(True)
        self._status_label.setText(f'Collecting for "{name}"...')
        self._mode_ref[0] = 'collect'

    def _on_collect_result(self, result):
        count = result.get('count', 0)
        self._progress.setValue(count)
        self._stage_label.setText(result.get('stage_hint', ''))
        if result.get('complete'):
            self._status_label.setText(f'Done! Collected {count} images.')
            self._finish()

    def _finish(self):
        self._camera_thread.collect_ready.disconnect(self._on_collect_result)
        self.accept()

    def reject(self):
        self._camera_thread.collect_ready.disconnect(self._on_collect_result)
        self.collected_name = None
        super().reject()
