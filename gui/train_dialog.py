from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QProgressBar,
    QHBoxLayout, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class TrainWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(dict)

    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def run(self):
        result = self.engine.train(progress_callback=self._cb)
        self.finished.emit(result)

    def _cb(self, step, pct):
        self.progress.emit(step, pct)


class TrainDialog(QDialog):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.result_data = None
        self._build_ui()
        self._show_summary()

    def _build_ui(self):
        self.setWindowTitle('Train Model')
        self.setMinimumWidth(450)
        layout = QVBoxLayout(self)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._status = QLabel()
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)
        self._results = QLabel()
        self._results.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._results.setVisible(False)
        layout.addWidget(self._results)
        btn_row = QHBoxLayout()
        self._btn_train = QPushButton('Start Training')
        self._btn_train.clicked.connect(self._on_start)
        btn_row.addWidget(self._btn_train)
        self._btn_close = QPushButton('Close')
        self._btn_close.clicked.connect(self.accept)
        self._btn_close.setEnabled(False)
        btn_row.addWidget(self._btn_close)
        layout.addLayout(btn_row)

    def _show_summary(self):
        info = self.engine.get_dataset_info()
        lines = ['<b>Dataset Summary:</b>']
        for p in info['persons']:
            lines.append(f'{p["name"]}: {p["count"]} images')
        lines.append(f'<b>Total: {info["total_images"]} images</b>')
        if len(info['persons']) < 2:
            lines.append('<br><span style="color:red;">Need at least 2 persons to train.</span>'
                         if False else '')
        self._summary.setText('<br>'.join(lines))

    def _on_start(self):
        self._btn_train.setEnabled(False)
        self._progress.setVisible(True)
        self._status.setText('Starting...')
        self._worker = TrainWorker(self.engine)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, step, pct):
        self._progress.setValue(pct)
        self._status.setText(f'{step} ({pct}%)')

    def _on_finished(self, result):
        self.result_data = result
        self._progress.setValue(100)
        if result['success']:
            txt = (
                f'<span style="color:green; font-size:14px;">Training complete!</span><br>'
                f'Train accuracy: {result["train_acc"]:.2%}<br>'
                f'Validation accuracy: {result["val_acc"]:.2%}<br>'
                f'Persons: {result["n_persons"]}  |  Embeddings: {result["n_images"]}'
            )
            self._results.setText(txt)
        else:
            txt = f'<span style="color:red;">Error: {result["error"]}</span>'
            self._results.setText(txt)
        self._results.setVisible(True)
        self._status.setText('Done.')
        self._btn_close.setEnabled(True)
