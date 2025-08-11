import sys
import os
import cv2
import numpy as np
import time
import traceback
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QFileDialog,
    QHBoxLayout, QVBoxLayout, QTextEdit, QStatusBar, QSizePolicy, QSpacerItem, QFrame
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont

# Ultralytics YOLO OBB
from ultralytics import YOLO

def cvimg2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

class OBBWindow(QMainWindow):
    # 新增信号
    result_img_signal = pyqtSignal(object)
    info_signal = pyqtSignal(str)
    input_img_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OBB检测窗口")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.model = None
        self.weight_path = None
        self.input_type = "图片"
        self.input_path = None
        self.input_img = None
        self.running = False
        self.save_video_frames = []
        self.init_ui()
        self.init_statusbar()
        self.init_timer()

        self.result_img_signal.connect(self.show_result_img)
        self.info_signal.connect(self.show_info)
        self.input_img_signal.connect(self.show_input_img)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(QLabel("输入方式："))
        self.input_combo = QComboBox()
        self.input_combo.addItems(["图片", "视频", "摄像头"])
        self.input_combo.setFixedWidth(90)
        self.input_combo.currentTextChanged.connect(self.on_input_type_changed)
        toolbar.addWidget(self.input_combo)

        self.btn_upload = QPushButton("上传/打开")
        self.btn_upload.setFixedWidth(90)
        self.btn_upload.clicked.connect(self.upload_input)
        toolbar.addWidget(self.btn_upload)

        self.btn_reset = QPushButton("复位")
        self.btn_reset.setFixedWidth(70)
        self.btn_reset.clicked.connect(self.reset_all)
        toolbar.addWidget(self.btn_reset)

        toolbar.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.btn_weight = QPushButton("上传权重")
        self.btn_weight.setFixedWidth(90)
        self.btn_weight.clicked.connect(self.select_weight)
        toolbar.addWidget(self.btn_weight)
        self.weight_label = QLabel("未选择")
        self.weight_label.setMinimumWidth(70)
        toolbar.addWidget(self.weight_label)

        self.btn_infer = QPushButton("开始检测")
        self.btn_infer.setFixedWidth(90)
        self.btn_infer.clicked.connect(self.start_infer)
        toolbar.addWidget(self.btn_infer)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedWidth(70)
        self.btn_stop.clicked.connect(self.stop_infer)
        toolbar.addWidget(self.btn_stop)

        main_layout.addLayout(toolbar)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)
        body_layout.setContentsMargins(0, 0, 0, 0)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(2)
        input_title = QLabel("原始输入区")
        input_title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        input_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(input_title)
        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFrameShape(QFrame.Box)
        self.input_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.input_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.input_label)
        body_layout.addLayout(left_layout, 1)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(2)
        self.right_title = QLabel("OBB检测结果")
        self.right_title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        self.right_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.right_title)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.result_label)
        body_layout.addLayout(right_layout, 1)

        main_layout.addLayout(body_layout, 5)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.info_text, 2)

    def init_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_time = QLabel("检测时间: 0 ms")
        self.status.addWidget(self.status_time)

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)

    def update_status(self):
        pass

    def log(self, msg):
        now = time.strftime("[%H:%M:%S] ")
        self.info_text.append(now + msg)

    def show_error(self, msg):
        self.info_text.append(f"[错误] {msg}")

    def upload_input(self):
        if self.input_type == "图片":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
            if path:
                self.input_path = path
                img = cv2.imread(path)
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.log(f"已选择图片: {path}")
        elif self.input_type == "视频":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
            if path:
                self.input_path = path
                cap = cv2.VideoCapture(path)
                ret, img = cap.read()
                if ret:
                    self.input_img = img
                    self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                cap.release()
                self.log(f"已选择视频: {path}")
        elif self.input_type == "摄像头":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.show_error("摄像头无法打开，请检查设备。")
                return
            ret, img = cap.read()
            if ret:
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.input_path = 0
                self.log("已打开摄像头")
            cap.release()

    def on_input_type_changed(self, text):
        self.input_type = text
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.input_path = None
        self.save_video_frames = []

    def select_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择OBB权重文件", "", "PyTorch Weights (*.pt *.pth)")
        if path:
            self.weight_path = path
            self.weight_label.setText(os.path.basename(path))
            try:
                self.model = YOLO(path, task="obb")
                self.log("OBB模型加载成功")
            except Exception as e:
                self.show_error(f"模型加载失败: {e}")
                self.model = None

    def start_infer(self):
        if self.model is None:
            self.show_error("请先上传OBB权重")
            return
        if self.input_path is None:
            self.show_error("请先上传输入数据")
            return
        self.running = True
        self.info_text.clear()
        self.result_label.clear()
        self.save_video_frames = []
        import threading
        threading.Thread(target=self.infer_thread, daemon=True).start()

    def stop_infer(self):
        self.running = False
        self.log("停止检测")

    def infer_thread(self):
        try:
            t0 = time.time()
            if self.input_type == "图片":
                img = cv2.imread(self.input_path)
                self.input_img = img
                self.input_img_signal.emit(img)
                result_img, info = self.run_obb(img)
                self.result_img_signal.emit(result_img)
                self.info_signal.emit(info)
            elif self.input_type == "视频":
                cap = cv2.VideoCapture(self.input_path)
                self.save_video_frames = []
                while self.running and cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break
                    self.input_img_signal.emit(img)
                    result_img, info = self.run_obb(img)
                    self.result_img_signal.emit(result_img)
                    self.info_signal.emit(info)
                    self.save_video_frames.append(result_img.copy())
                    cv2.waitKey(1)
                cap.release()
            elif self.input_type == "摄像头":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    self.show_error("摄像头无法打开，请检查设备。")
                    return
                self.save_video_frames = []
                while self.running and cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break
                    self.input_img_signal.emit(img)
                    result_img, info = self.run_obb(img)
                    self.result_img_signal.emit(result_img)
                    self.info_signal.emit(info)
                    self.save_video_frames.append(result_img.copy())
                    cv2.waitKey(1)
                cap.release()
            t1 = time.time()
            infer_time = int((t1 - t0) * 1000)
            self.status_time.setText(f"检测时间: {infer_time} ms")
            self.log(f"检测完成，耗时{infer_time} ms")
        except Exception as e:
            self.show_error(f"检测异常: {e}\n{traceback.format_exc()}")

    def run_obb(self, img):
        # OBB推理，强制用CUDA
        results = self.model(img, device='cuda')
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        info = self.get_obb_info(results)
        return result_img, info

    def get_obb_info(self, results):
        lines = []
        for i, r in enumerate(results):
            if hasattr(r, 'obb') and r.obb is not None:
                for j, (xyxy, conf, cls) in enumerate(zip(r.obb.xyxy.cpu().numpy(), r.obb.conf.cpu().numpy(), r.obb.cls.cpu().numpy())):
                    lines.append(f"Obj{j}: 坐标{xyxy}, 类别{int(cls)}, 置信度:{conf:.2f}")
        return '\n'.join(lines) if lines else "无OBB检测结果"

    def show_input_img(self, img):
        # 保证输入流显示BGR->RGB一致
        if img is not None:
            img_bgr = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.input_label.setPixmap(cvimg2qt(img_bgr).scaled(self.input_label.size(), Qt.KeepAspectRatio))

    def show_result_img(self, img):
        # 保证输出流显示BGR->RGB一致
        if img is not None:
            img_bgr = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.result_label.setPixmap(cvimg2qt(img_bgr).scaled(self.result_label.size(), Qt.KeepAspectRatio))

    def show_info(self, info):
        self.info_text.setPlainText(info)

    def reset_all(self):
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.input_path = None
        self.save_video_frames = []
        self.model = None
        self.weight_path = None
        self.input_type = "图片"
        self.input_combo.setCurrentIndex(0)
        self.weight_label.setText("未选择")
        self.log("已复位所有界面和变量")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OBBWindow()
    win.show()
    sys.exit(app.exec_())
