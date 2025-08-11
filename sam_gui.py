import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QFileDialog, QHBoxLayout, QVBoxLayout,
    QTextEdit, QApplication, QFrame, QSizePolicy, QSpacerItem, QDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont

from ultralytics import SAM

def cvimg2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

class SAMWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM实例分割")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)
        self.model = None
        self.weight_path = None
        self.input_img = None
        self.input_path = None
        self.input_type = "图片"
        self.save_video_frames = []
        self.timer = None
        self.cap = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
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

        self.btn_weight = QPushButton("上传权重")
        self.btn_weight.setFixedWidth(90)
        self.btn_weight.clicked.connect(self.select_weight)
        toolbar.addWidget(self.btn_weight)
        self.weight_label = QLabel("未选择")
        self.weight_label.setMinimumWidth(120)
        toolbar.addWidget(self.weight_label)

        self.btn_infer = QPushButton("开始分割")
        self.btn_infer.setFixedWidth(90)
        self.btn_infer.clicked.connect(self.start_infer)
        toolbar.addWidget(self.btn_infer)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedWidth(70)
        self.btn_stop.clicked.connect(self.stop_infer)
        toolbar.addWidget(self.btn_stop)

        toolbar.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(toolbar)

        body_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
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
        result_title = QLabel("分割结果")
        result_title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(result_title)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.result_label)
        body_layout.addLayout(right_layout, 1)

        main_layout.addLayout(body_layout, 5)

        info_layout = QVBoxLayout()
        info_title = QLabel("分割信息")
        info_title.setFont(QFont("微软雅黑", 9, QFont.Bold))
        info_title.setAlignment(Qt.AlignLeft)
        info_layout.addWidget(info_title)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        info_layout.addWidget(self.info_text)
        main_layout.addLayout(info_layout, 2)

        # 使左右界面随窗口自适应
        self.setLayout(main_layout)

    def resizeEvent(self, event):
        # 窗口大小变化时自适应图片显示
        if self.input_img is not None:
            self.input_label.setPixmap(cvimg2qt(self.input_img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
        if hasattr(self, 'last_result_img') and self.last_result_img is not None:
            self.result_label.setPixmap(cvimg2qt(self.last_result_img).scaled(self.result_label.size(), Qt.KeepAspectRatio))
        super().resizeEvent(event)

    def select_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择SAM权重文件", "", "SAM Weights (*.pt)")
        if path:
            self.weight_path = path
            self.weight_label.setText(os.path.basename(path))
            try:
                self.model = SAM(self.weight_path)
                QMessageBox.information(self, "加载成功", f"权重加载成功: {os.path.basename(path)}")
            except Exception as e:
                self.model = None
                QMessageBox.critical(self, "加载失败", f"权重加载失败: {e}")
                self.weight_label.setText("未选择")

    def on_input_type_changed(self, text):
        self.input_type = text
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.input_img = None
        self.input_path = None
        self.save_video_frames = []

    def upload_input(self):
        if self.input_type == "图片":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
            if path:
                self.input_path = path
                img = cv2.imread(path)
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
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
        elif self.input_type == "摄像头":
            cap = cv2.VideoCapture(0)
            ret, img = cap.read()
            if ret:
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.input_path = 0
            cap.release()

    def start_infer(self):
        self.result_label.clear()
        self.info_text.clear()
        if self.model is None:
            QMessageBox.warning(self, "未加载权重", "请先上传并加载SAM权重文件！")
            return
        if self.input_type == "图片":
            if self.input_img is not None:
                self.run_sam(self.input_img)
        elif self.input_type == "视频":
            if self.input_path is not None:
                self.cap = cv2.VideoCapture(self.input_path)
                self.timer = QTimer(self)
                self.timer.timeout.connect(self.process_video_frame)
                self.timer.start(30)
        elif self.input_type == "摄像头":
            self.cap = cv2.VideoCapture(0)
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.process_video_frame)
            self.timer.start(30)

    def stop_infer(self):
        if self.timer:
            self.timer.stop()
        if self.cap:
            self.cap.release()

    def process_video_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return
        ret, img = self.cap.read()
        if not ret:
            self.stop_infer()
            return
        self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
        self.run_sam(img)

    def run_sam(self, img):
        try:
            results = self.model(img, device='cuda')
            masks = results[0].masks.data.cpu().numpy() if hasattr(results[0], "masks") and results[0].masks is not None else []
            classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []
            confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []
            boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []

            # 可视化分割结果
            seg_img = img.copy()
            info_lines = []
            contour_color = (0, 255, 0)  # 统一轮廓颜色为绿色
            for idx, mask in enumerate(masks):
                color = np.random.randint(0, 255, (3,), dtype=np.uint8)
                # 填充mask区域
                seg_img[mask > 0.5] = seg_img[mask > 0.5] * 0.5 + color * 0.5
                # 轮廓着色（统一颜色）
                mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(seg_img, contours, -1, contour_color, 2)
                if idx < len(boxes):
                    box = boxes[idx]
                    cls_id = int(classes[idx]) if idx < len(classes) else -1
                    conf = confs[idx] if idx < len(confs) else 0
                    info_lines.append(f"Obj{idx}: 坐标{box.astype(int).tolist()}, 类别{cls_id}, 置信度:{conf:.2f}")
            self.last_result_img = seg_img.copy()
            self.result_label.setPixmap(cvimg2qt(seg_img).scaled(self.result_label.size(), Qt.KeepAspectRatio))
            self.info_text.setPlainText("\n".join(info_lines) if info_lines else "无分割结果")
        except Exception as e:
            QMessageBox.critical(self, "分割失败", f"分割失败: {e}")
