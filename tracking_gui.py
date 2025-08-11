import os
import cv2
import time
import threading
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QFileDialog, QHBoxLayout, QVBoxLayout, QTextEdit, QSizePolicy, QFrame, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def cvimg2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

class TrackingWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("目标追踪")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)  # 允许最大化/最小化/独立窗口
        self.tracker_type = "多目标追踪"
        self.model = None
        self.input_path = None
        self.input_type = None
        self.running = False
        self.save_frames = []
        self.track_history = []  # [(frame_idx, [track_info,...])]
        self.init_ui()

    def closeEvent(self, event):
        self.running = False  # 关闭窗口时自动停止追踪线程
        event.accept()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("追踪类型："))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["多目标追踪", "单目标追踪"])
        self.tracker_combo.currentTextChanged.connect(self.on_tracker_type_changed)
        toolbar.addWidget(self.tracker_combo)
        self.btn_upload = QPushButton("上传视频")
        self.btn_upload.clicked.connect(self.upload_video)
        toolbar.addWidget(self.btn_upload)
        self.btn_camera = QPushButton("摄像头")
        self.btn_camera.clicked.connect(self.open_camera)
        toolbar.addWidget(self.btn_camera)
        self.btn_start = QPushButton("开始追踪")
        self.btn_start.clicked.connect(self.start_tracking)
        toolbar.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_tracking)
        toolbar.addWidget(self.btn_stop)
        self.btn_save = QPushButton("保存追踪结果")
        self.btn_save.clicked.connect(self.save_results)
        toolbar.addWidget(self.btn_save)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        body_layout = QHBoxLayout()
        # 左侧原始视频
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("原始视频流", alignment=Qt.AlignCenter))
        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFrameShape(QFrame.Box)
        self.input_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.input_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.input_label)
        body_layout.addLayout(left_layout, 1)
        # 右侧追踪结果
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("目标追踪结果", alignment=Qt.AlignCenter))
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.result_label)
        body_layout.addLayout(right_layout, 1)
        main_layout.addLayout(body_layout, 5)
        # 下方信息区
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setFont(QFont("Consolas", 9))
        main_layout.addWidget(self.info_text, 2)

    def on_tracker_type_changed(self, text):
        self.tracker_type = text

    def upload_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.input_path = path
            self.input_type = "视频"
            cap = cv2.VideoCapture(path)
            ret, img = cap.read()
            if ret:
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
            cap.release()
            self.info_text.append(f"已选择视频: {path}")

    def open_camera(self):
        self.input_path = 0
        self.input_type = "摄像头"
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            QMessageBox.critical(self, "错误", "摄像头无法打开")
            return
        ret, img = cap.read()
        if ret:
            self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
        cap.release()
        self.info_text.append("已打开摄像头")

    def start_tracking(self):
        if YOLO is None:
            QMessageBox.critical(self, "错误", "未安装ultralytics库")
            return
        if self.input_path is None:
            QMessageBox.information(self, "提示", "请先上传视频或打开摄像头")
            return
        self.running = True
        self.save_frames = []
        self.track_history = []
        threading.Thread(target=self.tracking_thread, daemon=True).start()

    def stop_tracking(self):
        self.running = False
        self.info_text.append("已停止追踪")

    def tracking_thread(self):
        try:
            # 选择模型
            if self.tracker_type == "多目标追踪":
                model = YOLO("yolov8n.pt")  # 多目标追踪模型
                tracker = "bytetrack.yaml"
            else:
                model = YOLO("yolov8n.pt")  # 单目标追踪可用同一模型，后处理只保留最大目标
                tracker = "botsort.yaml"
            cap = cv2.VideoCapture(self.input_path)
            frame_idx = 0
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                self.input_label.setPixmap(cvimg2qt(frame).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                # 推理+追踪
                results = model.track(frame, persist=True, tracker=tracker)
                result_img = results[0].plot()
                # 解析追踪信息
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy() if hasattr(boxes, 'id') and boxes.id is not None else []
                xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else []
                confs = boxes.conf.cpu().numpy() if boxes is not None and hasattr(boxes, 'conf') else []
                info_lines = []
                track_info = []
                for i, box in enumerate(xyxy):
                    tid = int(ids[i]) if i < len(ids) else -1
                    conf = confs[i] if i < len(confs) else 0
                    x1, y1, x2, y2 = map(int, box)
                    info_lines.append(f"Frame{frame_idx} ID:{tid} 坐标:({x1},{y1},{x2},{y2}) 置信度:{conf:.2f}")
                    track_info.append({"id": tid, "bbox": [x1, y1, x2, y2], "conf": float(conf)})
                self.result_label.setPixmap(cvimg2qt(result_img).scaled(self.result_label.size(), Qt.KeepAspectRatio))
                self.info_text.append("\n".join(info_lines))
                self.track_history.append((frame_idx, track_info))
                self.save_frames.append(result_img.copy())
                frame_idx += 1
                cv2.waitKey(1)
            cap.release()
            self.info_text.append("追踪完成")
        except Exception as e:
            self.info_text.append(f"追踪异常: {e}")

    def save_results(self):
        if not self.save_frames:
            QMessageBox.information(self, "提示", "没有可保存的追踪结果")
            return
        # 保存视频
        base = "tracking_result"
        ext = "mp4"
        idx = 1
        while os.path.exists(f"{base}_{idx}.{ext}"):
            idx += 1
        video_path = f"{base}_{idx}.{ext}"
        h, w = self.save_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20, (w, h))
        for frame in self.save_frames:
            out.write(frame)
        out.release()
        # 保存追踪坐标
        txt_path = f"{base}_{idx}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for frame_idx, track_info in self.track_history:
                for obj in track_info:
                    f.write(f"Frame{frame_idx} ID:{obj['id']} BBox:{obj['bbox']} Conf:{obj['conf']:.2f}\n")
        QMessageBox.information(self, "保存成功", f"追踪视频: {video_path}\n追踪坐标: {txt_path}")
