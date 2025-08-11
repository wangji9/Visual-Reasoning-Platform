import cv2
import threading
import time
import os
from PyQt5.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from trajectory import TrajectoryGenerator

def cvimg2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

class TrajectoryWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("轨迹生成")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)  # 独立窗口，允许最大化/最小化
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.input_path = None
        self.input_type = None
        self.running = False
        self.paused = False
        self.trajectory_gen = TrajectoryGenerator()
        self.video_frames = []
        self.traj_frames = []
        self.info_lines = []
        self.thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        top_layout = QHBoxLayout()
        self.label_input = QLabel("原始视频区")
        self.label_input.setAlignment(Qt.AlignCenter)
        self.label_input.setStyleSheet("background:#f7f7f7;border:1px solid #bbb;")
        self.label_traj = QLabel("轨迹生成区")
        self.label_traj.setAlignment(Qt.AlignCenter)
        self.label_traj.setStyleSheet("background:#f7f7f7;border:1px solid #bbb;")
        top_layout.addWidget(self.label_input, 1)
        top_layout.addWidget(self.label_traj, 1)
        layout.addLayout(top_layout, 5)
        btn_layout = QHBoxLayout()
        self.btn_upload = QPushButton("上传视频")
        self.btn_upload.clicked.connect(self.upload_video)
        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.clicked.connect(self.open_camera)
        self.btn_start = QPushButton("开始轨迹生成")
        self.btn_start.clicked.connect(self.start_trajectory)
        self.btn_pause = QPushButton("暂停")
        self.btn_pause.clicked.connect(self.pause_trajectory)
        self.btn_pause.setEnabled(False)
        self.btn_reset = QPushButton("复位")
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addWidget(self.btn_camera)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        layout.addWidget(self.text_info, 2)

    def upload_video(self):
        if self.running:
            QMessageBox.warning(self, "提示", "请先停止或复位后再上传新视频！")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self.input_path = path
            self.input_type = "视频"
            self.text_info.append(f"已选择视频: {path}")

    def open_camera(self):
        if self.running:
            QMessageBox.warning(self, "提示", "请先停止或复位后再打开摄像头！")
            return
        self.input_path = 0
        self.input_type = "摄像头"
        self.text_info.append("已打开摄像头")

    def start_trajectory(self):
        if self.input_path is None:
            QMessageBox.warning(self, "提示", "请先上传视频或打开摄像头！")
            return
        if self.running:
            QMessageBox.information(self, "提示", "轨迹生成已在进行中！")
            return
        self.running = True
        self.paused = False
        self.btn_pause.setText("暂停")
        self.btn_pause.setEnabled(True)
        self.btn_start.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.trajectory_gen.reset()
        self.video_frames = []
        self.traj_frames = []
        self.info_lines = []
        self.thread = threading.Thread(target=self.trajectory_thread, daemon=True)
        self.thread.start()

    def pause_trajectory(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.setText("继续")
            self.append_info("已暂停。")
        else:
            self.btn_pause.setText("暂停")
            self.append_info("继续轨迹生成。")

    def trajectory_thread(self):
        cap = cv2.VideoCapture(self.input_path)
        frame_idx = 0
        while self.running and cap.isOpened():
            if self.paused:
                time.sleep(0.1)
                continue
            ret, img = cap.read()
            if not ret:
                break
            self.video_frames.append(img.copy())
            centers, track_history, bboxes, track_ids, results = self.trajectory_gen.infer(img)
            traj_img = self.trajectory_gen.draw_trajectories(img, track_history, bboxes, track_ids)
            self.traj_frames.append(traj_img.copy())
            self.update_display(img, traj_img)
            # 展示轨迹坐标
            info_lines = [f"帧{frame_idx}: 检测目标数={len(centers)}"]
            for tid in track_ids:
                pts = track_history.get(tid, [])
                info_lines.append(f"目标ID {tid} 轨迹: {pts}")
            info = "\n".join(info_lines)
            self.info_lines.append(info)
            self.append_info(info)
            frame_idx += 1
            cv2.waitKey(1)
        cap.release()
        self.running = False
        self.btn_pause.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_camera.setEnabled(True)
        self.append_info("轨迹生成完成！")

    def update_display(self, img, traj_img):
        h, w = self.label_input.height(), self.label_input.width()
        self.label_input.setPixmap(cvimg2qt(img).scaled(w, h, Qt.KeepAspectRatio))
        h2, w2 = self.label_traj.height(), self.label_traj.width()
        self.label_traj.setPixmap(cvimg2qt(traj_img).scaled(w2, h2, Qt.KeepAspectRatio))

    def append_info(self, msg):
        self.text_info.append(msg)

    def reset_all(self):
        self.running = False
        self.paused = False
        self.input_path = None
        self.input_type = None
        self.trajectory_gen.reset()
        self.video_frames = []
        self.traj_frames = []
        self.info_lines = []
        self.label_input.clear()
        self.label_input.setText("原始视频区")
        self.label_traj.clear()
        self.label_traj.setText("轨迹生成区")
        self.text_info.clear()
        self.btn_pause.setText("暂停")
        self.btn_pause.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_camera.setEnabled(True)

    def save_result(self):
        if not self.traj_frames:
            QMessageBox.information(self, "提示", "没有可保存的轨迹生成结果！")
            return
        base = "trajectory_result"
        ext = "mp4"
        idx = 1
        while os.path.exists(f"{base}_{idx}.{ext}"):
            idx += 1
        path = f"{base}_{idx}.{ext}"
        h, w = self.traj_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 20, (w, h))
        for frame in self.traj_frames:
            out.write(frame)
        out.release()
        # 保存中间信息和轨迹坐标
        info_path = f"{base}_{idx}_info.txt"
        with open(info_path, "w", encoding="utf-8") as f:
            for line in self.info_lines:
                f.write(line + "\n")
        QMessageBox.information(self, "保存成功", f"轨迹视频保存为: {path}\n信息保存为: {info_path}")

    def closeEvent(self, event):
        self.running = False
        self.paused = False
        event.accept()
