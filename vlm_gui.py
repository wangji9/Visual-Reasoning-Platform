from PyQt5.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QFileDialog, QSpinBox, QSizePolicy, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
import datetime
import os
from vlm import QwenVL2B

class VLMWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像理解 (通义千问2-VL-2B)")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.api_key = None
        self.model = "qwen-vl-plus"
        self.vlm = None
        self.input_type = "图片"
        self.input_path = None
        self.interval = 1
        self.timer = None
        self.cap = None
        self.last_frame = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        # 顶部API-Key输入
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("API-Key:"))
        self.api_input = QLineEdit()
        self.api_input.setEchoMode(QLineEdit.Password)
        self.api_input.setPlaceholderText("请输入通义千问API-Key")
        api_layout.addWidget(self.api_input)
        self.btn_api = QPushButton("确认")
        self.btn_api.clicked.connect(self.confirm_api)
        api_layout.addWidget(self.btn_api)
        self.api_status = QLabel()
        api_layout.addWidget(self.api_status)
        layout.addLayout(api_layout)

        # 工具栏
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(QLabel("输入方式："))
        self.input_combo = QComboBox()
        self.input_combo.addItems(["图片", "视频", "摄像头"])
        self.input_combo.currentTextChanged.connect(self.on_input_type_changed)
        tool_layout.addWidget(self.input_combo)
        self.btn_open = QPushButton("读取/打开")
        self.btn_open.clicked.connect(self.open_input)
        tool_layout.addWidget(self.btn_open)
        tool_layout.addWidget(QLabel("间隔(帧/秒)："))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 1000)
        self.interval_spin.setValue(1)
        self.interval_spin.setSingleStep(1)
        tool_layout.addWidget(self.interval_spin)
        self.btn_start = QPushButton("开始理解")
        self.btn_start.clicked.connect(self.start_infer)
        tool_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_infer)
        tool_layout.addWidget(self.btn_stop)
        # 新增保存结果按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        tool_layout.addWidget(self.btn_save)
        tool_layout.addStretch()
        layout.addLayout(tool_layout)

        # 主体区域
        body_layout = QHBoxLayout()
        # 左侧：原始画面
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("原始输入"))
        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFrameShape(QLabel.Box)
        self.input_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.input_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.input_label)
        body_layout.addLayout(left_layout, 1)
        # 右侧：理解结果
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("理解结果"))

        # === 新增：理解结果区域一分为二 ===
        result_split_layout = QVBoxLayout()

        # 上半部分：用户输入区域
        user_input_layout = QHBoxLayout()
        self.user_prompt_input = QLineEdit()
        self.user_prompt_input.setPlaceholderText("请输入您对图像的提问（如：请描述画面中的主要内容）")
        user_input_layout.addWidget(self.user_prompt_input)
        self.btn_set_prompt = QPushButton("设置提问")
        self.btn_set_prompt.clicked.connect(self.set_user_prompt)
        user_input_layout.addWidget(self.btn_set_prompt)
        result_split_layout.addLayout(user_input_layout)

        # 下半部分：内容理解区域
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 10))
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_split_layout.addWidget(self.result_text)

        right_layout.addLayout(result_split_layout)
        body_layout.addLayout(right_layout, 1)
        layout.addLayout(body_layout, 5)

        # 新增：默认用户提问
        self.user_prompt = "请用中文描述当前画面"

    def confirm_api(self):
        key = self.api_input.text().strip()
        if not key:
            self.api_status.setText("<font color='red'>请输入API-Key</font>")
            return
        try:
            self.vlm = QwenVL2B(api_key=key)
            self.api_key = key
            self.api_input.setEchoMode(QLineEdit.Password)
            self.api_input.setEnabled(False)
            self.btn_api.setEnabled(False)
            self.api_status.setText("<font color='green'>API-Key已确认</font>")
        except Exception as e:
            self.api_status.setText(f"<font color='red'>无效Key: {e}</font>")

    def on_input_type_changed(self, text):
        self.input_type = text
        self.input_label.clear()
        self.result_text.clear()
        self.input_path = None
        self.stop_infer()

    def open_input(self):
        if self.input_type == "图片":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
            if path:
                self.input_path = path
                img = cv2.imread(path)
                self.last_frame = img
                self.show_image(img)
        elif self.input_type == "视频":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
            if path:
                self.input_path = path
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                if ret:
                    self.last_frame = frame
                    self.show_image(frame)
                cap.release()
        elif self.input_type == "摄像头":
            self.input_path = None
            cap = cv2.VideoCapture(0) 
            ret, frame = cap.read()
            if ret:
                self.last_frame = frame
                self.show_image(frame)
            cap.release()

    def show_image(self, img):
        # 固定显示区域大小，防止原始输入视频大小变化
        fixed_width = 480
        fixed_height = 360
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img)
        self.input_label.setPixmap(pix.scaled(fixed_width, fixed_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_user_prompt(self):
        prompt = self.user_prompt_input.text().strip()
        if prompt:
            self.user_prompt = prompt
            self.result_text.append(f"[用户提问已设置]：{prompt}")
        else:
            self.user_prompt = "请用中文描述当前画面"
            self.result_text.append("[用户提问已重置为默认]")

    def start_infer(self):
        if not self.vlm:
            self.api_status.setText("<font color='red'>请先确认API-Key</font>")
            return
        # self.result_text.clear()  # 移除这里的清空，避免推理结果被覆盖
        self.interval = self.interval_spin.value()
        if self.input_type == "图片":
            if self.last_frame is None:
                self.result_text.setPlainText("请先选择图片")
                return
            self.do_image_infer(self.last_frame)
        elif self.input_type == "视频":
            if not self.input_path:
                self.result_text.setPlainText("请先选择视频")
                return
            self.do_video_infer(self.input_path)
        elif self.input_type == "摄像头":
            self.do_camera_infer()

    def do_image_infer(self, img):
        prompt = self.user_prompt if hasattr(self, 'user_prompt') else "请用中文描述图片"
        try:
            text = self.vlm.infer_image(img, prompt=prompt)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.result_text.setPlainText(
                f"[用户提问]\n{prompt}\n\n[{now}]\n模型: {self.model}\n理解内容:\n[提问内容] {prompt}\n[模型回答] {text}"
            )
        except Exception as e:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.result_text.setPlainText(
                f"[用户提问]\n{prompt}\n\n[{now}]\n模型: {self.model}\n理解内容:\n[提问内容] {prompt}\n[模型回答] 推理失败: {str(e)}"
            )

    def do_video_infer(self, path):
        prompt = self.user_prompt if hasattr(self, 'user_prompt') else "请用中文描述当前帧"
        cap = cv2.VideoCapture(path)
        idx = 0
        interval = self.interval
        all_results = []
        save_dir = os.path.join(os.getcwd(), "图像理解")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_name = os.path.basename(path)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_txt_path = os.path.join(save_dir, f"{now}_{video_name}_all.txt")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                self.last_frame = frame  # 便于保存
                self.show_image(frame)
                try:
                    text = self.vlm.infer_image(frame, prompt=prompt)
                except Exception as e:
                    text = f"推理失败: {str(e)}"
                readable_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result = (
                    f"[用户提问]\n{prompt}\n\n[{readable_time}]\n视频: {path}\n第{idx}帧\n模型: {self.model}\n理解内容:\n[提问内容] {prompt}\n[模型回答] {text}\n"
                )
                all_results.append(result)
                self.result_text.setPlainText(result)
                QApplication.processEvents()
            idx += 1
        cap.release()
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_results))
        self.result_text.append(f"\n[提示] 所有帧的理解内容已保存到: {summary_txt_path}")

    def do_camera_infer(self):
        prompt = self.user_prompt if hasattr(self, 'user_prompt') else "请用中文描述当前画面"
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # 固定分辨率，保证视频流大小一致
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera_frame_idx = 0
        self.camera_history = []
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(os.getcwd(), "图像理解")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.camera_video_path = os.path.join(save_dir, f"{now}_camera.avi")
        self.camera_history_path = os.path.join(save_dir, f"{now}_camera_history.txt")
        self._camera_history_file = open(self.camera_history_path, "w", encoding="utf-8")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20.0
        width = 640
        height = 480
        self.camera_video_writer = cv2.VideoWriter(self.camera_video_path, fourcc, fps, (width, height))
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.camera_step_optimized(prompt))
        self.timer.start(33)  # 固定约30fps流畅刷新
        self._camera_infer_interval = self.interval

    def camera_step_optimized(self, prompt):
        if not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.result_text.setPlainText("无法读取摄像头")
            self.stop_infer()
            return
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        self.last_frame = frame
        self.show_image(frame)
        if hasattr(self, 'camera_video_writer') and self.camera_video_writer is not None:
            self.camera_video_writer.write(frame)
        # 只每N帧推理一次，其余帧只刷新画面
        if self.camera_frame_idx % self._camera_infer_interval == 0:
            # 直接在主线程推理，保证推理结果能显示出来，哪怕视频卡顿
            try:
                text = self.vlm.infer_image(frame, prompt=prompt)
                if isinstance(text, bytes):
                    text = text.decode('utf-8', errors='ignore')
            except Exception as e:
                text = f"推理失败: {str(e)}"
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = (
                f"[用户提问]\n{prompt}\n\n[{now}]\n摄像头第{self.camera_frame_idx}帧\n模型: {self.model}\n理解内容:\n[提问内容] {prompt}\n[模型回答] {text}\n"
            )
            self.result_text.setReadOnly(False)
            self.result_text.append(result)
            self.result_text.ensureCursorVisible()
            self.result_text.repaint()
            self.result_text.viewport().update()
            self.result_text.setReadOnly(True)
            if hasattr(self, 'camera_history'):
                self.camera_history.append(result)
            if hasattr(self, '_camera_history_file') and self._camera_history_file:
                self._camera_history_file.write(result + "\n")
                self._camera_history_file.flush()
        self.camera_frame_idx += 1
        QApplication.processEvents()

    def closeEvent(self, event):
        self.stop_infer()
        event.accept()

    def stop_infer(self):
        """停止推理相关的定时器和资源释放。"""
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if hasattr(self, 'camera_video_writer') and self.camera_video_writer is not None:
            self.camera_video_writer.release()
            self.camera_video_writer = None
        # 关闭内容理解txt文件
        if hasattr(self, '_camera_history_file') and self._camera_history_file:
            self._camera_history_file.close()
            self._camera_history_file = None

    def save_result(self):
        """保存当前理解结果到用户指定的txt文件。"""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        text = self.result_text.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "提示", "没有可保存的内容！")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存理解结果", "理解结果.txt", "Text Files (*.txt)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                QMessageBox.information(self, "保存成功", f"理解结果已保存到: {path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"保存文件时出错: {e}")
