import sys
import os
import time
import threading
import traceback
import psutil
import cv2
import datetime
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox, QFileDialog,
    QHBoxLayout, QVBoxLayout, QTextEdit, QStatusBar, QLineEdit, QDoubleSpinBox, QMessageBox,
    QMenuBar, QAction, QSizePolicy, QSpacerItem, QFrame, QDialog  # ← 新增QDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QPixmap, QImage, QFont, QColor, QIcon

from detect import ObjectDetector
from seg import Segmentor
from pos import PoseEstimator
from handpos import HandPoseEstimator
from face import FaceLandmarkGaze, draw_face_landmarks_and_gaze
from image_processing_gui import ImageProcessingWindow

# 将OpenCV的BGR图像转换为Qt可用的QPixmap
def cvimg2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

# 信号类，用于多线程推理时主线程与子线程通信
class WorkerSignals(QObject):
    result = pyqtSignal(object, object, object, object)
    status = pyqtSignal(int)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    input_img = pyqtSignal(object)

class MainWindow(QMainWindow):
    # 主窗口初始化，设置界面、信号、菜单等
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的多任务视觉推理平台v1.0")
        # 设置窗口icon为logo.png
        logo_icon_path = os.path.join(os.path.dirname(__file__), 'logo.png')
        if os.path.exists(logo_icon_path):
            self.setWindowIcon(QIcon(logo_icon_path))

        self.setMinimumSize(900, 600)
        self.setMaximumSize(1920, 1200)

        self.model_type = None
        self.weight_path = None
        self.model = None
        self.input_type = "图片"
        self.input_path = None
        self.running = False
        self.log_lines = []
        self.last_result_img = None
        self.save_video_frames = []
        self.result_history = []  # 新增：用于保存每次推理的结果信息

        self.signals = WorkerSignals()
        self.signals.result.connect(self.on_infer_result)
        self.signals.status.connect(self.on_infer_status)
        self.signals.error.connect(self.show_error)
        self.signals.log.connect(self.log)
        self.signals.input_img.connect(self.on_input_img)

        self.image_processing_window = None  # 新增：用于保存子窗口实例
        self.vlm_window = None  # 新增：用于保存VLMWindow实例
        self.model_explain_window = None  # 新增：用于保存模型解释窗口实例
        self.trajectory_window = None  # 新增：用于保存TrajectoryWindow实例
        self.tracking_window = None  # 新增：用于保存TrackingWindow实例
        self.sam_window = None  # 新增：用于保存SAMWindow实例
        self.obb_window = None  # 新增：用于保存OBBWindow实例
        self.solution_window = None  # 新增：用于保存解决方案窗口实例

        self.init_menu()
        self.init_ui()
        self.init_statusbar()
        self.init_timer()

    # 初始化菜单栏及各功能菜单
    def init_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件")
        open_action = QAction("打开图片/视频", self)
        open_action.triggered.connect(self.upload_input)
        file_menu.addAction(open_action)

        # 新增保存结果菜单项
        save_action = QAction("保存结果", self)
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)

        # 新增保存日志菜单项
        save_log_action = QAction("保存日志", self)
        save_log_action.triggered.connect(self.save_log)
        file_menu.addAction(save_log_action)

        # 新增保存结果信息菜单项
        save_info_action = QAction("保存结果信息", self)
        save_info_action.triggered.connect(self.save_result_info)
        file_menu.addAction(save_info_action)

        # 新增复位菜单项
        reset_action = QAction("复位", self)
        reset_action.triggered.connect(self.reset_all)
        file_menu.addAction(reset_action)

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 新增模型信息菜单项
        modelinfo_action = QAction("模型信息", self)
        modelinfo_action.triggered.connect(self.show_model_info)
        menubar.addAction(modelinfo_action)

        # 新增图像处理功能菜单项
        process_action = QAction("图像处理", self)
        process_action.triggered.connect(self.open_image_processing_window)
        menubar.addAction(process_action)

        # 新增图像理解功能菜单项
        from vlm_gui import VLMWindow
        vlm_action = QAction("图像理解", self)
        def open_vlm_window():
            if self.vlm_window is None:
                self.vlm_window = VLMWindow(self)
            self.vlm_window.show()
            self.vlm_window.raise_()
            self.vlm_window.activateWindow()
        vlm_action.triggered.connect(open_vlm_window)
        menubar.addAction(vlm_action)

        # 新增模型解释功能菜单项
        from model_explain_gui import ModelExplainWindow
        self.model_explain_window = None
        explain_action = QAction("模型解释", self)
        def open_model_explain_window():
            if self.model_explain_window is None:
                self.model_explain_window = ModelExplainWindow(self)
            self.model_explain_window.show()
            self.model_explain_window.raise_()
            self.model_explain_window.activateWindow()
        explain_action.triggered.connect(open_model_explain_window)
        menubar.addAction(explain_action)

        # 新增轨迹生成菜单项
        from trajectory_gui import TrajectoryWindow
        self.trajectory_window = None
        traj_action = QAction("轨迹生成", self)
        def open_trajectory_window():
            if self.trajectory_window is None:
                self.trajectory_window = TrajectoryWindow(self)
            self.trajectory_window.show()
            self.trajectory_window.raise_()
            self.trajectory_window.activateWindow()
        traj_action.triggered.connect(open_trajectory_window)
        menubar.addAction(traj_action)

        # 新增目标追踪菜单项
        from tracking_gui import TrackingWindow
        self.tracking_window = None
        tracking_action = QAction("目标追踪", self)
        def open_tracking_window():
            if self.tracking_window is None:
                self.tracking_window = TrackingWindow(self)
            self.tracking_window.show()
            self.tracking_window.raise_()
            self.tracking_window.activateWindow()
        tracking_action.triggered.connect(open_tracking_window)
        menubar.addAction(tracking_action)

        # 新增SAM实例分割菜单项
        from sam_gui import SAMWindow
        self.sam_window = None
        sam_action = QAction("SAM", self)
        def open_sam_window():
            if self.sam_window is None:
                self.sam_window = SAMWindow(self)
            self.sam_window.show()
            self.sam_window.raise_()
            self.sam_window.activateWindow()
        sam_action.triggered.connect(open_sam_window)
        menubar.addAction(sam_action)

        # 新增OBB检测菜单项
        from obbgui import OBBWindow
        self.obb_window = None
        obb_action = QAction("OBB检测", self)
        def open_obb_window():
            if self.obb_window is None:
                self.obb_window = OBBWindow(self)
            self.obb_window.show()
            self.obb_window.raise_()
            self.obb_window.activateWindow()
        obb_action.triggered.connect(open_obb_window)
        menubar.addAction(obb_action)

        # 新增Ultralytics解决方案菜单项
        from solution_gui import SolutionGUI
        self.solution_window = None
        solution_action = QAction("解决方案", self)
        def open_solution_window():
            if self.solution_window is None:
                self.solution_window = SolutionGUI(self)
            self.solution_window.show()
            self.solution_window.raise_()
            self.solution_window.activateWindow()
        solution_action.triggered.connect(open_solution_window)
        menubar.addAction(solution_action)

    # 打开图像处理子窗口
    def open_image_processing_window(self):
        if self.image_processing_window is None:
            self.image_processing_window = ImageProcessingWindow(self)
            self.image_processing_window.setWindowFlags(
                self.image_processing_window.windowFlags() | Qt.Window
            )  # 允许拖动、独立窗口
        self.image_processing_window.show()
        self.image_processing_window.raise_()
        self.image_processing_window.activateWindow()

    # 初始化主界面UI布局
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 工具栏分为两行
        toolbar1 = QHBoxLayout()
        toolbar2 = QHBoxLayout()
        toolbar1.setSpacing(8)
        toolbar2.setSpacing(8)
        # 第一行：输入方式、上传、复位、模型分组、模型类型、无需权重模型
        toolbar1.addWidget(QLabel("输入方式："))
        self.input_combo = QComboBox()
        self.input_combo.addItems(["图片", "视频", "摄像头"])
        self.input_combo.setFixedWidth(90)
        self.input_combo.currentTextChanged.connect(self.on_input_type_changed)
        toolbar1.addWidget(self.input_combo)

        self.btn_upload = QPushButton("上传/打开")
        self.btn_upload.setFixedWidth(90)
        self.btn_upload.clicked.connect(self.upload_input)
        toolbar1.addWidget(self.btn_upload)

        # 新增复位按钮
        self.btn_reset = QPushButton("复位")
        self.btn_reset.setFixedWidth(70)
        self.btn_reset.clicked.connect(self.reset_all)
        toolbar1.addWidget(self.btn_reset)

        toolbar1.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))

        toolbar1.addWidget(QLabel("模型分组："))
        self.model_group = QComboBox()
        self.model_group.addItems(["检测/分割/姿态", "无需权重模型"])
        self.model_group.setFixedWidth(110)
        self.model_group.currentTextChanged.connect(self.on_model_group_changed)
        toolbar1.addWidget(self.model_group)

        # 模型选择下拉框，首项为空
        self.model_combo = QComboBox()
        self.model_combo.addItems(["", "目标检测", "图像分割", "姿态估计"])
        self.model_combo.setFixedWidth(110)
        self.model_combo.currentTextChanged.connect(self.on_model_type_changed)
        toolbar1.addWidget(self.model_combo)

        self.no_weight_combo = QComboBox()
        self.no_weight_combo.addItems(["", "手部关键点检测", "面部关键点与视线方向"])
        self.no_weight_combo.setFixedWidth(150)
        self.no_weight_combo.currentTextChanged.connect(self.on_no_weight_model_changed)
        self.no_weight_combo.setVisible(False)
        toolbar1.addWidget(self.no_weight_combo)

        # 第二行：权重、YOLO版本、下载权重、置信度、推理按钮
        self.btn_weight = QPushButton("上传权重")
        self.btn_weight.setFixedWidth(90)
        self.btn_weight.clicked.connect(self.select_weight)
        self.btn_weight.setEnabled(False)  # 初始化时禁用
        toolbar2.addWidget(self.btn_weight)
        self.weight_label = QLabel("未选择")
        self.weight_label.setMinimumWidth(70)
        self.weight_label.setEnabled(False)  # 初始化时禁用
        toolbar2.addWidget(self.weight_label)

        # YOLO权重具体文件名下拉框，自动获取官方支持权重
        self.yolo_weight_combo = QComboBox()
        weight_names = []
        try:
            from ultralytics import YOLO
            if hasattr(YOLO, 'models') and callable(YOLO.models):
                # YOLO.models() 返回dict或list
                models = YOLO.models()
                if isinstance(models, dict):
                    weight_names = [k for k in models.keys() if k.endswith('.pt')]
                elif isinstance(models, list):
                    weight_names = [k for k in models if isinstance(k, str) and k.endswith('.pt')]
        except Exception:
            pass
        if not weight_names:
            weight_names = ["yolov8n.pt"]
        weight_names.sort()
        self.yolo_weight_combo.addItems(weight_names)
        self.yolo_weight_combo.setVisible(False)
        toolbar2.addWidget(QLabel("权重："))
        toolbar2.addWidget(self.yolo_weight_combo)
        self.btn_download_weight = QPushButton("下载权重")
        self.btn_download_weight.setVisible(False)
        self.btn_download_weight.clicked.connect(self.download_yolo_weight)
        toolbar2.addWidget(self.btn_download_weight)

        toolbar2.addWidget(QLabel("置信度："))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setFixedWidth(60)
        toolbar2.addWidget(self.conf_spin)

        self.btn_infer = QPushButton("开始推理")
        self.btn_infer.setFixedWidth(90)
        self.btn_infer.clicked.connect(self.start_infer)
        toolbar2.addWidget(self.btn_infer)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedWidth(70)
        self.btn_stop.clicked.connect(self.stop_infer)
        toolbar2.addWidget(self.btn_stop)

        main_layout.addLayout(toolbar1)
        main_layout.addLayout(toolbar2)

        # 主体区域
        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)
        body_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧输入
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

        # 右侧结果
        right_layout = QVBoxLayout()
        right_layout.setSpacing(2)
        self.right_title = QLabel("推理结果")
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

        # 下方结果信息和日志并排
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(8)

        # 结果信息
        info_layout = QVBoxLayout()
        info_title = QLabel("结果信息")
        info_title.setFont(QFont("微软雅黑", 9, QFont.Bold))
        info_title.setAlignment(Qt.AlignLeft)
        info_layout.addWidget(info_title)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 关键：自适应
        info_layout.addWidget(self.info_text)
        bottom_layout.addLayout(info_layout, 1)

        # 日志
        log_layout = QVBoxLayout()
        log_title = QLabel("日志")
        log_title.setFont(QFont("微软雅黑", 9, QFont.Bold))
        log_title.setAlignment(Qt.AlignLeft)
        log_layout.addWidget(log_title)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # 关键：自适应
        log_layout.addWidget(self.log_text)
        bottom_layout.addLayout(log_layout, 1)

        main_layout.addLayout(bottom_layout, 2)

    # 初始化状态栏
    def init_statusbar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_time = QLabel("推理时间: 0 ms")
        self.status.addWidget(self.status_time)
        self.status_cpu = QLabel("CPU: 0%")
        self.status.addWidget(self.status_cpu)
        self.status_mem = QLabel("内存: 0%")
        self.status.addWidget(self.status_mem)
        self.status_clock = QLabel(time.strftime("%H:%M:%S"))
        self.status.addWidget(self.status_clock)

    # 初始化定时器，定时刷新状态栏信息
    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)

    # 刷新CPU和内存状态
    def update_status(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        self.status_cpu.setText(f"CPU: {cpu}%")
        self.status_mem.setText(f"内存: {mem}%")

    # 刷新时钟显示
    def update_clock(self):
        self.status_clock.setText(time.strftime("%H:%M:%S"))

    # 日志输出到日志区
    def log(self, msg):
        now = time.strftime("[%H:%M:%S] ")
        self.log_lines.append(now + msg)
        self.log_text.append(now + msg)

    # 弹窗显示错误信息
    def show_error(self, msg):
        QMessageBox.critical(self, "错误", msg)
        self.log(f"错误: {msg}")

    # 弹窗显示提示信息
    def show_info(self, msg):
        QMessageBox.information(self, "提示", msg)
        self.log(f"提示: {msg}")

    # 上传输入图片、视频或摄像头
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

    # 输入方式切换时的处理
    def on_input_type_changed(self, text):
        self.input_type = text
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.last_result_img = None
        self.input_path = None
        self.save_video_frames = []

    # 切换模型类型时的处理
    def on_model_type_changed(self, text):
        self.model_type = text
        self.right_title.setText(f"{text}推理结果" if text else "推理结果")
        enable = text in ["目标检测", "图像分割", "姿态估计"]
        self.btn_weight.setEnabled(enable)
        self.weight_label.setEnabled(enable)
        self.model = None
        self.weight_path = None
        self.weight_label.setText("未选择")
        # 目标检测时显示YOLO权重选择和下载权重按钮
        if text == "目标检测":
            self.yolo_weight_combo.setVisible(True)
            self.btn_download_weight.setVisible(True)
        else:
            self.yolo_weight_combo.setVisible(False)
            self.btn_download_weight.setVisible(False)

    # 切换模型分组时的处理
    def on_model_group_changed(self, text):
        if text == "检测/分割/姿态":
            self.model_combo.setVisible(True)
            self.btn_weight.setVisible(True)
            self.weight_label.setVisible(True)
            self.no_weight_combo.setVisible(False)
            self.model_type = None
            self.model_combo.setCurrentIndex(0)  # 默认空
            self.model = None
            self.weight_path = None
            self.weight_label.setText("未选择")
            self.btn_weight.setEnabled(False)
            self.weight_label.setEnabled(False)
        else:
            self.model_combo.setVisible(False)
            self.btn_weight.setVisible(False)
            self.weight_label.setVisible(False)
            self.no_weight_combo.setVisible(True)
            self.model_type = None
            self.no_weight_combo.setCurrentIndex(0)  # 默认空
            self.weight_path = None
            self.weight_label.setText("无需权重")
            self.on_no_weight_model_changed(self.model_type)
        self.right_title.setText("推理结果")

    # 切换无需权重模型时的处理
    def on_no_weight_model_changed(self, text):
        self.model_type = text
        self.right_title.setText(f"{text}推理结果")
        self.btn_weight.setEnabled(False)
        self.weight_label.setText("无需权重")
        if self.model_type == "手部关键点检测":
            self.model = HandPoseEstimator()
        elif self.model_type == "面部关键点与视线方向":
            self.model = FaceLandmarkGaze()
        else:
            self.model = None

    # 选择权重文件并加载模型
    def select_weight(self):
        if self.model_group.currentText() != "检测/分割/姿态":
            self.show_info("当前模型无需上传权重文件。")
            return
        if self.model_type not in ["目标检测", "图像分割", "姿态估计"]:
            self.show_info("请选择正确的模型类型。")
            return
        # 目标检测时，优先用yolo_version_combo选择的权重
        if self.model_type == "目标检测" and self.yolo_weight_combo.isVisible() and self.weight_path:
            try:
                self.model = ObjectDetector(self.weight_path)
                self.weight_label.setText(os.path.basename(self.weight_path))
                self.log("模型加载成功")
            except Exception as e:
                self.show_error(f"模型加载失败: {e}")
                self.model = None
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch Weights (*.pt *.pth)")
        if path:
            self.weight_path = path
            self.weight_label.setText(os.path.basename(path))
            try:
                if self.model_type == "目标检测":
                    self.model = ObjectDetector(path)
                elif self.model_type == "图像分割":
                    self.model = Segmentor(path)
                elif self.model_type == "姿态估计":
                    self.model = PoseEstimator(path)
                else:
                    raise ValueError("未知模型类型")
                self.log("模型加载成功")
            except Exception as e:
                self.show_error(f"模型加载失败: {e}")
                self.model = None

    # 启动推理线程
    def start_infer(self):
        self.running = False
        # 检测/分割/姿态模型
        if self.model_group.currentText() == "检测/分割/姿态":
            if self.model is None:
                self.show_error("请先选择模型和权重")
                return
            if self.input_path is None:
                self.show_error("请先上传输入数据")
                return
        # 无需权重模型
        elif self.model_group.currentText() == "无需权重模型":
            if self.model_type == "手部关键点检测":
                self.model = HandPoseEstimator()
            elif self.model_type == "面部关键点与视线方向":
                self.model = FaceLandmarkGaze()
            else:
                self.model = None
            if self.model is None:
                self.show_error("未知模型类型")
                return
            if self.input_path is None:
                self.show_error("请先上传输入数据")
                return
        else:
            self.show_error("未知模型类型")
            return
        self.running = True
        self.log("开始推理")
        self.info_text.clear()
        self.result_label.clear()
        self.last_result_img = None
        self.save_video_frames = []
        threading.Thread(target=self.infer_thread, daemon=True).start()

    # 停止推理
    def stop_infer(self):
        self.running = False
        self.log("停止推理")

    # 推理线程，处理图片/视频/摄像头的推理主循环
    def infer_thread(self):
        try:
            t0 = time.time()
            conf = self.conf_spin.value()
            if self.input_type == "图片":
                img = cv2.imread(self.input_path)
                self.input_img = img
                self.signals.input_img.emit(img)
                result_img, info, classes, results = self.run_infer(img, conf)
                self.signals.result.emit(result_img, info, classes, results)
            elif self.input_type == "视频":
                cap = cv2.VideoCapture(self.input_path)
                self.save_video_frames = []
                while self.running and cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break
                    self.signals.input_img.emit(img)
                    result_img, info, classes, results = self.run_infer(img, conf)
                    self.signals.result.emit(result_img, info, classes, results)
                    self.save_video_frames.append(result_img.copy())
                    cv2.waitKey(1)
                cap.release()
            elif self.input_type == "摄像头":
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("摄像头无法打开，请检查设备。")
                self.save_video_frames = []
                while self.running and cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break
                    self.signals.input_img.emit(img)  # 关键：每帧都emit，实时显示
                    result_img, info, classes, results = self.run_infer(img, conf)
                    self.signals.result.emit(result_img, info, classes, results)
                    self.save_video_frames.append(result_img.copy())
                    cv2.waitKey(1)
                cap.release()
            t1 = time.time()
            infer_time = int((t1 - t0) * 1000)
            self.signals.status.emit(infer_time)
            self.signals.log.emit(f"推理完成，耗时{infer_time} ms")
        except Exception as e:
            self.signals.error.emit(f"推理异常: {e}\n{traceback.format_exc()}")

    # 执行推理，返回推理结果和信息
    def run_infer(self, img, conf):
        if self.model_type == "目标检测":
            result_img, boxes_info, classes, results = self.model.infer(img)
            confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
            return result_img, (boxes_info, classes, confs), classes, results
        elif self.model_type == "图像分割":
            result_img, seg_info, results = self.model.infer(img)
            confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []
            classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []
            return result_img, (seg_info, classes, confs), classes, results
        elif self.model_type == "姿态估计":
            # 用cuda推理
            result_img, keypoints_info, results = self.model.infer(img)
            confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []
            classes = results[0].boxes.cls.cpu().numpy() if hasattr(results[0], "boxes") and results[0].boxes is not None else []
            return result_img, (keypoints_info, classes, confs), classes, results
        elif self.model_type == "手部关键点检测":
            result_img, hand_keypoints_info, results = self.model.infer(img)
            return result_img, (hand_keypoints_info, None, None), None, results
        elif self.model_type == "面部关键点与视线方向":
            result_img, faces, gaze_list = self.model.infer(img)
            return result_img, (faces, gaze_list), None, None
        else:
            raise ValueError("未知模型类型")

    # 推理结果回调，显示结果图片和信息
    def on_infer_result(self, result_img, info, classes, results):
        self.last_result_img = result_img.copy()
        self.input_img = self.input_img  # 保证有输入图像
        img = result_img.copy()
        # 直接渲染详细信息（坐标、类别、置信度等）
        if self.model_type == "目标检测":
            boxes_info, classes, confs = info
            for box in boxes_info:
                if isinstance(box, (tuple, list, np.ndarray)) and len(box) == 4:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        elif self.model_type == "图像分割":
            seg_info, classes, confs = info
            for coords in seg_info:
                if coords:
                    pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], isClosed=True, color=(0,0,255), thickness=2)
        elif self.model_type == "姿态估计":
            keypoints_info, classes, confs = info
            for point in keypoints_info:
                if isinstance(point, (tuple, list, np.ndarray)) and len(point) == 2:
                    x, y = point
                    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
        elif self.model_type == "手部关键点检测":
            hand_keypoints_info, _, _ = info
            for hand in hand_keypoints_info:
                for point in hand:
                    if isinstance(point, (tuple, list, np.ndarray)) and len(point) == 2:
                        x, y = point
                        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        elif self.model_type == "面部关键点与视线方向":
            faces, gaze_list = info
            img = draw_face_landmarks_and_gaze(img, faces, gaze_list)
        self.result_label.setPixmap(cvimg2qt(img).scaled(self.result_label.size(), Qt.KeepAspectRatio))
        # 新增：记录每次推理的结果信息和时间戳
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = self.get_result_info_text(info, classes)
        self.result_history.append((timestamp, info_text))
        self.show_result_info(info, classes)

    # 生成推理结果信息文本
    def get_result_info_text(self, info, classes):
        """将结果信息转为字符串，便于保存"""
        lines = []
        if self.model_type == "目标检测":
            boxes_info, classes, confs = info
            for idx, (box, cls_id, conf) in enumerate(zip(boxes_info, classes, confs)):
                lines.append(f"Obj{idx}: 坐标{box}, 类别{int(cls_id)}, 置信度:{conf:.2f}")
        elif self.model_type == "图像分割":
            seg_info, classes, confs = info
            for idx, (coords, cls_id, conf) in enumerate(zip(seg_info, classes, confs)):
                lines.append(f"Obj{idx}: 类别{int(cls_id)}, 置信度:{conf:.2f}")
        elif self.model_type == "姿态估计":
            keypoints_info, classes, confs = info
            for idx, ((x, y), cls_id, conf) in enumerate(zip(keypoints_info, classes, confs)):
                lines.append(f"关键点{idx}: ({x},{y}), 类别{int(cls_id)}, 置信度:{conf:.2f}")
        elif self.model_type == "手部关键点检测":
            hand_keypoints_info, _, _ = info
            for hand_idx, hand in enumerate(hand_keypoints_info):
                lines.append(f"手{hand_idx+1}关键点: {hand}")
        elif self.model_type == "面部关键点与视线方向":
            faces, gaze_list = info
            for idx, (face, gaze) in enumerate(zip(faces, gaze_list)):
                lines.append(f"人脸{idx+1}关键点数: {len(face)}")
                lines.append(f"左眼三维坐标: {gaze['left_pupil_3d']}")
                lines.append(f"右眼三维坐标: {gaze['right_pupil_3d']}")
                lines.append(f"左眼视线方向: {gaze['left_eye_vec']}")
                lines.append(f"右眼视线方向: {gaze['right_eye_vec']}")
                lines.append("")
        else:
            lines.append("无推理结果")
        return "\n".join(lines)

    # 显示推理结果信息到界面
    def show_result_info(self, info, classes):
        """显示推理结果信息到界面"""
        info_text = self.get_result_info_text(info, classes)
        self.info_text.setPlainText(info_text)

    # 保存日志到本地文件
    def save_log(self):
        import glob
        base = "log"
        ext = "txt"
        files = glob.glob(f"{base}_*.{ext}")
        nums = []
        for f in files:
            try:
                num = int(os.path.splitext(f)[0].split("_")[-1])
                nums.append(num)
            except Exception:
                continue
        next_num = max(nums) + 1 if nums else 1
        path = f"{base}_{next_num}.{ext}"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log_lines))
        self.log(f"日志已保存到: {path}")
        self.show_info(f"日志保存成功！文件名：{path}")

    # 保存推理结果信息到本地文件
    def save_result_info(self):
        import glob
        base = "result_info"
        ext = "txt"
        files = glob.glob(f"{base}_*.{ext}")
        nums = []
        for f in files:
            try:
                num = int(os.path.splitext(f)[0].split("_")[-1])
                nums.append(num)
            except Exception:
                continue
        next_num = max(nums) + 1 if nums else 1
        path = f"{base}_{next_num}.{ext}"
        with open(path, "w", encoding="utf-8") as f:
            for timestamp, info_text in self.result_history:
                f.write(f"推理时间: {timestamp}\n")
                f.write(info_text)
                f.write("\n" + "="*40 + "\n")
        self.log(f"结果信息已保存到: {path}")
        self.show_info(f"结果信息保存成功！文件名：{path}")

    # 保存推理结果图片或视频到本地
    def save_result(self):
        import glob

        if self.last_result_img is None and not self.save_video_frames:
            self.show_info("没有可保存的推理结果。")
            return

        def get_next_filename(base, ext):
            files = glob.glob(f"{base}_*.{ext}")
            nums = []
            for f in files:
                try:
                    num = int(os.path.splitext(f)[0].split("_")[-1])
                    nums.append(num)
                except Exception:
                    continue
            next_num = max(nums) + 1 if nums else 1
            return f"{base}_{next_num}.{ext}"

        if self.input_type == "图片":
            # 自动生成文件名
            base = "result"
            ext = "jpg"
            path = get_next_filename(base, ext)
            cv2.imwrite(path, self.last_result_img)
            self.log(f"推理结果图片已保存到: {path}")
            self.show_info(f"图片保存成功！文件名：{path}")
        elif self.input_type in ["视频", "摄像头"]:
            if not self.save_video_frames:
                self.show_info("没有可保存的视频结果。")
                return
            base = "result_video"
            ext = "mp4"
            path = get_next_filename(base, ext)
            h, w = self.save_video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 20, (w, h))
            for frame in self.save_video_frames:
                out.write(frame)
            out.release()
            self.log(f"推理结果视频已保存到: {path}")
            self.show_info(f"视频保存成功！文件名：{path}")
        else:
            self.show_info("当前输入类型不支持保存。")

    # 推理完成后更新时间显示
    def on_infer_status(self, infer_time):
        """推理完成后更新时间显示"""
        self.status_time.setText(f"推理时间: {infer_time} ms")

    # 实时显示输入帧到原始输入区
    def on_input_img(self, img):
        # 实时显示输入帧到原始输入区
        self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))

    # 复位所有界面和变量到初始状态
    def reset_all(self):
        """复位所有界面和变量到初始状态"""
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.log_text.clear()
        self.last_result_img = None
        self.input_path = None
        self.save_video_frames = []
        self.result_history = []
        self.model = None
        self.weight_path = None
        self.model_type = None
        self.input_type = "图片"
        self.input_combo.setCurrentIndex(0)
        self.model_group.setCurrentIndex(0)
        self.model_combo.setCurrentIndex(0)
        self.no_weight_combo.setCurrentIndex(0)
        self.weight_label.setText("未选择")
        self.btn_weight.setEnabled(False)
        self.weight_label.setEnabled(False)
        self.log("已复位所有界面和变量")

    # 弹窗显示当前已加载模型的参数统计和结构信息，仅限检测/分割/姿态分组
    def show_model_info(self):
        """弹窗显示当前已加载模型的参数统计和结构信息，仅限检测/分割/姿态分组"""
        if self.model_group.currentText() != "检测/分割/姿态" or self.model is None or self.weight_path is None:
            QMessageBox.information(self, "模型信息", "当前无可统计的模型信息，仅支持已上传权重的检测/分割/姿态模型。")
            return
        # 获取模型参数统计和结构
        try:
            info_str = str(self.model.model.model)  # 结构
            param_info = self.model.model.info()    # 参数统计
        except Exception as e:
            QMessageBox.warning(self, "模型信息", f"获取模型信息失败：{e}")
            return
        # 弹窗显示
        dlg = QDialog(self)
        dlg.setWindowTitle("模型参数与结构信息")
        dlg.setMinimumSize(700, 500)
        layout = QVBoxLayout(dlg)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Consolas", 10))
        text.setPlainText(f"【模型参数统计】\n{param_info}\n\n【模型结构】\n{info_str}")
        layout.addWidget(text)
        btn = QPushButton("关闭")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn, alignment=Qt.AlignRight)
        dlg.setLayout(layout)
        dlg.exec_()

    def download_yolo_weight(self):
        import shutil
        import glob
        import os
        import sys
        weight_name = self.yolo_weight_combo.currentText()
        try:
            from ultralytics import YOLO
            self.log(f"准备下载/加载YOLO权重: {weight_name}")
            # 先尝试在当前目录查找
            local_path = os.path.abspath(weight_name)
            if os.path.exists(local_path):
                self.log(f"检测到本地已存在权重文件: {local_path}")
                self.weight_path = local_path
                self.model = ObjectDetector(self.weight_path)
                self.weight_label.setText(os.path.basename(self.weight_path))
                self.log(f"{weight_name}已存在于当前目录，直接加载: {self.weight_path}")
                QMessageBox.information(self, "加载完成", f"{weight_name}已存在于当前目录，直接加载！")
                return
            self.log(f"本地未找到{weight_name}，开始用Ultralytics自动下载...")
            model = YOLO(weight_name)
            # 获取实际权重路径
            weight_path = None
            if hasattr(model, 'ckpt_path') and model.ckpt_path and os.path.exists(model.ckpt_path):
                weight_path = model.ckpt_path
            elif hasattr(model, 'model') and hasattr(model.model, 'pt_path') and model.model.pt_path and os.path.exists(model.model.pt_path):
                weight_path = model.model.pt_path
            # 如果Ultralytics未返回权重路径，则手动查找缓存目录
            if not weight_path or not os.path.exists(weight_path):
                # Ultralytics默认缓存目录
                if sys.platform.startswith('win'):
                    cache_dir = os.path.expandvars(r'%USERPROFILE%/.cache/ultralytics/weights')
                else:
                    cache_dir = os.path.expanduser('~/.cache/ultralytics/weights')
                search_path = os.path.join(cache_dir, weight_name)
                self.log(f"尝试在缓存目录查找: {search_path}")
                if os.path.exists(search_path):
                    weight_path = search_path
                else:
                    # 兼容ultralytics/weights下所有同名文件
                    files = glob.glob(os.path.join(cache_dir, f"*{weight_name}"))
                    if files:
                        weight_path = files[0]
                if not weight_path or not os.path.exists(weight_path):
                    self.log(f"权重文件未找到: {weight_name}。可能下载失败或权重名错误。")
                    QMessageBox.critical(self, "下载失败", f"权重文件未找到: {weight_name}\n请检查网络或权重名是否正确。")
                    return
            # 复制到当前目录
            if not os.path.exists(local_path):
                shutil.copy(weight_path, local_path)
                self.log(f"已将权重从{weight_path}复制到当前目录: {local_path}")
            else:
                self.log(f"目标文件{local_path}已存在，无需复制。")
            self.weight_path = local_path
            self.weight_label.setText(os.path.basename(self.weight_path))
            self.model = ObjectDetector(self.weight_path)
            self.log(f"{weight_name}权重下载并复制到当前目录并加载成功: {self.weight_path}")
            QMessageBox.information(self, "下载完成", f"{weight_name}权重下载并复制到当前目录并加载成功！")
        except Exception as e:
            self.log(f"权重下载失败: {e}")
            QMessageBox.critical(self, "下载失败", f"权重下载失败: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())