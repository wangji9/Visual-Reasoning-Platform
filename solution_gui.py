import sys
import os
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSizePolicy,
    QFileDialog, QTextEdit, QComboBox, QFrame, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from ultralytics import solutions

# 解决方案子功能及中文
SOLUTION_FEATURES = [
    ("目标计数",),
    ("目标裁剪",),
    ("目标模糊",),
    ("锻炼监测",),
    ("区域内目标计数",),
    ("安全报警系统",),
    ("热力图",),
    ("实例分割与目标跟踪",),
    ("VisionEye视图对象映射",),
    ("速度估计",),
    ("距离计算",),
    ("排队管理",),
    ("停车管理",),
    ("分析",),
    ("实时推理",),
    ("区域内目标跟踪",),
    ("相似性检索",)
]

class SolutionSubWindow(QMainWindow):
    def __init__(self, cn_name, parent=None):
        super().__init__(parent)
        self.cn_name = cn_name
        self.setWindowTitle(f"{cn_name}")
        self.setMinimumSize(900, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.model_path = None
        self.input_type = "图片"
        self.input_path = None
        self.input_img = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_video_frame)
        self.sol = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        # 工具栏分为两行：输入操作区和参数设置区
        toolbar1 = QHBoxLayout()
        toolbar2 = QHBoxLayout()
        # 输入操作区
        self.input_combo = QComboBox()
        self.input_combo.addItems(["图片", "视频", "摄像头"])
        self.input_combo.setFixedWidth(90)
        self.input_combo.currentTextChanged.connect(self.on_input_type_changed)
        toolbar1.addWidget(QLabel("输入方式："))
        toolbar1.addWidget(self.input_combo)
        self.btn_upload = QPushButton("上传/打开")
        self.btn_upload.setFixedWidth(90)
        self.btn_upload.clicked.connect(self.upload_input)
        toolbar1.addWidget(self.btn_upload)
        self.btn_weight = QPushButton("上传权重")
        self.btn_weight.setFixedWidth(90)
        self.btn_weight.clicked.connect(self.select_weight)
        toolbar1.addWidget(self.btn_weight)
        self.weight_label = QLabel("未选择")
        self.weight_label.setMinimumWidth(70)
        toolbar1.addWidget(self.weight_label)
        self.btn_start = QPushButton("开始")
        self.btn_start.setFixedWidth(70)
        self.btn_start.clicked.connect(self.start_infer)
        toolbar1.addWidget(self.btn_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setFixedWidth(70)
        self.btn_stop.clicked.connect(self.stop_infer)
        toolbar1.addWidget(self.btn_stop)
        self.btn_reset = QPushButton("复位")
        self.btn_reset.setFixedWidth(70)
        self.btn_reset.clicked.connect(self.reset_all)
        toolbar1.addWidget(self.btn_reset)
        self.param_combo = QComboBox()
        self.param_combo.setFixedWidth(120)
        self.param_combo.addItem("默认参数")
        toolbar1.addWidget(self.param_combo)
        main_layout.addLayout(toolbar1)
        # 参数设置区（原toolbar改为toolbar2）
        toolbar = toolbar2
        # 主体区域
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
        result_title = QLabel("{0}结果区".format(self.cn_name))
        result_title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(result_title)
        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.result_label)
        # 相似性检索专用：文本输入和检索按钮
        if self.cn_name == "相似性检索":
            search_layout = QHBoxLayout()
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("请输入检索文本，如：一只狗坐在长椅上")
            search_layout.addWidget(self.search_input)
            self.search_btn = QPushButton("检索")
            self.search_btn.clicked.connect(self.on_search_clicked)
            search_layout.addWidget(self.search_btn)
            right_layout.addLayout(search_layout)
        body_layout.addLayout(right_layout, 1)
        main_layout.addLayout(body_layout, 5)
        # 信息区
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.info_text, 2)

        # 目标计数专用参数设置区
        if self.cn_name == "目标计数":
            param_layout = QHBoxLayout()
            self.region_combo = QComboBox()
            self.region_combo.addItems(["矩形区域", "直线计数", "多边形区域"])
            self.region_combo.setFixedWidth(100)
            self.region_combo.currentTextChanged.connect(self.on_region_type_changed)
            param_layout.addWidget(QLabel("计数区域类型："))
            param_layout.addWidget(self.region_combo)
            self.class_input = QLineEdit()
            self.class_input.setPlaceholderText("指定类别（如0,2,3，留空为全部）")
            self.class_input.setFixedWidth(180)
            param_layout.addWidget(QLabel("计数类别："))
            param_layout.addWidget(self.class_input)
            self.conf_input = QLineEdit()
            self.conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.conf_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.conf_input)
            self.iou_input = QLineEdit()
            self.iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.iou_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.iou_input)
            self.tracker_combo = QComboBox()
            self.tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.tracker_combo.setFixedWidth(120)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.tracker_combo)
            toolbar.addLayout(param_layout)
        # 目标裁剪专用参数设置区
        if self.cn_name == "目标裁剪":
            param_layout = QHBoxLayout()
            self.crop_class_input = QLineEdit()
            self.crop_class_input.setPlaceholderText("指定类别（如0,2，留空为全部）")
            self.crop_class_input.setFixedWidth(180)
            param_layout.addWidget(QLabel("裁剪类别："))
            param_layout.addWidget(self.crop_class_input)
            self.crop_conf_input = QLineEdit()
            self.crop_conf_input.setPlaceholderText("置信度阈值，默认0.25")
            self.crop_conf_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.crop_conf_input)
            self.crop_dir_input = QLineEdit()
            self.crop_dir_input.setPlaceholderText("裁剪保存目录，默认cropped-detections")
            self.crop_dir_input.setFixedWidth(180)
            param_layout.addWidget(QLabel("保存目录："))
            param_layout.addWidget(self.crop_dir_input)
            toolbar.addLayout(param_layout)
        # 目标模糊专用参数设置区
        if self.cn_name == "目标模糊":
            param_layout = QHBoxLayout()
            self.blur_class_input = QLineEdit()
            self.blur_class_input.setPlaceholderText("指定类别（如0,2，留空为全部）")
            self.blur_class_input.setFixedWidth(180)
            param_layout.addWidget(QLabel("模糊类别："))
            param_layout.addWidget(self.blur_class_input)
            self.blur_conf_input = QLineEdit()
            self.blur_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.blur_conf_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.blur_conf_input)
            self.blur_iou_input = QLineEdit()
            self.blur_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.blur_iou_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.blur_iou_input)
            self.blur_ratio_input = QLineEdit()
            self.blur_ratio_input.setPlaceholderText("模糊强度0.1-1.0，默认0.5")
            self.blur_ratio_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("模糊强度："))
            param_layout.addWidget(self.blur_ratio_input)
            self.blur_tracker_combo = QComboBox()
            self.blur_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.blur_tracker_combo.setFixedWidth(120)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.blur_tracker_combo)
            toolbar.addLayout(param_layout)
        # 锻炼监测专用参数设置区
        if self.cn_name == "锻炼监测":
            param_layout = QHBoxLayout()
            self.gym_type_combo = QComboBox()
            self.gym_type_combo.addItems(["俯卧撑", "引体向上", "仰卧起坐"])
            self.gym_type_combo.setFixedWidth(100)
            param_layout.addWidget(QLabel("锻炼类型："))
            param_layout.addWidget(self.gym_type_combo)
            self.kpts_input = QLineEdit()
            self.kpts_input.setPlaceholderText("关键点序号，如6,8,10")
            self.kpts_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("关键点："))
            param_layout.addWidget(self.kpts_input)
            self.up_angle_input = QLineEdit()
            self.up_angle_input.setPlaceholderText("上肢角度，默认145.0")
            self.up_angle_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("上肢角度："))
            param_layout.addWidget(self.up_angle_input)
            self.down_angle_input = QLineEdit()
            self.down_angle_input.setPlaceholderText("下肢角度，默认90.0")
            self.down_angle_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("下肢角度："))
            param_layout.addWidget(self.down_angle_input)
            self.gym_conf_input = QLineEdit()
            self.gym_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.gym_conf_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.gym_conf_input)
            self.gym_iou_input = QLineEdit()
            self.gym_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.gym_iou_input.setFixedWidth(100)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.gym_iou_input)
            self.gym_tracker_combo = QComboBox()
            self.gym_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.gym_tracker_combo.setFixedWidth(120)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.gym_tracker_combo)
            toolbar.addLayout(param_layout)
            # 类型切换自动填充关键点
            self.gym_type_combo.currentTextChanged.connect(self.on_gym_type_changed)
        # 区域内目标计数专用参数设置区
        if self.cn_name == "区域内目标计数":
            param_layout = QHBoxLayout()
            # 区域类型选择
            self.region_type_combo = QComboBox()
            self.region_type_combo.addItems(["单区域", "多区域", "字典"])
            self.region_type_combo.setFixedWidth(80)
            param_layout.addWidget(QLabel("区域类型："))
            param_layout.addWidget(self.region_type_combo)
            # 区域点多行输入
            self.region_points_input = QTextEdit()
            self.region_points_input.setPlaceholderText("每行一个区域，格式如：x1,y1;x2,y2;...\n字典格式示例：region-01:x1,y1;x2,y2;x3,y3;x4,y4")
            self.region_points_input.setFixedHeight(50)
            self.region_points_input.setFixedWidth(260)
            param_layout.addWidget(QLabel("区域点："))
            param_layout.addWidget(self.region_points_input)
            # 类别
            self.region_class_input = QLineEdit()
            self.region_class_input.setPlaceholderText("指定类别（如0,2,3，留空为全部）")
            self.region_class_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("类别："))
            param_layout.addWidget(self.region_class_input)
            # 置信度
            self.region_conf_input = QLineEdit()
            self.region_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.region_conf_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.region_conf_input)
            # IoU
            self.region_iou_input = QLineEdit()
            self.region_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.region_iou_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.region_iou_input)
            # 跟踪算法
            self.region_tracker_combo = QComboBox()
            self.region_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.region_tracker_combo.setFixedWidth(110)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.region_tracker_combo)
            # 显示置信度
            self.region_show_conf_combo = QComboBox()
            self.region_show_conf_combo.addItems(["显示", "不显示"])
            self.region_show_conf_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示置信度："))
            param_layout.addWidget(self.region_show_conf_combo)
            # 显示标签
            self.region_show_labels_combo = QComboBox()
            self.region_show_labels_combo.addItems(["显示", "不显示"])
            self.region_show_labels_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示标签："))
            param_layout.addWidget(self.region_show_labels_combo)
            # 线宽
            self.region_line_width_input = QLineEdit()
            self.region_line_width_input.setPlaceholderText("线宽，留空自动")
            self.region_line_width_input.setFixedWidth(60)
            param_layout.addWidget(QLabel("线宽："))
            param_layout.addWidget(self.region_line_width_input)
            toolbar.addLayout(param_layout)
            # 区域类型切换时自动填充示例
            self.region_type_combo.currentTextChanged.connect(self.on_region_type_changed)
        # 安全报警系统专用参数设置区
        if self.cn_name == "安全报警系统":
            param_layout = QHBoxLayout()
            # 检测类别
            self.alarm_class_input = QLineEdit()
            self.alarm_class_input.setPlaceholderText("指定类别（如0,2,3，留空为全部）")
            self.alarm_class_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("检测类别："))
            param_layout.addWidget(self.alarm_class_input)
            # 置信度
            self.alarm_conf_input = QLineEdit()
            self.alarm_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.alarm_conf_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.alarm_conf_input)
            # IoU
            self.alarm_iou_input = QLineEdit()
            self.alarm_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.alarm_iou_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.alarm_iou_input)
            # 跟踪算法
            self.alarm_tracker_combo = QComboBox()
            self.alarm_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.alarm_tracker_combo.setFixedWidth(110)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.alarm_tracker_combo)
            # 触发报警目标数
            self.alarm_records_input = QLineEdit()
            self.alarm_records_input.setPlaceholderText("报警目标数，默认5")
            self.alarm_records_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("报警目标数："))
            param_layout.addWidget(self.alarm_records_input)
            # 显示置信度
            self.alarm_show_conf_combo = QComboBox()
            self.alarm_show_conf_combo.addItems(["显示", "不显示"])
            self.alarm_show_conf_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示置信度："))
            param_layout.addWidget(self.alarm_show_conf_combo)
            # 显示标签
            self.alarm_show_labels_combo = QComboBox()
            self.alarm_show_labels_combo.addItems(["显示", "不显示"])
            self.alarm_show_labels_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示标签："))
            param_layout.addWidget(self.alarm_show_labels_combo)
            # 线宽
            self.alarm_line_width_input = QLineEdit()
            self.alarm_line_width_input.setPlaceholderText("线宽，留空自动")
            self.alarm_line_width_input.setFixedWidth(60)
            param_layout.addWidget(QLabel("线宽："))
            param_layout.addWidget(self.alarm_line_width_input)
            # 邮箱参数
            self.alarm_from_email = QLineEdit()
            self.alarm_from_email.setPlaceholderText("发件邮箱")
            self.alarm_from_email.setFixedWidth(140)
            param_layout.addWidget(QLabel("发件邮箱："))
            param_layout.addWidget(self.alarm_from_email)
            self.alarm_email_pwd = QLineEdit()
            self.alarm_email_pwd.setPlaceholderText("邮箱授权码/密码")
            self.alarm_email_pwd.setEchoMode(QLineEdit.Password)
            self.alarm_email_pwd.setFixedWidth(120)
            param_layout.addWidget(QLabel("邮箱授权码："))
            param_layout.addWidget(self.alarm_email_pwd)
            self.alarm_to_email = QLineEdit()
            self.alarm_to_email.setPlaceholderText("收件邮箱")
            self.alarm_to_email.setFixedWidth(140)
            param_layout.addWidget(QLabel("收件邮箱："))
            param_layout.addWidget(self.alarm_to_email)
            toolbar.addLayout(param_layout)
        # 热力图专用参数设置区
        if self.cn_name == "热力图":
            param_layout = QHBoxLayout()
            # 选择colormap
            self.heatmap_colormap_combo = QComboBox()
            self.heatmap_colormap_combo.addItems([
                "JET", "PARULA", "AUTUMN", "BONE", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", "PINK", "HOT", "MAGMA", "INFERNO", "PLASMA", "VIRIDIS", "CIVIDIS", "TWILIGHT", "TWILIGHT_SHIFTED", "TURBO", "DEEPGREEN"
            ])
            self.heatmap_colormap_combo.setFixedWidth(120)
            param_layout.addWidget(QLabel("色图类型："))
            param_layout.addWidget(self.heatmap_colormap_combo)
            # 显示进出计数
            self.heatmap_show_in_combo = QComboBox()
            self.heatmap_show_in_combo.addItems(["显示", "不显示"])
            self.heatmap_show_in_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示进入计数："))
            param_layout.addWidget(self.heatmap_show_in_combo)
            self.heatmap_show_out_combo = QComboBox()
            self.heatmap_show_out_combo.addItems(["显示", "不显示"])
            self.heatmap_show_out_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示离开计数："))
            param_layout.addWidget(self.heatmap_show_out_combo)
            # 区域点
            self.heatmap_region_input = QLineEdit()
            self.heatmap_region_input.setPlaceholderText("区域点，如20,400;1080,400")
            self.heatmap_region_input.setFixedWidth(200)
            param_layout.addWidget(QLabel("区域点："))
            param_layout.addWidget(self.heatmap_region_input)
            # 检测类别
            self.heatmap_class_input = QLineEdit()
            self.heatmap_class_input.setPlaceholderText("指定类别（如0,2，留空为全部）")
            self.heatmap_class_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("检测类别："))
            param_layout.addWidget(self.heatmap_class_input)
            # 置信度
            self.heatmap_conf_input = QLineEdit()
            self.heatmap_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.heatmap_conf_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.heatmap_conf_input)
            # IoU
            self.heatmap_iou_input = QLineEdit()
            self.heatmap_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.heatmap_iou_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.heatmap_iou_input)
            # 跟踪算法
            self.heatmap_tracker_combo = QComboBox()
            self.heatmap_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.heatmap_tracker_combo.setFixedWidth(110)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.heatmap_tracker_combo)
            # 显示置信度
            self.heatmap_show_conf_combo = QComboBox()
            self.heatmap_show_conf_combo.addItems(["显示", "不显示"])
            self.heatmap_show_conf_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示置信度："))
            param_layout.addWidget(self.heatmap_show_conf_combo)
            # 显示标签
            self.heatmap_show_labels_combo = QComboBox()
            self.heatmap_show_labels_combo.addItems(["显示", "不显示"])
            self.heatmap_show_labels_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示标签："))
            param_layout.addWidget(self.heatmap_show_labels_combo)
            # 线宽
            self.heatmap_line_width_input = QLineEdit()
            self.heatmap_line_width_input.setPlaceholderText("线宽，留空自动")
            self.heatmap_line_width_input.setFixedWidth(60)
            param_layout.addWidget(QLabel("线宽："))
            param_layout.addWidget(self.heatmap_line_width_input)
            toolbar.addLayout(param_layout)
        # 实例分割与目标跟踪专用参数设置区
        if self.cn_name == "实例分割与目标跟踪":
            param_layout = QHBoxLayout()
            # 区域点
            self.seg_region_input = QLineEdit()
            self.seg_region_input.setPlaceholderText("区域点，如20,400;1080,400（可选）")
            self.seg_region_input.setFixedWidth(200)
            param_layout.addWidget(QLabel("区域点："))
            param_layout.addWidget(self.seg_region_input)
            # 检测类别
            self.seg_class_input = QLineEdit()
            self.seg_class_input.setPlaceholderText("指定类别（如0,2，留空为全部）")
            self.seg_class_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("检测类别："))
            param_layout.addWidget(self.seg_class_input)
            # 置信度
            self.seg_conf_input = QLineEdit()
            self.seg_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.seg_conf_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.seg_conf_input)
            # IoU
            self.seg_iou_input = QLineEdit()
            self.seg_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.seg_iou_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.seg_iou_input)
            # 跟踪算法
            self.seg_tracker_combo = QComboBox()
            self.seg_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.seg_tracker_combo.setFixedWidth(110)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.seg_tracker_combo)
            # 显示置信度
            self.seg_show_conf_combo = QComboBox()
            self.seg_show_conf_combo.addItems(["显示", "不显示"])
            self.seg_show_conf_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示置信度："))
            param_layout.addWidget(self.seg_show_conf_combo)
            # 显示标签
            self.seg_show_labels_combo = QComboBox()
            self.seg_show_labels_combo.addItems(["显示", "不显示"])
            self.seg_show_labels_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示标签："))
            param_layout.addWidget(self.seg_show_labels_combo)
            # 线宽
            self.seg_line_width_input = QLineEdit()
            self.seg_line_width_input.setPlaceholderText("线宽，留空自动")
            self.seg_line_width_input.setFixedWidth(60)
            param_layout.addWidget(QLabel("线宽："))
            param_layout.addWidget(self.seg_line_width_input)
            toolbar.addLayout(param_layout)
        # VisionEye视图对象映射专用参数设置区
        if self.cn_name == "VisionEye视图对象映射":
            param_layout = QHBoxLayout()
            # 视点
            self.visioneye_point_input = QLineEdit()
            self.visioneye_point_input.setPlaceholderText("视点坐标，如50,50")
            self.visioneye_point_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("视点坐标："))
            param_layout.addWidget(self.visioneye_point_input)
            # 检测类别
            self.visioneye_class_input = QLineEdit()
            self.visioneye_class_input.setPlaceholderText("指定类别（如0,2，留空为全部）")
            self.visioneye_class_input.setFixedWidth(120)
            param_layout.addWidget(QLabel("检测类别："))
            param_layout.addWidget(self.visioneye_class_input)
            # 置信度
            self.visioneye_conf_input = QLineEdit()
            self.visioneye_conf_input.setPlaceholderText("置信度阈值，默认0.3")
            self.visioneye_conf_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("置信度："))
            param_layout.addWidget(self.visioneye_conf_input)
            # IoU
            self.visioneye_iou_input = QLineEdit()
            self.visioneye_iou_input.setPlaceholderText("IoU阈值，默认0.5")
            self.visioneye_iou_input.setFixedWidth(80)
            param_layout.addWidget(QLabel("IoU："))
            param_layout.addWidget(self.visioneye_iou_input)
            # 跟踪算法
            self.visioneye_tracker_combo = QComboBox()
            self.visioneye_tracker_combo.addItems(["botsort.yaml", "bytetrack.yaml"])
            self.visioneye_tracker_combo.setFixedWidth(110)
            param_layout.addWidget(QLabel("跟踪算法："))
            param_layout.addWidget(self.visioneye_tracker_combo)
            # 显示置信度
            self.visioneye_show_conf_combo = QComboBox()
            self.visioneye_show_conf_combo.addItems(["显示", "不显示"])
            self.visioneye_show_conf_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示置信度："))
            param_layout.addWidget(self.visioneye_show_conf_combo)
            # 显示标签
            self.visioneye_show_labels_combo = QComboBox()
            self.visioneye_show_labels_combo.addItems(["显示", "不显示"])
            self.visioneye_show_labels_combo.setFixedWidth(70)
            param_layout.addWidget(QLabel("显示标签："))
            param_layout.addWidget(self.visioneye_show_labels_combo)
            # 线宽
            self.visioneye_line_width_input = QLineEdit()
            self.visioneye_line_width_input.setPlaceholderText("线宽，留空自动")
            self.visioneye_line_width_input.setFixedWidth(60)
            param_layout.addWidget(QLabel("线宽："))
            param_layout.addWidget(self.visioneye_line_width_input)
            toolbar.addLayout(param_layout)
        # ...existing code...

    def on_input_type_changed(self, text):
        self.input_type = text
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.input_path = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()

    def upload_input(self):
        if self.input_type == "图片":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
            if path:
                self.input_path = path
                img = cv2.imread(path)
                self.input_img = img
                self.input_label.setPixmap(self.cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.info_text.append(f"已选择图片: {path}")
        elif self.input_type == "视频":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
            if path:
                self.input_path = path
                self.cap = cv2.VideoCapture(path)
                ret, img = self.cap.read()
                if ret:
                    self.input_img = img
                    self.input_label.setPixmap(self.cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.info_text.append(f"已选择视频: {path}")
        elif self.input_type == "摄像头":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.info_text.append("摄像头无法打开，请检查设备。")
                return
            ret, img = self.cap.read()
            if ret:
                self.input_img = img
                self.input_label.setPixmap(self.cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.input_path = 0
                self.info_text.append("已打开摄像头")

    def select_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch Weights (*.pt *.pth)")
        if path:
            self.model_path = path
            self.weight_label.setText(os.path.basename(path))
            self.info_text.append(f"已选择权重: {path}")

    def start_infer(self):
        if not self.model_path:
            self.info_text.append("请先上传权重文件！")
            return
        if self.input_type == "图片":
            if self.input_img is not None:
                self.run_solution(self.input_img)
        elif self.input_type == "视频":
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.timer.start(1)
        elif self.input_type == "摄像头":
            if self.cap is not None:
                self.timer.start(1)

    def stop_infer(self):
        self.timer.stop()
        self.info_text.append("已停止推理")

    def reset_all(self):
        self.input_label.clear()
        self.result_label.clear()
        self.info_text.clear()
        self.input_path = None
        self.input_img = None
        self.model_path = None
        self.weight_label.setText("未选择")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()
        self.info_text.append("已复位所有界面和变量")

    def process_video_frame(self):
        # 视频流卡顿优化：每帧都推理，提升流畅度
        if self.cap is not None and self.cap.isOpened():
            ret, img = self.cap.read()
            if not ret:
                self.timer.stop()
                self.info_text.append("视频/摄像头读取结束")
                return
            self.input_img = img
            self.input_label.setPixmap(self.cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
            results = self.run_solution(img, return_results=True)
            self.show_result(results)

    def on_region_type_changed(self, text):
        # 可根据选择自动切换region_points
        if text == "矩形区域":
            self.region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]
        elif text == "直线计数":
            self.region_points = [(20, 400), (1080, 400)]
        elif text == "多边形区域":
            self.region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]
        else:
            self.region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

    def on_gym_type_changed(self, text):
        # 根据锻炼类型自动填充关键点
        if text == "俯卧撑":
            self.kpts_input.setText("6,8,10")
        elif text == "引体向上":
            self.kpts_input.setText("5,7,9")
        elif text == "仰卧起坐":
            self.kpts_input.setText("12,14,16")

    def run_solution(self, img, return_results=False):
        try:
            if self.cn_name == "目标计数":
                # 读取参数
                region_points = getattr(self, 'region_points', [(20, 400), (1080, 400), (1080, 360), (20, 360)])
                class_str = self.class_input.text() if hasattr(self, 'class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.conf_input.text()) if hasattr(self, 'conf_input') and self.conf_input.text() else 0.3
                iou = float(self.iou_input.text()) if hasattr(self, 'iou_input') and self.iou_input.text() else 0.5
                tracker = self.tracker_combo.currentText() if hasattr(self, 'tracker_combo') else 'botsort.yaml'
                self.sol = solutions.ObjectCounter(
                    show=True,
                    region=region_points,
                    model=self.model_path,
                    classes=classes,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "目标裁剪":
                # 读取参数
                class_str = self.crop_class_input.text() if hasattr(self, 'crop_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.crop_conf_input.text()) if hasattr(self, 'crop_conf_input') and self.crop_conf_input.text() else 0.25
                crop_dir = self.crop_dir_input.text() if hasattr(self, 'crop_dir_input') and self.crop_dir_input.text() else "cropped-detections"
                self.sol = solutions.ObjectCropper(
                    show=True,
                    model=self.model_path,
                    classes=classes,
                    conf=conf,
                    crop_dir=crop_dir,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "目标模糊":
                # 读取参数
                class_str = self.blur_class_input.text() if hasattr(self, 'blur_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.blur_conf_input.text()) if hasattr(self, 'blur_conf_input') and self.blur_conf_input.text() else 0.3
                iou = float(self.blur_iou_input.text()) if hasattr(self, 'blur_iou_input') and self.blur_iou_input.text() else 0.5
                blur_ratio = float(self.blur_ratio_input.text()) if hasattr(self, 'blur_ratio_input') and self.blur_ratio_input.text() else 0.5
                tracker = self.blur_tracker_combo.currentText() if hasattr(self, 'blur_tracker_combo') else 'botsort.yaml'
                self.sol = solutions.ObjectBlurrer(
                    show=True,
                    model=self.model_path,
                    classes=classes,
                    conf=conf,
                    iou=iou,
                    blur_ratio=blur_ratio,
                    tracker=tracker,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "锻炼监测":
                # 读取参数
                kpts_str = self.kpts_input.text() if hasattr(self, 'kpts_input') and self.kpts_input.text() else "6,8,10"
                kpts = [int(x) for x in kpts_str.split(',') if x.strip().isdigit()]
                up_angle = float(self.up_angle_input.text()) if hasattr(self, 'up_angle_input') and self.up_angle_input.text() else 145.0
                down_angle = float(self.down_angle_input.text()) if hasattr(self, 'down_angle_input') and self.down_angle_input.text() else 90.0
                conf = float(self.gym_conf_input.text()) if hasattr(self, 'gym_conf_input') and self.gym_conf_input.text() else 0.3
                iou = float(self.gym_iou_input.text()) if hasattr(self, 'gym_iou_input') and self.gym_iou_input.text() else 0.5
                tracker = self.gym_tracker_combo.currentText() if hasattr(self, 'gym_tracker_combo') else 'botsort.yaml'
                self.sol = solutions.AIGym(
                    show=True,
                    kpts=kpts,
                    up_angle=up_angle,
                    down_angle=down_angle,
                    model=self.model_path,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "区域内目标计数":
                # 读取参数
                region_type = self.region_type_combo.currentText() if hasattr(self, 'region_type_combo') else "单区域"
                region_points_text = self.region_points_input.toPlainText() if hasattr(self, 'region_points_input') else ""
                # 区域点解析
                if region_type == "单区域":
                    # 只取第一行，格式 x1,y1;x2,y2;...
                    lines = [l for l in region_points_text.strip().split('\n') if l.strip()]
                    if lines:
                        pts = [tuple(map(int, p.split(','))) for p in lines[0].split(';') if ',' in p]
                        regions = [pts]
                    else:
                        regions = [[(20, 400), (1080, 400), (1080, 360), (20, 360)]]
                elif region_type == "多区域":
                    # 每行一个区域
                    regions = []
                    for line in region_points_text.strip().split('\n'):
                        if line.strip():
                            pts = [tuple(map(int, p.split(','))) for p in line.split(';') if ',' in p]
                            if pts:
                                regions.append(pts)
                    if not regions:
                        regions = [[(20, 400), (1080, 400)], [(20, 360), (1080, 360)]]
                elif region_type == "字典":
                    # 每行 region-01:x1,y1;x2,y2;...
                    regions = {}
                    for line in region_points_text.strip().split('\n'):
                        if ':' in line:
                            key, pts_str = line.split(':', 1)
                            pts = [tuple(map(int, p.split(','))) for p in pts_str.split(';') if ',' in p]
                            if pts:
                                regions[key.strip()] = pts
                    if not regions:
                        regions = {"region-01": [(50, 50), (250, 50), (250, 250), (50, 250)], "region-02": [(640, 640), (780, 640), (780, 720), (640, 720)]}
                else:
                    regions = [[(20, 400), (1080, 400), (1080, 360), (20, 360)]]
                # 其它参数
                class_str = self.region_class_input.text() if hasattr(self, 'region_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.region_conf_input.text()) if hasattr(self, 'region_conf_input') and self.region_conf_input.text() else 0.3
                iou = float(self.region_iou_input.text()) if hasattr(self, 'region_iou_input') and self.region_iou_input.text() else 0.5
                tracker = self.region_tracker_combo.currentText() if hasattr(self, 'region_tracker_combo') else 'botsort.yaml'
                show_conf = self.region_show_conf_combo.currentText() == "显示" if hasattr(self, 'region_show_conf_combo') else True
                show_labels = self.region_show_labels_combo.currentText() == "显示" if hasattr(self, 'region_show_labels_combo') else True
                line_width = int(self.region_line_width_input.text()) if hasattr(self, 'region_line_width_input') and self.region_line_width_input.text().isdigit() else None
                self.sol = solutions.RegionCounter(
                    show=True,
                    region=regions,
                    model=self.model_path,
                    classes=classes,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    show_conf=show_conf,
                    show_labels=show_labels,
                    line_width=line_width,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "安全报警系统":
                # 读取参数
                class_str = self.alarm_class_input.text() if hasattr(self, 'alarm_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.alarm_conf_input.text()) if hasattr(self, 'alarm_conf_input') and self.alarm_conf_input.text() else 0.3
                iou = float(self.alarm_iou_input.text()) if hasattr(self, 'alarm_iou_input') and self.alarm_iou_input.text() else 0.5
                tracker = self.alarm_tracker_combo.currentText() if hasattr(self, 'alarm_tracker_combo') else 'botsort.yaml'
                records = int(self.alarm_records_input.text()) if hasattr(self, 'alarm_records_input') and self.alarm_records_input.text().isdigit() else 5
                show_conf = self.alarm_show_conf_combo.currentText() == "显示" if hasattr(self, 'alarm_show_conf_combo') else True
                show_labels = self.alarm_show_labels_combo.currentText() == "显示" if hasattr(self, 'alarm_show_labels_combo') else True
                line_width = int(self.alarm_line_width_input.text()) if hasattr(self, 'alarm_line_width_input') and self.alarm_line_width_input.text().isdigit() else None
                from_email = self.alarm_from_email.text() if hasattr(self, 'alarm_from_email') else ''
                email_pwd = self.alarm_email_pwd.text() if hasattr(self, 'alarm_email_pwd') else ''
                to_email = self.alarm_to_email.text() if hasattr(self, 'alarm_to_email') else ''
                self.sol = solutions.SecurityAlarm(
                    show=True,
                    model=self.model_path,
                    records=records,
                    tracker=tracker,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                    show_conf=show_conf,
                    show_labels=show_labels,
                    line_width=line_width,
                    device='cuda'
                )
                # 邮箱认证
                if from_email and email_pwd and to_email:
                    self.sol.authenticate(from_email, email_pwd, to_email)
                    self.info_text.append(f"已设置报警邮箱：{from_email} -> {to_email}")
                else:
                    self.info_text.append("未设置报警邮箱参数，仅本地报警显示")
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "热力图":
                # 读取参数
                colormap_map = {
                    "JET": cv2.COLORMAP_JET,
                    "PARULA": cv2.COLORMAP_PARULA if hasattr(cv2, 'COLORMAP_PARULA') else cv2.COLORMAP_JET,
                    "AUTUMN": cv2.COLORMAP_AUTUMN,
                    "BONE": cv2.COLORMAP_BONE,
                    "WINTER": cv2.COLORMAP_WINTER,
                    "RAINBOW": cv2.COLORMAP_RAINBOW,
                    "OCEAN": cv2.COLORMAP_OCEAN,
                    "SUMMER": cv2.COLORMAP_SUMMER,
                    "SPRING": cv2.COLORMAP_SPRING,
                    "COOL": cv2.COLORMAP_COOL,
                    "HSV": cv2.COLORMAP_HSV,
                    "PINK": cv2.COLORMAP_PINK,
                    "HOT": cv2.COLORMAP_HOT,
                    "MAGMA": cv2.COLORMAP_MAGMA if hasattr(cv2, 'COLORMAP_MAGMA') else cv2.COLORMAP_JET,
                    "INFERNO": cv2.COLORMAP_INFERNO if hasattr(cv2, 'COLORMAP_INFERNO') else cv2.COLORMAP_JET,
                    "PLASMA": cv2.COLORMAP_PLASMA if hasattr(cv2, 'COLORMAP_PLASMA') else cv2.COLORMAP_JET,
                    "VIRIDIS": cv2.COLORMAP_VIRIDIS if hasattr(cv2, 'COLORMAP_VIRIDIS') else cv2.COLORMAP_JET,
                    "CIVIDIS": cv2.COLORMAP_CIVIDIS if hasattr(cv2, 'COLORMAP_CIVIDIS') else cv2.COLORMAP_JET,
                    "TWILIGHT": cv2.COLORMAP_TWILIGHT if hasattr(cv2, 'COLORMAP_TWILIGHT') else cv2.COLORMAP_JET,
                    "TWILIGHT_SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED if hasattr(cv2, 'COLORMAP_TWILIGHT_SHIFTED') else cv2.COLORMAP_JET,
                    "TURBO": cv2.COLORMAP_TURBO if hasattr(cv2, 'COLORMAP_TURBO') else cv2.COLORMAP_JET,
                    "DEEPGREEN": cv2.COLORMAP_DEEPGREEN if hasattr(cv2, 'COLORMAP_DEEPGREEN') else cv2.COLORMAP_JET,
                }
                colormap_name = self.heatmap_colormap_combo.currentText()
                colormap = colormap_map.get(colormap_name, cv2.COLORMAP_JET)
                show_in = self.heatmap_show_in_combo.currentText() == "显示"
                show_out = self.heatmap_show_out_combo.currentText() == "显示"
                region_text = self.heatmap_region_input.text()
                region = None
                if region_text:
                    try:
                        pts = [tuple(map(int, p.split(','))) for p in region_text.split(';') if ',' in p]
                        if pts:
                            region = pts
                    except Exception:
                        region = None
                class_str = self.heatmap_class_input.text() if hasattr(self, 'heatmap_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.heatmap_conf_input.text()) if hasattr(self, 'heatmap_conf_input') and self.heatmap_conf_input.text() else 0.3
                iou = float(self.heatmap_iou_input.text()) if hasattr(self, 'heatmap_iou_input') and self.heatmap_iou_input.text() else 0.5
                tracker = self.heatmap_tracker_combo.currentText() if hasattr(self, 'heatmap_tracker_combo') else 'botsort.yaml'
                show_conf = self.heatmap_show_conf_combo.currentText() == "显示" if hasattr(self, 'heatmap_show_conf_combo') else True
                show_labels = self.heatmap_show_labels_combo.currentText() == "显示" if hasattr(self, 'heatmap_show_labels_combo') else True
                line_width = int(self.heatmap_line_width_input.text()) if hasattr(self, 'heatmap_line_width_input') and self.heatmap_line_width_input.text().isdigit() else None
                self.sol = solutions.Heatmap(
                    show=True,
                    model=self.model_path,
                    colormap=colormap,
                    show_in=show_in,
                    show_out=show_out,
                    region=region,
                    classes=classes,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    show_conf=show_conf,
                    show_labels=show_labels,
                    line_width=line_width,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "实例分割与目标跟踪":
                # 读取参数
                region_text = self.seg_region_input.text() if hasattr(self, 'seg_region_input') else ''
                region = None
                if region_text:
                    try:
                        pts = [tuple(map(int, p.split(','))) for p in region_text.split(';') if ',' in p]
                        if pts:
                            region = pts
                    except Exception:
                        region = None
                class_str = self.seg_class_input.text() if hasattr(self, 'seg_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.seg_conf_input.text()) if hasattr(self, 'seg_conf_input') and self.seg_conf_input.text() else 0.3
                iou = float(self.seg_iou_input.text()) if hasattr(self, 'seg_iou_input') and self.seg_iou_input.text() else 0.5
                tracker = self.seg_tracker_combo.currentText() if hasattr(self, 'seg_tracker_combo') else 'botsort.yaml'
                show_conf = self.seg_show_conf_combo.currentText() == "显示" if hasattr(self, 'seg_show_conf_combo') else True
                show_labels = self.seg_show_labels_combo.currentText() == "显示" if hasattr(self, 'seg_show_labels_combo') else True
                line_width = int(self.seg_line_width_input.text()) if hasattr(self, 'seg_line_width_input') and self.seg_line_width_input.text().isdigit() else None
                self.sol = solutions.InstanceSegmentation(
                    show=True,
                    model=self.model_path,
                    region=region,
                    classes=classes,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    show_conf=show_conf,
                    show_labels=show_labels,
                    line_width=line_width,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "VisionEye视图对象映射":
                # 读取参数
                point_text = self.visioneye_point_input.text() if hasattr(self, 'visioneye_point_input') else ''
                vision_point = (20, 20)
                if point_text:
                    try:
                        x, y = map(int, point_text.split(','))
                        vision_point = (x, y)
                    except Exception:
                        vision_point = (20, 20)
                class_str = self.visioneye_class_input.text() if hasattr(self, 'visioneye_class_input') else ''
                classes = [int(x) for x in class_str.split(',') if x.strip().isdigit()] if class_str else None
                conf = float(self.visioneye_conf_input.text()) if hasattr(self, 'visioneye_conf_input') and self.visioneye_conf_input.text() else 0.3
                iou = float(self.visioneye_iou_input.text()) if hasattr(self, 'visioneye_iou_input') and self.visioneye_iou_input.text() else 0.5
                tracker = self.visioneye_tracker_combo.currentText() if hasattr(self, 'visioneye_tracker_combo') else 'botsort.yaml'
                show_conf = self.visioneye_show_conf_combo.currentText() == "显示" if hasattr(self, 'visioneye_show_conf_combo') else True
                show_labels = self.visioneye_show_labels_combo.currentText() == "显示" if hasattr(self, 'visioneye_show_labels_combo') else True
                line_width = int(self.visioneye_line_width_input.text()) if hasattr(self, 'visioneye_line_width_input') and self.visioneye_line_width_input.text().isdigit() else None
                self.sol = solutions.VisionEye(
                    show=True,
                    model=self.model_path,
                    vision_point=vision_point,
                    classes=classes,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    show_conf=show_conf,
                    show_labels=show_labels,
                    line_width=line_width,
                    device='cuda'
                )
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "速度估计":
                self.sol = solutions.SpeedEstimator(show=True, model=self.model_path, device='cuda')
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "距离计算":
                self.sol = solutions.DistanceCalculation(show=True, model=self.model_path, device='cuda')
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "排队管理":
                self.sol = solutions.QueueManager(show=True, model=self.model_path, device='cuda')
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "停车管理":
                self.sol = solutions.ParkingPtsSelection(show=True, model=self.model_path, device='cuda')
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "分析":
                self.sol = solutions.Analytics(show=True, model=self.model_path, device='cuda')
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "实时推理":
                solutions.Inference(model=self.model_path, source=self.input_path if self.input_path else 0, device='cuda')
                self.info_text.append("已启动实时推理窗口")
            elif self.cn_name == "区域内目标跟踪":
                self.sol = solutions.TrackZone(show=True, model=self.model_path, device='cuda')
                results = self.sol(img)
                if return_results:
                    return results
                self.show_result(results)
            elif self.cn_name == "相似性检索":
                # 兼容旧接口，优先用文本检索
                if hasattr(self, 'search_input') and hasattr(self, 'searcher'):
                    # 已有文本检索控件，不自动推理
                    return
                else:
                    self.sol = solutions.SearchApp(data="images", model="clip")
                    self.info_text.append("请在上方输入检索文本并点击检索按钮")
            else:
                self.info_text.append("暂未实现该功能")
        except Exception as e:
            self.info_text.append(f"推理异常: {e}")
        if return_results:
            return None

    def show_result(self, results):
        # 针对每个子功能详细显示中间信息
        if results is None:
            self.info_text.append("无推理结果")
            return
        if hasattr(results, 'plot_im'):
            img = results.plot_im
            img_bgr = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.result_label.setPixmap(self.cvimg2qt(img_bgr).scaled(self.result_label.size(), Qt.KeepAspectRatio))
        info = []
        # 展示Ultralytics Solutions参数（如有）
        if hasattr(self, 'sol') and hasattr(self.sol, '__dict__'):
            sol_params = {}
            for k, v in self.sol.__dict__.items():
                if not k.startswith('_') and not callable(v):
                    sol_params[k] = v
            if sol_params:
                info.append("【Ultralytics Solutions参数】")
                for k, v in sol_params.items():
                    info.append(f"{k}: {v}")
        # 展示Ultralytics Solutions运行日志（如有）
        if hasattr(results, 'log') and results.log:
            info.append("【Ultralytics Solutions日志】")
            if isinstance(results.log, str):
                info.append(results.log)
            elif isinstance(results.log, list):
                info.extend([str(line) for line in results.log])
        elif hasattr(results, 'speed') and hasattr(results, 'shape'):
            # 兼容部分solutions返回的速度/shape信息
            info.append("【Ultralytics Solutions日志】")
            info.append(f"Speed: {getattr(results, 'speed', '')} per image at shape {getattr(results, 'shape', '')}")
        # 目标计数
        if self.cn_name == "目标计数":
            if hasattr(results, 'in_count'):
                info.append(f"进入计数: {results.in_count}")
            if hasattr(results, 'out_count'):
                info.append(f"离开计数: {results.out_count}")
            if hasattr(results, 'classwise_counts'):
                info.append(f"类别计数: {results.classwise_counts}")
            if hasattr(results, 'track_ids'):
                info.append(f"当前跟踪ID: {results.track_ids}")
            if hasattr(results, 'boxes'):
                info.append(f"检测框数: {len(results.boxes) if results.boxes is not None else 0}")
        # 目标裁剪
        elif self.cn_name == "目标裁剪":
            if hasattr(results, 'crops'):
                crops = results.crops
                info.append(f"裁剪目标数: {len(crops) if crops is not None else 0}")
                if crops is not None and hasattr(crops, 'files'):
                    info.append(f"裁剪文件: {getattr(crops, 'files', None)}")
        # 目标模糊
        elif self.cn_name == "目标模糊":
            if hasattr(results, 'blurred_count'):
                info.append(f"模糊目标数: {results.blurred_count}")
            if hasattr(results, 'blurred_classes'):
                info.append(f"模糊类别: {results.blurred_classes}")
        # 锻炼监测
        elif self.cn_name == "锻炼监测":
            if hasattr(results, 'action_type'):
                info.append(f"动作类型: {results.action_type}")
            if hasattr(results, 'count'):
                info.append(f"锻炼次数: {results.count}")
            if hasattr(results, 'status'):
                info.append(f"锻炼状态: {results.status}")
        # 区域内目标计数
        elif self.cn_name == "区域内目标计数":
            if hasattr(results, 'region_counts'):
                info.append(f"各区域计数: {results.region_counts}")
            if hasattr(results, 'classwise_counts'):
                info.append(f"区域类别计数: {results.classwise_counts}")
        # 安全报警系统
        elif self.cn_name == "安全报警系统":
            if hasattr(results, 'alarm_status'):
                info.append(f"报警状态: {results.alarm_status}")
            if hasattr(results, 'alarm_classes'):
                info.append(f"报警目标类别: {results.alarm_classes}")
        # 热力图
        elif self.cn_name == "热力图":
            if hasattr(results, 'heatmap'):
                info.append(f"热力图shape: {getattr(results.heatmap, 'shape', None) if results.heatmap is not None else None}")
            if hasattr(results, 'stats'):
                info.append(f"热力图统计: {results.stats}")
        # 实例分割与目标跟踪
        elif self.cn_name == "实例分割与目标跟踪":
            if hasattr(results, 'masks'):
                info.append(f"分割目标数: {len(results.masks) if results.masks is not None else 0}")
            if hasattr(results, 'classes'):
                info.append(f"分割类别: {results.classes}")
            if hasattr(results, 'track_ids'):
                info.append(f"跟踪ID: {results.track_ids}")
        # VisionEye视图对象映射
        elif self.cn_name == "VisionEye视图对象映射":
            if hasattr(results, 'mapping'):
                info.append(f"对象映射: {results.mapping}")
            if hasattr(results, 'ids'):
                info.append(f"目标ID: {results.ids}")
        # 速度估计
        elif self.cn_name == "速度估计":
            if hasattr(results, 'speeds'):
                info.append(f"目标速度: {results.speeds}")
            if hasattr(results, 'speed_unit'):
                info.append(f"速度单位: {results.speed_unit}")
        # 距离计算
        elif self.cn_name == "距离计算":
            if hasattr(results, 'distances'):
                info.append(f"目标距离: {results.distances}")
            if hasattr(results, 'distance_unit'):
                info.append(f"距离单位: {results.distance_unit}")
        # 排队管理
        elif self.cn_name == "排队管理":
            if hasattr(results, 'queue_length'):
                info.append(f"队列长度: {results.queue_length}")
            if hasattr(results, 'queue_ids'):
                info.append(f"排队目标ID: {results.queue_ids}")
        # 停车管理
        elif self.cn_name == "停车管理":
            if hasattr(results, 'free_slots'):
                info.append(f"空余车位: {results.free_slots}")
            if hasattr(results, 'occupied_slots'):
                info.append(f"已停车辆: {results.occupied_slots}")
        # 分析
        elif self.cn_name == "分析":
            if hasattr(results, 'analysis'):
                info.append(f"分析结果: {results.analysis}")
            if hasattr(results, 'stats'):
                info.append(f"统计数据: {results.stats}")
        # 区域内目标跟踪
        elif self.cn_name == "区域内目标跟踪":
            if hasattr(results, 'zone_tracks'):
                info.append(f"区域跟踪目标: {results.zone_tracks}")
            if hasattr(results, 'track_ids'):
                info.append(f"跟踪ID: {results.track_ids}")
        # 相似性检索
        elif self.cn_name == "相似性检索":
            if hasattr(results, 'filenames') and hasattr(results, 'scores'):
                for fname, score in zip(results.filenames, results.scores):
                    info.append(f"{fname} | 相似度: {score:.4f}")
        # 通用字段（只在无详细信息时兜底显示）
        if not info:
            if hasattr(results, 'in_count'):
                info.append(f"in_count: {results.in_count}")
            if hasattr(results, 'out_count'):
                info.append(f"out_count: {results.out_count}")
            if hasattr(results, 'region_counts'):
                info.append(f"region_counts: {results.region_counts}")
            if hasattr(results, 'crops'):
                info.append(f"crops: {results.crops}")
        self.info_text.clear()
        self.info_text.append("\n".join([str(i) for i in info]) if info else "推理完成")

    def cvimg2qt(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

    def closeEvent(self, event):
        # 关闭窗口时自动停止推理和释放资源
        self.stop_infer()
        if self.cap:
            self.cap.release()
            self.cap = None
        event.accept()

    def on_search_clicked(self):
        # 相似性检索文本输入回调
        query = self.search_input.text().strip()
        if not query:
            self.info_text.append("请输入检索文本！")
            return
        if not hasattr(self, 'searcher'):
            from ultralytics import solutions
            self.searcher = solutions.VisualAISearch(device='cuda')
        self.info_text.append(f"正在检索：{query}")
        results = self.searcher(query)
        if hasattr(results, 'images') and len(results.images) > 0:
            # 显示第一张检索结果图片
            img = results.images[0]
            if img is not None:
                if img.shape[2] == 3:
                    img_bgr = img
                else:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                self.result_label.setPixmap(self.cvimg2qt(img_bgr).scaled(self.result_label.size(), Qt.KeepAspectRatio))
        # 显示所有检索结果文件名和分数
        info = []
        if hasattr(results, 'filenames') and hasattr(results, 'scores'):
            for fname, score in zip(results.filenames, results.scores):
                info.append(f"{fname} | 相似度: {score:.4f}")
        self.info_text.clear()
        self.info_text.append("\n".join(info) if info else "未检索到结果")

class SolutionGUI(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("解决方案功能集")
        self.setMinimumSize(400, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.sub_windows = {}
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        title = QLabel("解决方案功能")
        title.setFont(QFont("微软雅黑", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        for (cn,) in SOLUTION_FEATURES:
            btn = QPushButton(f"{cn}")
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setFont(QFont("微软雅黑", 11))
            btn.clicked.connect(lambda checked, cn=cn: self.open_subwindow(cn))
            layout.addWidget(btn)
        layout.addStretch(1)

    def open_subwindow(self, cn):
        key = f"{cn}"
        if key not in self.sub_windows:
            self.sub_windows[key] = SolutionSubWindow(cn, self)
        win = self.sub_windows[key]
        win.show()
        win.raise_()
        win.activateWindow()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SolutionGUI()
    win.show()
    sys.exit(app.exec_())
