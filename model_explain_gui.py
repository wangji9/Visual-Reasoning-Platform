import os
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QDialog, QLabel, QPushButton, QComboBox, QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QDoubleSpinBox, QMessageBox, QSizePolicy, QFrame, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
from model_explain import ModelExplainer
from detect import ObjectDetector
from seg import Segmentor
from pos import PoseEstimator

def cvimg2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QPixmap.fromImage(QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888))

class ModelExplainWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型可解释性分析")
        self.setMinimumSize(900, 600)
        self.model = None
        self.explainer = None
        self.input_img = None
        self.layer_names = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        # 顶部工具栏
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("输入方式："))
        self.input_combo = QComboBox()
        self.input_combo.addItems(["图片", "视频", "摄像头"])
        self.input_combo.currentTextChanged.connect(self.on_input_type_changed)
        toolbar.addWidget(self.input_combo)
        self.btn_upload = QPushButton("上传/打开")
        self.btn_upload.clicked.connect(self.upload_input)
        toolbar.addWidget(self.btn_upload)
        toolbar.addWidget(QLabel("模型类型："))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["", "目标检测", "图像分割", "姿态估计"])
        self.model_combo.currentTextChanged.connect(self.on_model_type_changed)
        toolbar.addWidget(self.model_combo)
        self.btn_weight = QPushButton("上传权重")
        self.btn_weight.clicked.connect(self.select_weight)
        toolbar.addWidget(self.btn_weight)
        self.weight_label = QLabel("未选择")
        toolbar.addWidget(self.weight_label)
        toolbar.addWidget(QLabel("层级："))
        self.layer_combo = QComboBox()
        toolbar.addWidget(self.layer_combo)
        self.btn_refresh_layers = QPushButton("刷新层级")
        self.btn_refresh_layers.clicked.connect(self.refresh_layers)
        toolbar.addWidget(self.btn_refresh_layers)
        self.btn_vis = QPushButton("可视化结果")
        self.btn_vis.clicked.connect(self.visualize)
        toolbar.addWidget(self.btn_vis)
        self.btn_vis_all = QPushButton("全部层级可视化")
        self.btn_vis_all.clicked.connect(self.show_all_layers_vis)
        toolbar.addWidget(self.btn_vis_all)
        self.btn_save_all = QPushButton("保存全部层级可视化")
        self.btn_save_all.clicked.connect(self.save_all_layers_vis)
        toolbar.addWidget(self.btn_save_all)
        self.btn_vis_channel = QPushButton("显示通道热图")
        self.btn_vis_channel.clicked.connect(self.show_channel_heatmaps)
        toolbar.addWidget(self.btn_vis_channel)
        self.btn_save_channel = QPushButton("保存当前层所有通道热图")
        self.btn_save_channel.clicked.connect(self.save_channel_heatmaps)
        toolbar.addWidget(self.btn_save_channel)
        layout.addLayout(toolbar)
        # 主体区域
        body = QHBoxLayout()
        # 左侧原始输入
        left = QVBoxLayout()
        input_title = QLabel("原始输入区")
        input_title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        input_title.setAlignment(Qt.AlignCenter)
        left.addWidget(input_title)
        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFrameShape(QFrame.Box)
        self.input_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.input_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.input_label)
        body.addLayout(left, 1)
        # 右侧热图
        right = QVBoxLayout()
        vis_title = QLabel("热图可视化区")
        vis_title.setFont(QFont("微软雅黑", 10, QFont.Bold))
        vis_title.setAlignment(Qt.AlignCenter)
        right.addWidget(vis_title)
        self.vis_label = QLabel()
        self.vis_label.setAlignment(Qt.AlignCenter)
        self.vis_label.setFrameShape(QFrame.Box)
        self.vis_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.vis_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.vis_label)
        body.addLayout(right, 1)
        layout.addLayout(body, 5)
        # 下方信息
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)

    def on_input_type_changed(self, text):
        self.input_label.clear()
        self.vis_label.clear()
        self.input_img = None

    def upload_input(self):
        t = self.input_combo.currentText()
        if t == "图片":
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png *.bmp)")
            if path:
                img = cv2.imread(path)
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.info_text.append(f"已选择图片: {path}")
        elif t == "视频":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov)")
            if path:
                cap = cv2.VideoCapture(path)
                ret, img = cap.read()
                if ret:
                    self.input_img = img
                    self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                cap.release()
                self.info_text.append(f"已选择视频: {path}")
        elif t == "摄像头":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                QMessageBox.critical(self, "错误", "摄像头无法打开")
                return
            ret, img = cap.read()
            if ret:
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.info_text.append("已打开摄像头")
            cap.release()

    def on_model_type_changed(self, text):
        self.model = None
        self.explainer = None
        self.layer_combo.clear()
        self.weight_label.setText("未选择")

    def select_weight(self):
        t = self.model_combo.currentText()
        if t not in ["目标检测", "图像分割", "姿态估计"]:
            QMessageBox.information(self, "提示", "请选择正确的模型类型")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "", "PyTorch Weights (*.pt *.pth)")
        if path:
            try:
                if t == "目标检测":
                    # ultralytics YOLO对象的 .model 才是真正的torch模型
                    self.model = ObjectDetector(path).model.model
                elif t == "图像分割":
                    self.model = Segmentor(path).model.model
                elif t == "姿态估计":
                    self.model = PoseEstimator(path).model.model
                self.explainer = ModelExplainer(self.model)
                self.weight_label.setText(os.path.basename(path))
                self.info_text.append("模型加载成功")
                self.refresh_layers()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"模型加载失败: {e}")
                self.model = None
                self.explainer = None

    def refresh_layers(self):
        if self.explainer is None:
            QMessageBox.information(self, "提示", "请先加载模型")
            return
        self.layer_names = self.explainer.get_all_layer_names()
        self.layer_combo.clear()
        self.layer_combo.addItems(self.layer_names)
        self.info_text.append(f"共{len(self.layer_names)}个可选层级")

    def visualize(self):
        if self.input_img is None:
            QMessageBox.information(self, "提示", "请先上传输入数据")
            return
        if self.explainer is None:
            QMessageBox.information(self, "提示", "请先加载模型")
            return
        layer = self.layer_combo.currentText()
        # 转为RGB并归一化
        img_rgb = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB)
        fmap = self.explainer.get_feature_map(img_rgb, layer)
        heatmap = self.explainer.featuremap_to_heatmap(fmap)
        if heatmap is not None:
            h, w = self.input_img.shape[:2]
            heatmap = cv2.resize(heatmap, (w, h))
            overlay = cv2.addWeighted(self.input_img, 0.5, heatmap, 0.5, 0)
            self.vis_label.setPixmap(cvimg2qt(overlay).scaled(self.vis_label.size(), Qt.KeepAspectRatio))
            self.info_text.append(f"已可视化层级: {layer}")
        else:
            QMessageBox.warning(self, "可视化失败", "未获取到特征图")

    def show_all_layers_vis(self):
        if self.input_img is None or self.explainer is None:
            QMessageBox.information(self, "提示", "请先上传输入数据并加载模型")
            return
        img_rgb = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB)
        layer_imgs = []
        for layer in self.layer_names:
            fmap = self.explainer.get_feature_map(img_rgb, layer)
            heatmap = self.explainer.featuremap_to_heatmap(fmap)
            if heatmap is not None:
                h, w = self.input_img.shape[:2]
                heatmap = cv2.resize(heatmap, (w, h))
                overlay = cv2.addWeighted(self.input_img, 0.5, heatmap, 0.5, 0)
                layer_imgs.append((layer, overlay.copy()))
        if not layer_imgs:
            QMessageBox.warning(self, "可视化失败", "未获取到任何特征图")
            return
        dlg = AllLayersVisDialog(layer_imgs, self)
        dlg.exec_()

    def save_all_layers_vis(self):
        if self.input_img is None or self.explainer is None:
            QMessageBox.information(self, "提示", "请先上传输入数据并加载模型")
            return
        img_rgb = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB)
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not save_dir:
            return
        saved = 0
        for layer in self.layer_names:
            fmap = self.explainer.get_feature_map(img_rgb, layer)
            heatmap = self.explainer.featuremap_to_heatmap(fmap)
            if heatmap is not None:
                h, w = self.input_img.shape[:2]
                heatmap = cv2.resize(heatmap, (w, h))
                overlay = cv2.addWeighted(self.input_img, 0.5, heatmap, 0.5, 0)
                # 文件名合法化
                fname = layer.replace('/', '_').replace('.', '_')
                out_path = os.path.join(save_dir, f"{fname}.jpg")
                cv2.imwrite(out_path, overlay)
                saved += 1
        QMessageBox.information(self, "保存完成", f"共保存{saved}个层级的可视化图像。")

    def show_channel_heatmaps(self):
        if self.input_img is None or self.explainer is None:
            QMessageBox.information(self, "提示", "请先上传输入数据并加载模型")
            return
        layer = self.layer_combo.currentText()
        if not layer:
            QMessageBox.information(self, "提示", "请先选择层级")
            return
        img_rgb = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB)
        fmap = self.explainer.get_feature_map(img_rgb, layer)
        heatmaps = self.explainer.featuremap_all_channels_to_heatmaps(fmap)
        if not heatmaps:
            QMessageBox.warning(self, "可视化失败", "该层无可用通道热图")
            return
        # 弹窗分页浏览
        dlg = ChannelHeatmapsDialog(heatmaps, self)
        dlg.exec_()

    def save_channel_heatmaps(self):
        if self.input_img is None or self.explainer is None:
            QMessageBox.information(self, "提示", "请先上传输入数据并加载模型")
            return
        layer = self.layer_combo.currentText()
        if not layer:
            QMessageBox.information(self, "提示", "请先选择层级")
            return
        img_rgb = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB)
        fmap = self.explainer.get_feature_map(img_rgb, layer)
        heatmaps = self.explainer.featuremap_all_channels_to_heatmaps(fmap)
        if not heatmaps:
            QMessageBox.warning(self, "保存失败", "该层无可用通道热图")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not save_dir:
            return
        for idx, heatmap in enumerate(heatmaps):
            out_path = os.path.join(save_dir, f"{layer.replace('/', '_').replace('.', '_')}_ch{idx}.jpg")
            cv2.imwrite(out_path, heatmap)
        QMessageBox.information(self, "保存完成", f"共保存{len(heatmaps)}个通道的热力图。")

class AllLayersVisDialog(QDialog):
    def __init__(self, layer_imgs, parent=None, page_size=16):
        super().__init__(parent)
        self.setWindowTitle("全部层级可视化")
        self.setMinimumSize(1600, 900)
        self.layer_imgs = layer_imgs  # [(layer_name, img)]
        self.page_size = page_size
        self.page = 0
        self.total_pages = (len(layer_imgs) + page_size - 1) // page_size
        self.init_ui()
        self.show_page()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.img_layout = QGridLayout()
        self.layout.addLayout(self.img_layout)
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一页")
        self.btn_prev.clicked.connect(self.prev_page)
        nav_layout.addWidget(self.btn_prev)
        self.page_label = QLabel()
        nav_layout.addWidget(self.page_label)
        self.btn_next = QPushButton("下一页")
        self.btn_next.clicked.connect(self.next_page)
        nav_layout.addWidget(self.btn_next)
        self.layout.addLayout(nav_layout)

    def show_page(self):
        # 清空旧内容
        for i in reversed(range(self.img_layout.count())):
            item = self.img_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        start = self.page * self.page_size
        end = min(start + self.page_size, len(self.layer_imgs))
        # 4x4 网格
        for idx, (layer_name, img) in enumerate(self.layer_imgs[start:end]):
            row = idx // 4
            col = idx % 4
            vbox = QVBoxLayout()
            label_img = QLabel()
            label_img.setPixmap(cvimg2qt(img).scaled(200, 200, Qt.KeepAspectRatio))
            label_img.setAlignment(Qt.AlignCenter)
            vbox.addWidget(label_img)
            label_name = QLabel(layer_name)
            label_name.setAlignment(Qt.AlignCenter)
            vbox.addWidget(label_name)
            frame = QWidget()
            frame.setLayout(vbox)
            self.img_layout.addWidget(frame, row, col)
        self.page_label.setText(f"第{self.page+1}/{self.total_pages}页")
        self.btn_prev.setEnabled(self.page > 0)
        self.btn_next.setEnabled(self.page < self.total_pages - 1)

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.show_page()

    def next_page(self):
        if self.page < self.total_pages - 1:
            self.page += 1
            self.show_page()

class ChannelHeatmapsDialog(QDialog):
    def __init__(self, heatmaps, parent=None, page_size=16):
        super().__init__(parent)
        self.setWindowTitle("通道热图可视化")
        self.setMinimumSize(1600, 900)
        self.heatmaps = heatmaps  # list of np.ndarray
        self.page_size = page_size
        self.page = 0
        self.total_pages = (len(heatmaps) + page_size - 1) // page_size
        self.init_ui()
        self.show_page()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.img_layout = QGridLayout()
        self.layout.addLayout(self.img_layout)
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一页")
        self.btn_prev.clicked.connect(self.prev_page)
        nav_layout.addWidget(self.btn_prev)
        self.page_label = QLabel()
        nav_layout.addWidget(self.page_label)
        self.btn_next = QPushButton("下一页")
        self.btn_next.clicked.connect(self.next_page)
        nav_layout.addWidget(self.btn_next)
        self.layout.addLayout(nav_layout)

    def show_page(self):
        # 清空旧内容
        for i in reversed(range(self.img_layout.count())):
            item = self.img_layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)
        start = self.page * self.page_size
        end = min(start + self.page_size, len(self.heatmaps))
        for idx, heatmap in enumerate(self.heatmaps[start:end]):
            row = idx // 4
            col = idx % 4
            vbox = QVBoxLayout()
            label_img = QLabel()
            label_img.setPixmap(cvimg2qt(heatmap).scaled(200, 200, Qt.KeepAspectRatio))
            label_img.setAlignment(Qt.AlignCenter)
            vbox.addWidget(label_img)
            label_name = QLabel(f"通道 {start + idx}")
            label_name.setAlignment(Qt.AlignCenter)
            vbox.addWidget(label_name)
            frame = QWidget()
            frame.setLayout(vbox)
            self.img_layout.addWidget(frame, row, col)
        self.page_label.setText(f"第{self.page+1}/{self.total_pages}页")
        self.btn_prev.setEnabled(self.page > 0)
        self.btn_next.setEnabled(self.page < self.total_pages - 1)

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.show_page()

    def next_page(self):
        if self.page < self.total_pages - 1:
            self.page += 1
            self.show_page()
