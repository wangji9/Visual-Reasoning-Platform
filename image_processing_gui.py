# filepath: e:\pyqt\image_processing_gui.py
import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 避免与Qt冲突
from image_processing import (
    to_gray, blur, canny, invert, threshold, sharpen, median_blur, bilateral_filter, emboss, sobel_edge, laplacian_edge, cartoon, flip, power_law, log_transform, hist_equalize, contrast_stretch, iterative_threshold, otsu_threshold, edge_guided_threshold, adaptive_threshold, region_growing, find_contours, pca_compress, erode, dilate, morph_open, morph_close, morph_gradient, morph_tophat, morph_blackhat, find_all_contours, draw_contour, approx_poly_contour, bounding_rect, min_area_rect, min_enclosing_circle, histogram_moments, glcm_features, region_properties
)
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QComboBox, QFileDialog, QHBoxLayout, QVBoxLayout, QApplication, QFrame, QSizePolicy,
    QSlider, QSpinBox, QLineEdit, QMessageBox, QGridLayout, QDialog
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class StatisticsDialog(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("统计信息")
        self.setMinimumSize(520, 380)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("background:#fff;border:1.5px solid #888;border-radius:8px;")
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5,3))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color:#333;font-size:13px;")
        self.layout.addWidget(self.info_label)
        self.setLayout(self.layout)
        self._last_func = None
        self._last_img = None
        self._last_data = None
        self._last_title = None
        self.setWindowFlags(self.windowFlags() | Qt.Window)

    def set_data(self, func, img, stat_funcs):
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        matplotlib.rcParams['axes.unicode_minus'] = False
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.info_label.clear()
        if img is None:
            ax.text(0.5, 0.5, "无图像", ha='center', va='center', fontsize=16)
            self.canvas.draw()
            return
        # 统计内容
        # 移除所有 plt 相关代码，全部用 self.figure/self.canvas 绘图
        # 1. 灰度/二值化直方图
        if func in ["灰度", "高斯模糊", "锐化", "中值模糊", "双边滤波", "浮雕", "卡通化", "反色", "图像翻转", "幂运算", "对数运算", "直方图均衡化", "对比度拉伸"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            hist = cv2.calcHist([gray], [0], None, [256], [0,256]).flatten()
            ax.plot(hist, color='black')
            ax.set_title(f"{func} - 灰度直方图")
            ax.set_xlabel("灰度值")
            ax.set_ylabel("像素数")
            ax.set_xlim([0,256])
            ax.grid(True, linestyle='--', alpha=0.5)
        elif func in ["二值化", "OTSU阈值分割", "迭代法阈值分割", "自适应阈值", "区域增长分割", "边缘引导阈值"]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            hist = cv2.calcHist([binary], [0], None, [256], [0,256]).flatten()
            ax.plot(hist, color='black')
            ax.set_title(f"{func} - 二值化直方图")
            ax.set_xlabel("灰度值")
            ax.set_ylabel("像素数")
            ax.set_xlim([0,256])
            ax.grid(True, linestyle='--', alpha=0.5)
        # 2. 统计矩
        elif func == "直方图统计矩":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            feats = stat_funcs['histogram_moments'](gray)
            ax.bar(list(feats.keys()), list(feats.values()), color='#4e79a7', edgecolor='black')
            ax.set_title("直方图统计矩")
            ax.set_xticklabels(list(feats.keys()), rotation=30, fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            self.info_label.setText("\n".join([f"{k}: {v:.2f}" for k,v in feats.items()]))
        # 3. 共生矩阵纹理
        elif func == "共生矩阵纹理":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            feats = stat_funcs['glcm_features'](gray)
            ax.bar(list(feats.keys()), list(feats.values()), color='#4e79a7', edgecolor='black')
            ax.set_title("共生矩阵纹理特征")
            ax.set_xticklabels(list(feats.keys()), rotation=30, fontsize=11)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            self.info_label.setText("\n".join([f"{k}: {v:.4f}" for k,v in feats.items()]))
        # 4. 区域属性
        elif func == "区域统计属性":
            _, contours = stat_funcs['find_contours'](img)
            if contours:
                feats = stat_funcs['region_properties'](contours[0])
                feats = {k: v for k, v in feats.items() if isinstance(v, (float, int))}
                ax.bar(list(feats.keys()), list(feats.values()), color='#4e79a7', edgecolor='black')
                ax.set_title("区域统计属性")
                ax.set_xticklabels(list(feats.keys()), rotation=30, fontsize=11)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                self.info_label.setText("\n".join([f"{k}: {v:.2f}" for k,v in feats.items()]))
            else:
                ax.text(0.5, 0.5, "未检测到目标区域", ha='center', va='center', fontsize=14)
        # 其它
        else:
            ax.text(0.5, 0.5, "当前功能暂不支持统计图表展示", ha='center', va='center', fontsize=14)
        self.figure.tight_layout()
        self.canvas.draw()

class ImageProcessingWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像处理工具")
        self.setMinimumSize(1400, 850)  # 启动窗口更大
        self.resize(1400, 850)
        self.input_img = None
        self.result_img = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.video_mode = None  # "video" or "camera"

        # 先初始化功能名和当前功能，避免属性未定义
        self.func_names = [
            "灰度", "高斯模糊", "Canny边缘", "反色", "二值化", "锐化",
            "中值模糊", "双边滤波", "浮雕", "Sobel边缘", "Laplacian边缘", "卡通化",
            "图像翻转", "幂运算", "对数运算", "直方图均衡化", "对比度拉伸",
            "迭代法阈值分割", "OTSU阈值分割", "边缘引导阈值", "自适应阈值", "区域增长分割",
            "提取目标边界", "目标特征", "边界统计矩", "区域描绘子", "直方图统计矩", "共生矩阵纹理", "Hu不变矩", "PCA压缩", "区域统计属性",
            "腐蚀", "膨胀", "开运算", "闭运算", "形态学梯度", "顶帽", "黑帽",
            "查找所有轮廓", "绘制轮廓", "多边形逼近", "外接矩形", "最小外接矩形", "最小外接圆"
        ]
        self.selected_func = self.func_names[0]
        self.stats_dialog = None

        main_layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)
        toolbar.addWidget(self.btn_open)

        self.btn_video = QPushButton("打开视频")
        self.btn_video.clicked.connect(self.open_video)
        toolbar.addWidget(self.btn_video)

        self.btn_camera = QPushButton("打开摄像头")
        self.btn_camera.clicked.connect(self.open_camera)
        toolbar.addWidget(self.btn_camera)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_video_camera)
        toolbar.addWidget(self.btn_stop)

        # 处理功能区：每个功能一个按钮，自动换行美观排布
        func_select_label = QLabel("功能选择：")
        func_select_label.setFont(QFont("微软雅黑", 15, QFont.Bold))
        func_select_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(func_select_label)
        self.func_buttons = []
        self.func_btn_widget = QWidget()
        self.func_btn_layout = QGridLayout()
        self.func_btn_layout.setSpacing(2)  # 行间距更小
        btns_per_row = 12  # 每行按钮数更多，按钮更小更紧凑
        for idx, name in enumerate(self.func_names):
            btn = QPushButton(name)
            btn.setFont(QFont("微软雅黑", 12))  # 字体更小
            btn.setStyleSheet(
                "font-size:12px; padding:2px 4px; margin:2px; min-width:56px; max-width:80px; min-height:22px; max-height:28px; border-radius:5px; border: 1.2px solid #888; text-align:center;"
            )
            btn.clicked.connect(self.on_func_btn_clicked)
            row = idx // btns_per_row
            col = idx % btns_per_row
            self.func_btn_layout.addWidget(btn, row, col, alignment=Qt.AlignCenter)
            self.func_buttons.append(btn)
        self.func_btn_widget.setLayout(self.func_btn_layout)
        main_layout.addWidget(self.func_btn_widget)
        self.func_btn_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.param_widgets = {}  # ← 提前初始化

        # 参数控件区
        self.param_layout = QHBoxLayout()
        toolbar.addLayout(self.param_layout)
        self.update_param_widgets()  # 初始化参数控件

        self.btn_process = QPushButton("执行处理")
        self.btn_process.clicked.connect(self.process_image)
        toolbar.addWidget(self.btn_process)

        # 统计信息按钮替换原有统计信息区
        self.btn_stats = QPushButton("统计信息")
        self.btn_stats.setFont(QFont("微软雅黑", 13))
        self.btn_stats.setStyleSheet("padding:4px 12px;margin-left:8px;")
        self.btn_stats.clicked.connect(self.show_statistics_window)
        toolbar.addWidget(self.btn_stats)

        # 新增保存结果按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_save.setFont(QFont("微软雅黑", 13))
        self.btn_save.setStyleSheet("padding:4px 12px;margin-left:8px;")
        self.btn_save.clicked.connect(self.save_result)
        toolbar.addWidget(self.btn_save)

        main_layout.addLayout(toolbar)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(8)

        # 原始输入区
        left_layout = QVBoxLayout()
        input_title = QLabel("原始输入")
        input_title.setFont(QFont("微软雅黑", 16, QFont.Bold))
        input_title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(input_title)

        self.input_label = QLabel()
        self.input_label.setAlignment(Qt.AlignCenter)
        self.input_label.setFrameShape(QFrame.Box)
        self.input_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.input_label.setMinimumSize(600, 480)  # 图像区更大
        self.input_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.input_label)

        body_layout.addLayout(left_layout, 1)

        # 处理结果区
        right_layout = QVBoxLayout()
        result_title = QLabel("处理结果")
        result_title.setFont(QFont("微软雅黑", 16, QFont.Bold))
        result_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(result_title)

        self.result_label = QLabel()
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFrameShape(QFrame.Box)
        self.result_label.setStyleSheet("background-color: #f7f7f7; border: 1px solid #bbb;")
        self.result_label.setMinimumSize(600, 480)  # 图像区更大
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.result_label)

        body_layout.addLayout(right_layout, 1)

        main_layout.addLayout(body_layout, 5)

        self.selected_func = self.func_names[0]
        self.update_param_widgets()

    def on_func_btn_clicked(self):
        sender = self.sender()
        if sender:
            self.selected_func = sender.text()
            self.update_param_widgets()
            self.process_image()

    def update_param_widgets(self):
        # 清空旧控件
        for i in reversed(range(self.param_layout.count())):
            w = self.param_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.param_widgets.clear()
        func = self.selected_func
        # 针对不同功能添加参数控件
        if func == "高斯模糊":
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(1); slider.setMaximum(31); slider.setValue(5); slider.setSingleStep(2)
            slider.setTickInterval(2)
            self.param_layout.addWidget(QLabel("ksize:"))
            self.param_layout.addWidget(slider)
            self.param_widgets["ksize"] = slider
        elif func == "Canny边缘":
            t1 = QSpinBox(); t1.setRange(0,255); t1.setValue(100)
            t2 = QSpinBox(); t2.setRange(0,255); t2.setValue(200)
            self.param_layout.addWidget(QLabel("阈值1:"))
            self.param_layout.addWidget(t1)
            self.param_layout.addWidget(QLabel("阈值2:"))
            self.param_layout.addWidget(t2)
            self.param_widgets["threshold1"] = t1
            self.param_widgets["threshold2"] = t2
        elif func == "Sobel边缘":
            ksize = QSpinBox(); ksize.setRange(1, 7); ksize.setSingleStep(2); ksize.setValue(3)
            self.param_layout.addWidget(QLabel("ksize:"))
            self.param_layout.addWidget(ksize)
            self.param_widgets["ksize"] = ksize
        elif func == "Laplacian边缘":
            ksize = QSpinBox(); ksize.setRange(1, 7); ksize.setSingleStep(2); ksize.setValue(3)
            self.param_layout.addWidget(QLabel("ksize:"))
            self.param_layout.addWidget(ksize)
            self.param_widgets["ksize"] = ksize
        elif func == "卡通化":
            b = QSpinBox(); b.setRange(1, 20); b.setValue(9)
            self.param_layout.addWidget(QLabel("双边滤波d:"))
            self.param_layout.addWidget(b)
            self.param_widgets["bilateral_d"] = b
        elif func == "二值化":
            t = QSpinBox(); t.setRange(0,255); t.setValue(127)
            self.param_layout.addWidget(QLabel("阈值:"))
            self.param_layout.addWidget(t)
            self.param_widgets["thresh"] = t
        elif func == "中值模糊":
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(1); slider.setMaximum(31); slider.setValue(5); slider.setSingleStep(2)
            slider.setTickInterval(2)
            self.param_layout.addWidget(QLabel("ksize:"))
            self.param_layout.addWidget(slider)
            self.param_widgets["ksize"] = slider
        elif func == "双边滤波":
            d = QSpinBox(); d.setRange(1,20); d.setValue(9)
            sigmaColor = QSpinBox(); sigmaColor.setRange(1,200); sigmaColor.setValue(75)
            sigmaSpace = QSpinBox(); sigmaSpace.setRange(1,200); sigmaSpace.setValue(75)
            self.param_layout.addWidget(QLabel("d:"))
            self.param_layout.addWidget(d)
            self.param_layout.addWidget(QLabel("sigmaColor:"))
            self.param_layout.addWidget(sigmaColor)
            self.param_layout.addWidget(QLabel("sigmaSpace:"))
            self.param_layout.addWidget(sigmaSpace)
            self.param_widgets["d"] = d
            self.param_widgets["sigmaColor"] = sigmaColor
            self.param_widgets["sigmaSpace"] = sigmaSpace
        elif func == "图像翻转":
            mode = QComboBox()
            mode.addItems(["垂直", "水平", "水平+垂直"])
            self.param_layout.addWidget(QLabel("模式:"))
            self.param_layout.addWidget(mode)
            self.param_widgets["mode"] = mode
        elif func == "幂运算":
            gamma = QLineEdit("1.0")
            self.param_layout.addWidget(QLabel("gamma:"))
            self.param_layout.addWidget(gamma)
            self.param_widgets["gamma"] = gamma
        elif func == "自适应阈值":
            block = QSpinBox(); block.setRange(3,99); block.setSingleStep(2); block.setValue(11)
            c = QSpinBox(); c.setRange(-20,20); c.setValue(2)
            self.param_layout.addWidget(QLabel("blockSize:"))
            self.param_layout.addWidget(block)
            self.param_layout.addWidget(QLabel("C:"))
            self.param_layout.addWidget(c)
            self.param_widgets["blockSize"] = block
            self.param_widgets["C"] = c
        elif func == "区域增长分割":
            thresh = QSpinBox(); thresh.setRange(1,50); thresh.setValue(5)
            self.param_layout.addWidget(QLabel("阈值:"))
            self.param_layout.addWidget(thresh)
            self.param_widgets["thresh"] = thresh
        elif func == "PCA压缩":
            n = QSpinBox(); n.setRange(1,100); n.setValue(20)
            self.param_layout.addWidget(QLabel("主成分数:"))
            self.param_layout.addWidget(n)
            self.param_widgets["num_components"] = n
        elif func in ["腐蚀", "膨胀", "开运算", "闭运算", "形态学梯度", "顶帽", "黑帽"]:
            ksize = QSpinBox(); ksize.setRange(1, 21); ksize.setValue(3); ksize.setSingleStep(2)
            self.param_layout.addWidget(QLabel("ksize:"))
            self.param_layout.addWidget(ksize)
            self.param_widgets["ksize"] = ksize
        elif func in ["查找所有轮廓", "绘制轮廓", "多边形逼近", "外接矩形", "最小外接矩形", "最小外接圆"]:
            pass  # 这些功能不需要额外参数

    def show_statistics_window(self):
        if self.input_img is None:
            QMessageBox.information(self, "提示", "请先加载图片！")
            return
        if self.stats_dialog is None:
            self.stats_dialog = StatisticsDialog(self)
        # 统计函数字典传递
        stat_funcs = {
            'histogram_moments': histogram_moments,
            'glcm_features': glcm_features,
            'find_contours': find_contours,
            'region_properties': region_properties
        }
        self.stats_dialog.set_data(self.selected_func, self.input_img, stat_funcs)
        self.stats_dialog.show()
        self.stats_dialog.raise_()
        self.stats_dialog.activateWindow()

    def save_result(self):
        # 保存静态图片
        if self.input_img is None or self.result_img is None:
            QMessageBox.information(self, "提示", "没有可保存的结果！")
            return
        func = self.selected_func
        if self.cap is None:
            # 静态图片
            base = f"{func}_result"
            idx = 1
            while os.path.exists(f"{base}{idx}.png"):
                idx += 1
            fname = f"{base}{idx}.png"
            cv2.imencode('.png', self.result_img)[1].tofile(fname)
            QMessageBox.information(self, "保存成功", f"结果已保存为 {fname}")
        else:
            # 视频/摄像头，保存为视频
            if not hasattr(self, '_video_writer') or self._video_writer is None:
                h, w = self.result_img.shape[:2]
                base = f"{func}_result"
                idx = 1
                while os.path.exists(f"{base}{idx}.avi"):
                    idx += 1
                fname = f"{base}{idx}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self._video_writer = cv2.VideoWriter(fname, fourcc, 20.0, (w, h))
                self._video_save_name = fname
                self._video_save_count = 0
            self._video_writer.write(self.result_img)
            self._video_save_count += 1
            # 若视频播放结束或停止，自动释放
            if not self.timer.isActive():
                self._video_writer.release()
                self._video_writer = None
                QMessageBox.information(self, "保存成功", f"视频已保存为 {self._video_save_name}，共{self._video_save_count}帧")

    def next_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video_camera()
            return
        self.input_img = frame
        self.input_label.setPixmap(cvimg2qt(frame).scaled(self.input_label.size(), Qt.KeepAspectRatio))
        # 实时处理
        func = self.selected_func
        img = frame.copy()
        p = self.param_widgets
        try:
            if func == "灰度":
                res = to_gray(img)
            elif func == "高斯模糊":
                k = p["ksize"].value() | 1
                res = blur(img, k)
            elif func == "Canny边缘":
                res = canny(img, p["threshold1"].value(), p["threshold2"].value())
            elif func == "反色":
                res = invert(img)
            elif func == "二值化":
                res = threshold(img, p["thresh"].value())
            elif func == "锐化":
                res = sharpen(img)
            elif func == "中值模糊":
                k = p["ksize"].value() | 1
                res = median_blur(img, k)
            elif func == "双边滤波":
                res = bilateral_filter(img, p["d"].value(), p["sigmaColor"].value(), p["sigmaSpace"].value())
            elif func == "浮雕":
                res = emboss(img)
            elif func == "Sobel边缘":
                res = sobel_edge(img)
            elif func == "Laplacian边缘":
                res = laplacian_edge(img)
            elif func == "卡通化":
                res = cartoon(img)
            elif func == "图像翻转":
                mode = {0:0, 1:1, 2:-1}[p["mode"].currentIndex()]
                res = flip(img, mode)
            elif func == "幂运算":
                try:
                    gamma = float(p["gamma"].text())
                except:
                    gamma = 1.0
                res = power_law(img, gamma)
            elif func == "对数运算":
                res = log_transform(img)
            elif func == "直方图均衡化":
                res = hist_equalize(img)
            elif func == "对比度拉伸":
                res = contrast_stretch(img)
            elif func == "迭代法阈值分割":
                res = iterative_threshold(img)
            elif func == "OTSU阈值分割":
                res = otsu_threshold(img)
            elif func == "边缘引导阈值":
                res = edge_guided_threshold(img)
            elif func == "自适应阈值":
                res = adaptive_threshold(img, p["blockSize"].value(), p["C"].value())
            elif func == "区域增长分割":
                res = region_growing(img, thresh=p["thresh"].value())
            elif func == "提取目标边界":
                res, _ = find_contours(img)
            elif func == "PCA压缩":
                res = pca_compress(img, p["num_components"].value())
            elif func == "腐蚀":
                res = erode(img, ksize=3)
            elif func == "膨胀":
                res = dilate(img, ksize=3)
            elif func == "开运算":
                res = morph_open(img, ksize=3)
            elif func == "闭运算":
                res = morph_close(img, ksize=3)
            elif func == "形态学梯度":
                res = morph_gradient(img, ksize=3)
            elif func == "顶帽":
                res = morph_tophat(img, ksize=3)
            elif func == "黑帽":
                res = morph_blackhat(img, ksize=3)
            elif func == "查找所有轮廓":
                contours, hierarchy = find_all_contours(img)
                res = img.copy()
                cv2.drawContours(res, contours, -1, (0, 255, 0), 2)
            elif func in ["绘制轮廓", "多边形逼近", "外接矩形", "最小外接矩形", "最小外接圆"]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                img_for_contour = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                contours, _ = find_all_contours(img_for_contour)
                if not contours:
                    QMessageBox.information(self, "提示", "未检测到轮廓，请尝试先进行二值化或调整图像对比度！")
                    res = img
                else:
                    if func == "绘制轮廓":
                        res = draw_contour(img, contours[0], color=(255,0,0), thickness=2)
                    elif func == "多边形逼近":
                        approx = approx_poly_contour(contours[0])
                        res = img.copy()
                        cv2.polylines(res, [approx], True, (0,0,255), 2)
                    elif func == "外接矩形":
                        x, y, w, h = bounding_rect(contours[0])
                        res = img.copy()
                        cv2.rectangle(res, (x, y), (x+w, y+h), (0,255,255), 2)
                    elif func == "最小外接矩形":
                        box, rect = min_area_rect(contours[0])
                        res = img.copy()
                        cv2.drawContours(res, [box], 0, (255,0,255), 2)
                    elif func == "最小外接圆":
                        center, radius = min_enclosing_circle(contours[0])
                        res = img.copy()
                        cv2.circle(res, center, radius, (0,128,255), 2)
            else:
                res = img
        except Exception:
            res = img
        self.result_img = res
        self.result_label.setPixmap(cvimg2qt(res).scaled(self.result_label.size(), Qt.KeepAspectRatio))
        # 实时刷新统计弹窗
        if self.stats_dialog and self.stats_dialog.isVisible():
            stat_funcs = {
                'histogram_moments': histogram_moments,
                'glcm_features': glcm_features,
                'find_contours': find_contours,
                'region_properties': region_properties
            }
            self.stats_dialog.set_data(self.selected_func, frame, stat_funcs)

    def process_image(self):
        if self.input_img is None:
            QMessageBox.information(self, "提示", "请先加载图片！")
            return
        img = self.input_img.copy()
        func = self.selected_func
        p = self.param_widgets
        try:
            if func == "灰度":
                res = to_gray(img)
            elif func == "高斯模糊":
                k = p["ksize"].value() | 1
                res = blur(img, k)
            elif func == "Canny边缘":
                res = canny(img, p["threshold1"].value(), p["threshold2"].value())
            elif func == "反色":
                res = invert(img)
            elif func == "二值化":
                res = threshold(img, p["thresh"].value())
            elif func == "锐化":
                res = sharpen(img)
            elif func == "中值模糊":
                k = p["ksize"].value() | 1
                res = median_blur(img, k)
            elif func == "双边滤波":
                res = bilateral_filter(img, p["d"].value(), p["sigmaColor"].value(), p["sigmaSpace"].value())
            elif func == "浮雕":
                res = emboss(img)
            elif func == "Sobel边缘":
                res = sobel_edge(img)
            elif func == "Laplacian边缘":
                res = laplacian_edge(img)
            elif func == "卡通化":
                res = cartoon(img)
            elif func == "图像翻转":
                mode = {0:0, 1:1, 2:-1}[p["mode"].currentIndex()]
                res = flip(img, mode)
            elif func == "幂运算":
                try:
                    gamma = float(p["gamma"].text())
                except:
                    gamma = 1.0
                res = power_law(img, gamma)
            elif func == "对数运算":
                res = log_transform(img)
            elif func == "直方图均衡化":
                res = hist_equalize(img)
            elif func == "对比度拉伸":
                res = contrast_stretch(img)
            elif func == "迭代法阈值分割":
                res = iterative_threshold(img)
            elif func == "OTSU阈值分割":
                res = otsu_threshold(img)
            elif func == "边缘引导阈值":
                res = edge_guided_threshold(img)
            elif func == "自适应阈值":
                res = adaptive_threshold(img, p["blockSize"].value(), p["C"].value())
            elif func == "区域增长分割":
                res = region_growing(img, thresh=p["thresh"].value())
            elif func == "提取目标边界":
                res, _ = find_contours(img)
            elif func == "PCA压缩":
                res = pca_compress(img, p["num_components"].value())
            elif func == "腐蚀":
                res = erode(img, ksize=3)
            elif func == "膨胀":
                res = dilate(img, ksize=3)
            elif func == "开运算":
                res = morph_open(img, ksize=3)
            elif func == "闭运算":
                res = morph_close(img, ksize=3)
            elif func == "形态学梯度":
                res = morph_gradient(img, ksize=3)
            elif func == "顶帽":
                res = morph_tophat(img, ksize=3)
            elif func == "黑帽":
                res = morph_blackhat(img, ksize=3)
            elif func in ["查找所有轮廓", "绘制轮廓", "多边形逼近", "外接矩形", "最小外接矩形", "最小外接圆"]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                img_for_contour = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                contours, _ = find_all_contours(img_for_contour)
                if not contours:
                    QMessageBox.information(self, "提示", "未检测到轮廓，请尝试先进行二值化或调整图像对比度！")
                    res = img
                else:
                    if func == "绘制轮廓":
                        res = draw_contour(img, contours[0], color=(255,0,0), thickness=2)
                    elif func == "多边形逼近":
                        approx = approx_poly_contour(contours[0])
                        res = img.copy()
                        cv2.polylines(res, [approx], True, (0,0,255), 2)
                    elif func == "外接矩形":
                        x, y, w, h = bounding_rect(contours[0])
                        res = img.copy()
                        cv2.rectangle(res, (x, y), (x+w, y+h), (0,255,255), 2)
                    elif func == "最小外接矩形":
                        box, rect = min_area_rect(contours[0])
                        res = img.copy()
                        cv2.drawContours(res, [box], 0, (255,0,255), 2)
                    elif func == "最小外接圆":
                        center, radius = min_enclosing_circle(contours[0])
                        res = img.copy()
                        cv2.circle(res, center, radius, (0,128,255), 2)
            else:
                res = img
        except Exception:
            res = img
        if 'res' not in locals():
            res = img
        self.result_img = res
        self.result_label.setPixmap(cvimg2qt(res).scaled(self.result_label.size(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.stop_video_camera()
        super().closeEvent(event)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        if fname:
            img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.input_img = img
                self.input_label.setPixmap(cvimg2qt(img).scaled(self.input_label.size(), Qt.KeepAspectRatio))
                self.result_label.clear()
                self.cap = None

    def open_video(self):
        fname, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if fname:
            self.cap = cv2.VideoCapture(fname)
            self.video_mode = "video"
            self.timer.start(30)

    def open_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.video_mode = "camera"
        self.timer.start(30)

    def stop_video_camera(self):
        if hasattr(self, '_video_writer') and self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._video_save_count = 0
        self._video_save_name = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self.timer.stop()

def cvimg2qt(img):
    """OpenCV图像转QPixmap"""
    from PyQt5.QtGui import QImage, QPixmap
    if len(img.shape) == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)