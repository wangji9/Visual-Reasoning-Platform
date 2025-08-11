import cv2
import time
import requests
import base64
import hashlib
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime, timedelta

class RealTimePersonDetector:
    def __init__(self, webhook_url):
        # 初始化YOLOv8模型
        self.model = YOLO('yolov8l.pt')  # 使用nano版本轻量模型
        self.cap = cv2.VideoCapture(0)
        self.webhook_url = webhook_url
        
        # 人体检测相关配置
        self.class_id = 0  # YOLO模型中人的类别ID
        self.min_conf = 0.6  # 最小置信度阈值
        self.best_shots = 1  # 最多保存的置信度最高的图片数量
        
        # 冷却机制配置
        self.last_alert_time = datetime.min  # 上次警报时间
        self.alert_cooldown = timedelta(seconds=1)  # 警报冷却时间(5秒)
        
        # 置信度最高的图片缓存
        self.best_detections = defaultdict(float)  # {filename: confidence}
        
        # 创建保存目录
        os.makedirs('detections', exist_ok=True)
        
    def process_frame(self):
        """处理视频帧并进行人体检测"""
        success, frame = self.cap.read()
        if not success:
            print("Warning: Failed to read frame")
            return [], None  # 返回空列表和None
        
        # 运行YOLO检测
        results = self.model.predict(frame, verbose=False)
        
        detections = []
        current_time = datetime.now()
        
        # 解析检测结果
        for result in results:
            for box in result.boxes:
                if box.cls == self.class_id and box.conf >= self.min_conf:
                    conf = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # 计算检测区域面积和中心点
                    area = (x2 - x1) * (y2 - y1)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detections.append({
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'area': area,
                        'center': (center_x, center_y),
                        'timestamp': current_time
                    })
        
        return detections, frame
    
    def save_frame(self, frame, max_confidence, timestamp):
        """保存整个画面而非裁剪区域"""
        if frame.size == 0:  # 确保帧有效
            return None
            
        # 压缩图像 (确保 < 2MB)
        quality = 95
        while quality >= 30:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img_bytes = buffer.tobytes()
            if len(img_bytes) < 2 * 1024 * 1024:  # 小于2MB
                break
            quality -= 5
        
        # 生成唯一文件名
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"detections/person_{timestamp_str}_{max_confidence:.2f}.jpg"
        
        # 保存图像
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        
        # 更新最佳检测缓存
        self.update_best_detections(filename, max_confidence)
        
        return filename, img_bytes
    
    def update_best_detections(self, filename, confidence):
        """更新置信度最高的图片缓存"""
        # 如果已存在相同的置信度，增加微小偏差确保唯一性
        while confidence in self.best_detections.values():
            confidence += 0.001
        
        # 添加新检测
        self.best_detections[filename] = confidence
        
        # 如果超过最大数量，移除置信度最低的
        if len(self.best_detections) > self.best_shots:
            min_conf_filename = min(self.best_detections, key=self.best_detections.get)
            self.best_detections.pop(min_conf_filename)
    
    def send_wechat_alert(self):
        """向企业微信发送警报"""
        if not self.best_detections:
            return False
            
        # 发送文本消息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text_msg = {
            "msgtype": "text",
            "text": {
                "content": f"⚠️ 实时检测到人员！\n时间：{timestamp}\n将发送{len(self.best_detections)}张置信度最高的图片"
            }
        }
        requests.post(self.webhook_url, json=text_msg)
        
        # 发送置信度最高的图片
        for filename, conf in sorted(self.best_detections.items(), key=lambda x: x[1], reverse=True):
            try:
                # 读取之前保存的文件
                with open(filename, 'rb') as f:
                    img_bytes = f.read()
                
                # 准备图片数据
                md5_hash = hashlib.md5(img_bytes).hexdigest()
                base64_data = base64.b64encode(img_bytes).decode('utf-8')
                
                # 构建图片消息
                image_msg = {
                    "msgtype": "image",
                    "image": {
                        "base64": base64_data,
                        "md5": md5_hash
                    }
                }
                
                # 发送图片
                requests.post(self.webhook_url, json=image_msg)
                
                # 添加置信度文本
                conf_text = {
                    "msgtype": "text",
                    "text": {
                        "content": f"置信度: {conf:.3f}"
                    }
                }
                requests.post(self.webhook_url, json=conf_text)
                
                # 小延迟避免消息阻塞
                time.sleep(0.2)
            except Exception as e:
                print(f"发送图片失败: {e}")
        
        # 清空缓存准备下一轮检测
        self.best_detections.clear()
        return True
    
    def run_detection(self):
        """运行实时检测循环"""
        print("启动实时人体检测系统...")
        print(f"最小置信度阈值: {self.min_conf}")
        print(f"保存最佳检测数: {self.best_shots}")
        
        # 设置视频编解码器（解决部分系统的兼容性问题）
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while self.cap.isOpened():
                # 处理当前帧
                detections, frame = self.process_frame()
                
                # 如果帧读取失败，跳过本次循环
                if frame is None:
                    time.sleep(0.1)  # 短暂等待后重试
                    continue
                
                current_time = datetime.now()
                
                # 检查是否在冷却期
                in_cooldown = current_time - self.last_alert_time < self.alert_cooldown
                
                # 计算当前帧的最大置信度
                max_confidence = max([d['confidence'] for d in detections]) if detections else 0.0
                
                # 如果检测到人且不在冷却期，保存并发送警报
                if detections and not in_cooldown:
                    # 保存整个画面
                    self.save_frame(frame, max_confidence, current_time)
                    
                    if self.send_wechat_alert():
                        self.last_alert_time = current_time
                        print(f"警报已发送: {self.last_alert_time.strftime('%H:%M:%S')}")
                
                # 显示状态信息
                # self.display_status(frame, detections, in_cooldown, max_confidence)
                
                # 按ESC退出
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("系统已安全关闭")
    
    def display_status(self, frame, detections, in_cooldown, max_confidence):
        """在画面上显示检测状态"""
        # 显示检测数量
        status_text = f"Detections: {len(detections)}"
        cv2.putText(frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示当前最高置信度
        max_conf_text = f"Max Conf: {max_confidence:.2f}"
        cv2.putText(frame, max_conf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # 显示状态
        state_color = (0, 0, 255) if in_cooldown else (255, 0, 0)
        state_text = "Cooldown" if in_cooldown else "Active"
        cv2.putText(frame, state_text, (frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # 显示最佳检测数量
        best_text = f"Best shots: {len(self.best_detections)}/{self.best_shots}"
        cv2.putText(frame, best_text, (frame.shape[1]-150, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        
        # 显示置信度阈值
        conf_text = f"Conf thresh: {self.min_conf:.2f}"
        cv2.putText(frame, conf_text, (frame.shape[1]-150, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)
        
        # 显示检测框
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # 画检测框和置信度
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            conf_text = f"{conf:.2f}"
            cv2.putText(frame, conf_text, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 显示预览画面
        cv2.imshow('Real-time Person Detection', frame)

if __name__ == "__main__":
    # 配置企业微信机器人Webhook URL
    WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a7b71c9f-8b6f-4158-8249-663e2dbc12d4"
    
    # 创建并运行检测器
    detector = RealTimePersonDetector(WEBHOOK_URL)
    detector.run_detection()
