import cv2
import numpy as np
from ultralytics import YOLO

class TrajectoryGenerator:
    def __init__(self, weight_path='yolov8n.pt', conf=0.3):
        self.model = YOLO(weight_path)
        self.conf = conf
        self.track_history = {}  # id: list of (x, y)
        self.colors = self._generate_colors(50)

    def _generate_colors(self, n):
        np.random.seed(42)
        return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(n)]

    def reset(self):
        self.track_history = {}

    def infer(self, img):
        # 使用YOLO的track接口，指定CUDA
        results = self.model.track(img, persist=True, device='cuda')
        boxes = results[0].boxes
        centers = []
        bboxes = []
        track_ids = []
        current_ids = set()
        # 修复：boxes.id 可能为 None
        if boxes is not None and hasattr(boxes, 'xywh') and hasattr(boxes, 'id') and boxes.id is not None:
            xywh = boxes.xywh.cpu().numpy()
            ids = boxes.id.int().cpu().numpy()
            for i, (box, tid) in enumerate(zip(xywh, ids)):
                x, y, w, h = box
                cx, cy = int(x), int(y)
                centers.append((cx, cy))
                x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                bboxes.append((x1, y1, x2, y2))
                # 轨迹追加，长度超过30则pop(0)
                if tid not in self.track_history:
                    self.track_history[tid] = []
                self.track_history[tid].append((cx, cy))
                if len(self.track_history[tid]) > 30:
                    self.track_history[tid].pop(0)
                track_ids.append(tid)
                current_ids.add(tid)
        # 移除未检测到的目标的历史轨迹
        remove_ids = [tid for tid in self.track_history if tid not in current_ids]
        for tid in remove_ids:
            del self.track_history[tid]
        return centers, self.track_history, bboxes, track_ids, results

    def draw_trajectories(self, img, track_history, bboxes, track_ids):
        out = img.copy()
        # 画框
        for i, (bbox, tid) in enumerate(zip(bboxes, track_ids)):
            color = self.colors[tid % len(self.colors)]
            color = tuple(int(x) for x in color)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # 画轨迹（只画当前帧）
        for tid, pts in track_history.items():
            color = self.colors[tid % len(self.colors)]
            color = tuple(int(x) for x in color)
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    cv2.line(out, pts[i-1], pts[i], color, 5)
            if pts:
                cv2.circle(out, pts[-1], 7, color, -1)
        return out
