import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def infer(self, image_bgr):
        """
        输入: image_bgr (OpenCV BGR格式)
        输出: result_img (带检测框的BGR图像), results (原始推理结果)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, device='cuda')
        result_img = results[0].plot()  # 实际为RGB格式
        # plot输出为RGB格式，需转回BGR
        if result_img.shape[2] == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        # 提取检测框坐标和类别
        boxes_info = []
        for box in results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []:
            x1, y1, x2, y2 = map(int, box[:4])
            boxes_info.append((x1, y1, x2, y2))
        classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
        return result_img, boxes_info, classes, results

if __name__ == "__main__":
    detector = ObjectDetector("yolov8n.pt")
    capture = cv2.VideoCapture(0)  # 打开摄像头
    print(detector.model.info())  # 打印模型的信息
    print(detector.model.model)  # 打印模型的结构

    while True:
        ret, img = capture.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        result_img, boxes_info, classes, results = detector.infer(img)
        # 在画面上渲染检测框坐标和类别
        for idx, (box, cls_id) in enumerate(zip(boxes_info, classes)):
            x1, y1, x2, y2 = box
            cv2.putText(result_img, f"Obj{idx} ({x1},{y1})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(result_img, f"cls:{int(cls_id)}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow("Detection", result_img)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break

    capture.release()
    cv2.destroyAllWindows()