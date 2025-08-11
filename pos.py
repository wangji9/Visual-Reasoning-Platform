import cv2
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def infer(self, image_bgr):
        """
        输入: image_bgr (OpenCV BGR格式)
        输出: result_img (带关键点的BGR图像), results (原始推理结果)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, task="pose", device='cuda')
        # result_img = results[0].plot()  # 带关键点的BGR图像（实际为RGB格式）
        result_img = results[0].plot()
        # plot输出为RGB格式，需转回BGR
        if result_img.shape[2] == 3:
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        keypoints_info = []
        for i, kp in enumerate(results[0].keypoints.xy):
            # kp: [num_keypoints, 2]
            for j, (x, y) in enumerate(kp):
                keypoints_info.append((int(x), int(y)))
        return result_img, keypoints_info, results

if __name__ == "__main__":
    estimator = PoseEstimator("yolov8n-pose.pt")
    capture = cv2.VideoCapture(0)  # 打开摄像头

    while True:
        ret, img = capture.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        result_img, keypoints_info, results = estimator.infer(img)
        # 在画面上渲染关键点坐标
        for idx, (x, y) in enumerate(keypoints_info):
            cv2.circle(result_img, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(result_img, f"{idx}:({x},{y})", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow("estimator", result_img)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break

    capture.release()
    cv2.destroyAllWindows()