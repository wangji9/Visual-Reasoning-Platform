import cv2
from ultralytics import YOLO

class Segmentor:
    def __init__(self, weight_path):
        self.model = YOLO(weight_path)

    def infer(self, image_bgr):
        """
        输入: image_bgr (OpenCV BGR格式)
        输出: result_img (带分割mask的BGR图像), results (原始推理结果)
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(image_rgb, task="segment", device='cuda')
        result_img = results[0].plot()  # 带分割mask的RGB图像
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)  # 转回BGR，保证输入输出一致
        # 提取分割坐标信息
        seg_info = []
        for i, mask in enumerate(results[0].masks.xy if results[0].masks is not None else []):
            # mask: [N,2]，N为多边形点数
            coords = [(int(x), int(y)) for x, y in mask]
            seg_info.append(coords)
        return result_img, seg_info, results

if __name__ == "__main__":
    segmentor = Segmentor("yolov8n-seg.pt")
    capture = cv2.VideoCapture(0)  # 打开摄像头

    while True:
        ret, img = capture.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        result_img, seg_info, results = segmentor.infer(img)
        # 在画面上渲染分割多边形的部分坐标
        for idx, coords in enumerate(seg_info):
            if coords:
                # 只显示第一个点和多边形中心
                x0, y0 = coords[0]
                cv2.circle(result_img, (x0, y0), 5, (0, 255, 0), -1)
                cv2.putText(result_img, f"Obj{idx}({x0},{y0})", (x0+5, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                # 计算多边形中心
                xs = [x for x, y in coords]
                ys = [y for x, y in coords]
                cx, cy = int(sum(xs)/len(xs)), int(sum(ys)/len(ys))
                cv2.circle(result_img, (cx, cy), 4, (255, 0, 0), -1)
                cv2.putText(result_img, f"C({cx},{cy})", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow("segmentor", result_img)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC退出
            break

    capture.release()
    cv2.destroyAllWindows()