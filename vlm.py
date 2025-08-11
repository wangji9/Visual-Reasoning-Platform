# 通义千问2-VL-2B 图像理解API调用（支持图片、视频帧、摄像头帧）
# 依赖: pip install requests
import requests
import base64
import cv2
import os
from openai import OpenAI

class QwenVL2B:
    def __init__(self, api_key=None, model="qwen-vl-plus"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if not self.api_key:
            raise ValueError("请设置通义千问API Key，可通过环境变量DASHSCOPE_API_KEY或构造参数传入。")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _img_to_base64url(self, img_bgr):
        _, buf = cv2.imencode('.jpg', img_bgr)
        b64 = base64.b64encode(buf).decode()
        return f"data:image/jpeg;base64,{b64}"

    def infer_image(self, img_bgr, prompt="请描述这张图片的内容"):
        img_url = self._img_to_base64url(img_bgr)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
        }]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        # 返回主内容
        return completion.choices[0].message.content

    def infer_video(self, video_path, prompt="请描述这段视频的内容", frame_interval=10):
        cap = cv2.VideoCapture(video_path)
        results = []
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                text = self.infer_image(frame, prompt)
                results.append((idx, text))
            idx += 1
        cap.release()
        return results

    def infer_camera(self, prompt="请描述当前画面", interval=310):
        import time
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            text = self.infer_image(frame, prompt)
            print("VLM分析结果:", text)
            display_frame = frame.copy()
            y0, dy = 30, 30
            for i, line in enumerate(text.split('\n')):
                y = y0 + i * dy
                cv2.putText(display_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow("QwenVL2B Camera", display_frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            time.sleep(interval)
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    qwen = QwenVL2B(api_key="sk-0bb39f9cd3cc4c8bab853940966c7294")
    img = cv2.imread("1.jpg")
    print(qwen.infer_image(img, prompt="请用中文描述图片"))

    # 示例：视频
    # results = qwen.infer_video("test.mp4", prompt="视频内容总结", frame_interval=60)
    # for idx, text in results:
    #     print(f"帧{idx}: {text}")

    # 示例：摄像头
    # qwen.infer_camera(prompt="请用中文描述当前画面", interval=5)