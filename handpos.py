import cv2
import numpy as np
import mediapipe as mp

class HandPoseEstimator:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

    def infer(self, image_bgr):
        """
        输入: image_bgr (OpenCV BGR格式)
        输出: 原图, hand_keypoints_info, results
        hand_keypoints_info: [ [ (x1, y1), (x2, y2), ... ], ... ]  # 每只手21个关键点
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        hand_keypoints_info = []
        if results.multi_hand_landmarks:
            h, w, _ = image_bgr.shape
            for hand_landmarks in results.multi_hand_landmarks:
                points = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                hand_keypoints_info.append(points)
        return image_bgr.copy(), hand_keypoints_info, results

if __name__ == "__main__":
    estimator = HandPoseEstimator()
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        result_img, hand_keypoints_info, results = estimator.infer(img)
        # 可视化关键点
        for hand in hand_keypoints_info:
            for idx, (x, y) in enumerate(hand):
                cv2.circle(result_img, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(result_img, str(idx), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow("HandPose", result_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()