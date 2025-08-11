import cv2
import numpy as np
import mediapipe as mp

class FaceLandmarkGaze:
    def __init__(self, static_mode=False, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            refine_landmarks=True  # 关键
        )

    def infer(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        h, w, _ = image_bgr.shape
        faces = []
        gaze_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points_2d = []
                points_3d = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points_2d.append((x, y))
                    points_3d.append((lm.x, lm.y, lm.z))
                faces.append(points_2d)

                left_pupil_idx = 468
                right_pupil_idx = 473
                if len(points_2d) > right_pupil_idx:
                    left_pupil_2d = points_2d[left_pupil_idx]
                    right_pupil_2d = points_2d[right_pupil_idx]
                    left_pupil_3d = points_3d[left_pupil_idx]
                    right_pupil_3d = points_3d[right_pupil_idx]
                    left_eye_corner_idx = 133
                    right_eye_corner_idx = 362
                    left_eye_vec = np.array(points_3d[left_pupil_idx]) - np.array(points_3d[left_eye_corner_idx])
                    right_eye_vec = np.array(points_3d[right_pupil_idx]) - np.array(points_3d[right_eye_corner_idx])
                    gaze_list.append({
                        "left_pupil_2d": left_pupil_2d,
                        "right_pupil_2d": right_pupil_2d,
                        "left_pupil_3d": left_pupil_3d,
                        "right_pupil_3d": right_pupil_3d,
                        "left_eye_vec": left_eye_vec,
                        "right_eye_vec": right_eye_vec
                    })
        return image_bgr.copy(), faces, gaze_list

def draw_face_landmarks_and_gaze(img, faces, gaze_list):
    for face, gaze in zip(faces, gaze_list):
        # 画面部关键点
        for (x, y) in face:
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        # 画瞳孔
        left_pupil_2d = tuple(map(int, gaze["left_pupil_2d"]))
        right_pupil_2d = tuple(map(int, gaze["right_pupil_2d"]))
        cv2.circle(img, left_pupil_2d, 4, (255, 0, 0), -1)
        cv2.circle(img, right_pupil_2d, 4, (0, 0, 255), -1)

        # 画二维视线方向（瞳孔到眼角）
        left_vec = gaze["left_eye_vec"]
        right_vec = gaze["right_eye_vec"]
        left_end_2d = (int(left_pupil_2d[0] + left_vec[0]*300), int(left_pupil_2d[1] + left_vec[1]*300))
        right_end_2d = (int(right_pupil_2d[0] + right_vec[0]*300), int(right_pupil_2d[1] + right_vec[1]*300))
        cv2.arrowedLine(img, left_pupil_2d, left_end_2d, (255, 0, 0), 2, tipLength=0.2)
        cv2.arrowedLine(img, right_pupil_2d, right_end_2d, (0, 0, 255), 2, tipLength=0.2)

        # 画三维视线方向（z轴影响用颜色或虚线区分，简单投影）
        left_pupil_3d = gaze["left_pupil_3d"]
        right_pupil_3d = gaze["right_pupil_3d"]
        # 假设视线方向为瞳孔z轴负方向（向前看），用z分量放大
        scale_3d = 1000
        left_end_3d = (
            int(left_pupil_2d[0] - left_pupil_3d[2]*scale_3d),
            int(left_pupil_2d[1] - left_pupil_3d[2]*scale_3d)
        )
        right_end_3d = (
            int(right_pupil_2d[0] - right_pupil_3d[2]*scale_3d),
            int(right_pupil_2d[1] - right_pupil_3d[2]*scale_3d)
        )
        # 用绿色虚线表示三维视线方向
        cv2.line(img, left_pupil_2d, left_end_3d, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(img, right_pupil_2d, right_end_3d, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(img, "3D", left_end_3d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img, "3D", right_end_3d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # 显示三维坐标
        cv2.putText(img, f"L:({left_pupil_3d[0]:.2f},{left_pupil_3d[1]:.2f},{left_pupil_3d[2]:.2f})",
                    (left_pupil_2d[0]+10, left_pupil_2d[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        cv2.putText(img, f"R:({right_pupil_3d[0]:.2f},{right_pupil_3d[1]:.2f},{right_pupil_3d[2]:.2f})",
                    (right_pupil_2d[0]+10, right_pupil_2d[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return img

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    estimator = FaceLandmarkGaze()
    while True:
        ret, img = cap.read()
        if not ret:
            break
        result_img, faces, gaze_list = estimator.infer(img)
        result_img = draw_face_landmarks_and_gaze(result_img, faces, gaze_list)
        cv2.imshow("Face Landmarks & Gaze", result_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()