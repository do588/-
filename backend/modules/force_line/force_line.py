import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

class ForceLineDrawer:
    def __init__(self):
        model_path = r"C:\Users\18316\Desktop\undergraduate\suffermore\fuchanghong\integrated\backend\pose_landmarker_full.task"

        # 检查文件是否存在
        print("模型路径：", model_path)
        print("文件是否存在：", os.path.exists(model_path))

        # 初始化检测器
        base_options = BaseOptions(model_asset_path=model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False
        )
        self.detector = PoseLandmarker.create_from_options(options)

    def detect_keypoints(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = self.detector.detect(mp_image)
            if result.pose_landmarks:
                print(f"成功检测到关键点，数量: {len(result.pose_landmarks[0])}")
                return [[p.x, p.y, p.z] for p in result.pose_landmarks[0]]
            else:
                print("未检测到关键点")
                return None
        except Exception as e:
            print(f"关键点检测失败: {e}")
            return None

    def calculate_center_of_mass(self, keypoints):
        if not keypoints: return None
        xs = [p[0] for p in keypoints]
        ys = [p[1] for p in keypoints]
        return np.mean(xs), np.mean(ys)

    def draw_force_line(self, frame):
        h, w = frame.shape[:2]
        keypoints = self.detect_keypoints(frame)
        if not keypoints:
            # 未检测到关键点时，在帧中间绘制一条黄色虚线作为参考
            # 保持原始帧内容不变
            for i in range(0, h, 20):  # 绘制虚线
                cv2.line(frame, (w//2, i), (w//2, min(i+10, h)), (0,255,255), 2)
            return frame, None

        # 使用关键身体部位计算力线
        # 关键点索引：0-鼻子, 11-左肩, 12-右肩, 23-左髋, 24-右髋, 27-左踝, 28-右踝
        key_points_indices = [0, 11, 12, 23, 24, 27, 28]
        key_points = [keypoints[i] for i in key_points_indices if i < len(keypoints)]
        
        if key_points:
            # 计算关键部位的质心
            xs = [p[0] for p in key_points]
            ys = [p[1] for p in key_points]
            com_x, com_y = np.mean(xs), np.mean(ys)
            cx, cy = int(com_x*w), int(com_y*h)
            
            # 绘制力线
            cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)  # 红色圆点表示重心
            cv2.line(frame, (cx, 0), (cx, h), (0,255,0), 3)  # 绿色垂直线表示力线
            
            # 可选：绘制关键部位的点
            for i in key_points_indices:
                if i < len(keypoints):
                    px, py = int(keypoints[i][0]*w), int(keypoints[i][1]*h)
                    cv2.circle(frame, (px, py), 3, (255,0,0), -1)  # 蓝色圆点表示关键部位
        else:
            # 如果没有关键部位，使用所有关键点的质心
            com_x, com_y = self.calculate_center_of_mass(keypoints)
            cx, cy = int(com_x*w), int(com_y*h)
            cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)
            cv2.line(frame, (cx,0), (cx,h), (0,255,0), 2)
        
        return frame, keypoints

# 全局实例
drawer = ForceLineDrawer()

def draw_force_line(frame):
    return drawer.draw_force_line(frame)