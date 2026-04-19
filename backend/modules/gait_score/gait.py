import numpy as np

class GaitAnalyzer:
    def __init__(self):
        self.frames = []

    def add_frame(self, keypoints):
        """添加关键点到分析队列"""
        if keypoints is not None:
            self.frames.append(keypoints)

    def calculate_angle(self, p1, p2, p3):
        """计算三点夹角（髋-膝-踝）"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_symmetry(self):
        """计算步态对称性评分"""
        if len(self.frames) < 5:
            return 0

        left_knee_angles = []
        right_knee_angles = []

        for kp in self.frames:
            # 左膝：左髋 - 左膝 - 左踝
            l_hip = kp[23][:2]
            l_knee = kp[25][:2]
            l_ankle = kp[27][:2]
            left_angle = self.calculate_angle(np.array(l_hip), np.array(l_knee), np.array(l_ankle))
            left_knee_angles.append(left_angle)

            # 右膝：右髋 - 右膝 - 右踝
            r_hip = kp[24][:2]
            r_knee = kp[26][:2]
            r_ankle = kp[28][:2]
            right_angle = self.calculate_angle(np.array(r_hip), np.array(r_knee), np.array(r_ankle))
            right_knee_angles.append(right_angle)

        avg_left = np.mean(left_knee_angles)
        avg_right = np.mean(right_knee_angles)

        angle_diff = abs(avg_left - avg_right)
        symmetry_score = max(0, 100 - (angle_diff * 2))

        return symmetry_score

    def calculate_gait_score(self):
        """综合步态评分"""
        symmetry = self.calculate_symmetry()
        return symmetry

# 全局接口（保持与原代码一致）
analyzer = GaitAnalyzer()

def calculate_gait_score(keypoints):
    """兼容原代码调用方式"""
    analyzer.add_frame(keypoints)
    return analyzer.calculate_gait_score()