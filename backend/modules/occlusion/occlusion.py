import numpy as np

class OcclusionDetector:
    def __init__(self):
        # 关键关节点索引
        self.key_joints = [
            0,   # 头部
            11,  # 左肩
            12,  # 右肩
            23,  # 左髋
            24,  # 右髋
            25,  # 左膝
            26,  # 右膝
            27,  # 左脚踝
            28   # 右脚踝
        ]
    
    def detect_occlusion(self, keypoints):
        """检测遮挡情况"""
        if keypoints is None or len(keypoints) < 33:
            return 0.0
        
        try:
            # 计算可见度评分
            visibility_scores = []
            for idx in self.key_joints:
                if idx < len(keypoints):
                    # 可见度值在0-1之间
                    visibility = keypoints[idx, 3]
                    visibility_scores.append(visibility)
                else:
                    visibility_scores.append(0)
            
            # 计算平均可见度
            if visibility_scores:
                avg_visibility = np.mean(visibility_scores)
                # 转换为0-100的评分
                occlusion_score = avg_visibility * 100
                return occlusion_score
            return 0.0
        except Exception as e:
            print(f"Error detecting occlusion: {e}")
            return 0.0

# 全局检测器实例
detector = OcclusionDetector()

# 直接的函数接口
def detect_occlusion(keypoints):
    return detector.detect_occlusion(keypoints)