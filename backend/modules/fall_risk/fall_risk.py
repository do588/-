import numpy as np

class FallRiskEvaluator:
    def __init__(self):
        # 风险等级定义
        self.risk_levels = {
            'Low': 0,
            'Medium': 1,
            'High': 2
        }
    
    def calculate_fall_risk(self, keypoints):
        """计算摔倒风险"""
        if keypoints is None or len(keypoints) == 0:
            return {'risk_level': 'Unknown', 'risk_factors': []}
        
        try:
            # 取第一帧的关键点进行分析
            first_frame_keypoints = keypoints[0]
            if first_frame_keypoints is None or len(first_frame_keypoints) < 33:
                return {'risk_level': 'Unknown', 'risk_factors': []}
            
            # 关键点索引
            LEFT_HIP = 23
            RIGHT_HIP = 24
            LEFT_KNEE = 25
            RIGHT_KNEE = 26
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            
            # 计算身体平衡性
            # 1. 髋关节高度差异
            hip_height_diff = abs(first_frame_keypoints[LEFT_HIP][1] - first_frame_keypoints[RIGHT_HIP][1])
            
            # 2. 膝关节角度差异
            def calculate_angle(p1, p2, p3):
                v1 = np.array(p1) - np.array(p2)
                v2 = np.array(p3) - np.array(p2)
                dot = np.dot(v1, v2)
                norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                if norm == 0:
                    return 0
                return np.degrees(np.arccos(np.clip(dot/norm, -1, 1)))
            
            left_knee_angle = calculate_angle(
                first_frame_keypoints[LEFT_HIP],
                first_frame_keypoints[LEFT_KNEE],
                first_frame_keypoints[LEFT_ANKLE]
            )
            right_knee_angle = calculate_angle(
                first_frame_keypoints[RIGHT_HIP],
                first_frame_keypoints[RIGHT_KNEE],
                first_frame_keypoints[RIGHT_ANKLE]
            )
            knee_angle_diff = abs(left_knee_angle - right_knee_angle)
            
            # 3. 步长估计
            step_length = abs(first_frame_keypoints[LEFT_ANKLE][0] - first_frame_keypoints[RIGHT_ANKLE][0])
            
            # 风险因素评估
            risk_factors = []
            risk_score = 0
            
            if hip_height_diff > 0.1:
                risk_factors.append('Imbalanced hip height')
                risk_score += 1
            
            if knee_angle_diff > 20:
                risk_factors.append('Asymmetric knee angles')
                risk_score += 1
            
            if step_length < 0.1 or step_length > 0.5:
                risk_factors.append('Abnormal step length')
                risk_score += 1
            
            # 确定风险等级
            if risk_score == 0:
                risk_level = 'Low'
            elif risk_score == 1:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            return {'risk_level': risk_level, 'risk_factors': risk_factors}
            
        except Exception as e:
            print(f"Error calculating fall risk: {e}")
            return {'risk_level': 'Unknown', 'risk_factors': []}

# 全局评估器实例
evaluator = FallRiskEvaluator()

# 直接的函数接口
def calculate_fall_risk(keypoints):
    return evaluator.calculate_fall_risk(keypoints)