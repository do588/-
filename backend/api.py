import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp

# 导入项目模块
from modules.enhancement.enhancement import enhance_low_light
from modules.force_line.force_line import draw_force_line, drawer
from modules.gait_score.gait import GaitAnalyzer
from modules.view_condition.view_condition import ViewConditionModel
from modules.fall_risk.fall_risk import FallRiskEvaluator
from modules.occlusion.occlusion import detect_occlusion

from flask_cors import CORS

# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)  # 解决跨域问题
os.makedirs('output', exist_ok=True)  # 确保输出目录存在

# 全局初始化模型（避免重复创建，提升性能）
view_classifier = ViewConditionModel()
fall_risk_predictor = FallRiskEvaluator()

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    """视频分析接口：接收视频，返回步态评分、力线分析等结果"""
    # 1. 校验请求
    if 'video' not in request.files:
        return jsonify({'error': '未上传视频文件'}), 400

    video_file = request.files['video']
    enhancement_method = request.form.get('enhancement_method', 'deep_learning')
    use_enhancement = True

    # 2. 保存临时视频
    temp_video_path = os.path.join('output', 'uploaded_video.mp4')
    video_file.save(temp_video_path)

    # 3. 处理视频
    results = process_video(temp_video_path, enhancement_method)
    if results is None:
        return jsonify({'error': '视频处理失败'}), 500

    return jsonify(results)

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """实时帧分析接口：接收单帧图像，返回实时分析结果"""
    # 1. 校验请求
    if 'frame' not in request.files:
        return jsonify({'error': '未上传帧图像'}), 400

    frame_file = request.files['frame']
    enhancement_method = request.form.get('enhancement_method', 'deep_learning')

    # 2. 读取帧图像
    import numpy as np
    import cv2
    from io import BytesIO

    # 读取图像数据
    img_bytes = frame_file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': '无法读取帧图像'}), 400

    # 3. 处理帧
    results = process_frame(frame, enhancement_method)
    if results is None:
        return jsonify({'error': '帧处理失败'}), 500

    return jsonify(results)

def process_frame(frame, enhancement_method='deep_learning'):
    """处理单个帧的分析"""
    # 1. 保存原始帧
    original_frame = frame.copy()
    
    # 2. 对帧进行低光增强
    enhanced_frame = enhance_low_light(frame, method=enhancement_method)
    
    # 3. 尝试检测关键点（在增强后的帧上）
    keypoints = drawer.detect_keypoints(enhanced_frame)
    
    # 4. 绘制力线
    if keypoints is not None:
        # 检测到关键点，绘制力线
        enhanced_frame, _ = drawer.draw_force_line(enhanced_frame)
        print(f"关键点检测: 成功，数量: {len(keypoints)}")
    else:
        # 未检测到关键点，尝试在原始帧上检测
        print("尝试在原始帧上检测关键点")
        keypoints = drawer.detect_keypoints(original_frame)
        if keypoints is not None:
            # 在原始帧上检测到关键点，使用原始帧
            enhanced_frame = original_frame.copy()
            enhanced_frame, _ = drawer.draw_force_line(enhanced_frame)
            print(f"原始帧上检测到关键点，数量: {len(keypoints)}")
        else:
            # 仍然未检测到，使用原始帧并绘制参考线
            enhanced_frame = original_frame.copy()
            h, w = enhanced_frame.shape[:2]
            for i in range(0, h, 20):
                cv2.line(enhanced_frame, (w//2, i), (w//2, min(i+10, h)), (0,255,255), 2)
            print("关键点检测: 失败")
    
    # 5. 遮挡检测
    occlusion = detect_occlusion(enhanced_frame)
    
    # 6. 初始化分析器
    gait_analyzer = GaitAnalyzer()
    all_keypoints = []

    # 7. 收集关键点，用于步态分析
    if keypoints is not None:
        gait_analyzer.add_frame(keypoints)
        all_keypoints.append(keypoints)
        print(f"已添加关键点，当前帧数量: {len(gait_analyzer.frames)}")

    # 8. 生成分析结果
    print("\n生成分析报告:")
    gait_score = gait_analyzer.calculate_gait_score()  # 步态评分
    results = {'gait_score': gait_score}

    # 8.1 视角评估
    view = "Unknown"
    condition = "Normal"  # 默认设为Normal，避免误判
    
    # 使用当前帧进行预测
    view_condition = view_classifier.predict(enhanced_frame)
    if isinstance(view_condition, tuple) and len(view_condition) == 2:
        view, c = view_condition
        # 过滤掉不合理的预测结果
        valid_conditions = ['Normal', 'Backpack', 'Bag', 'Coat']
        if c in valid_conditions:
            condition = c
    
    # 基于关键点的视角和背包检测
    if all_keypoints and len(all_keypoints) > 0:
        # 使用当前帧的关键点进行分析
        keypoints = all_keypoints[0]
        if keypoints is not None and len(keypoints) >= 33:
            # 计算肩宽和髋宽的比例
            left_shoulder = keypoints[11]
            right_shoulder = keypoints[12]
            left_hip = keypoints[23]
            right_hip = keypoints[24]
            
            if all(len(point) >= 3 for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                hip_width = abs(left_hip[0] - right_hip[0])
                
                # 视角判断
                if shoulder_width < 0.1 and hip_width < 0.1:
                    view = "Side"
                elif abs(shoulder_width - hip_width) < 0.1:
                    view = "Front"
                else:
                    view = "45 Degrees"
                print(f"基于关键点的视角判断: {view}")
                
                # 背包检测
                # 检查肩部和背部区域的关键点高度差异
                # 背包会导致肩部关键点相对于背部关键点位置异常
                left_ear = keypoints[7]
                right_ear = keypoints[8]
                left_shoulder_y = left_shoulder[1]
                right_shoulder_y = right_shoulder[1]
                left_ear_y = left_ear[1]
                right_ear_y = right_ear[1]
                
                # 计算肩耳高度差
                shoulder_ear_diff_left = abs(left_shoulder_y - left_ear_y)
                shoulder_ear_diff_right = abs(right_shoulder_y - right_ear_y)
                
                # 正常情况下，肩耳高度差应该在合理范围内
                # 背包会导致肩部变宽或高度异常
                has_backpack = False
                if shoulder_width > 0.3:  # 肩宽异常大
                    has_backpack = True
                elif shoulder_ear_diff_left > 0.2 or shoulder_ear_diff_right > 0.2:  # 肩耳高度差异常
                    has_backpack = True
                
                if has_backpack:
                    condition = "Backpack"
                    print("基于关键点检测到背包")
                else:
                    condition = "Normal"
                    print("基于关键点未检测到背包")
    
    print(f"最终视角状态: {view}, 行走条件: {condition}")

    # 8.2 跌倒风险评估
    fall_risk = "Low"
    risk_factors = []
    if all_keypoints:
        fall_risk_result = fall_risk_predictor.calculate_fall_risk(all_keypoints)
        if isinstance(fall_risk_result, dict):
            fall_risk = fall_risk_result.get('risk_level', 'Low')
            risk_factors = fall_risk_result.get('risk_factors', [])
        else:
            fall_risk = fall_risk_result
        print(f"跌倒风险: {fall_risk}")

    # 8.3 遮挡状态
    occlusion_score = 100.0
    if occlusion and isinstance(occlusion, dict):
        occlusion_score = occlusion.get('score', 100.0)
        print(f"遮挡状态: {occlusion_score}")

    # 9. 构建返回结果
    return {
        'success': True,
        'view': view,
        'condition': condition,
        'gait_score': results.get('gait_score', 0.0),
        'fall_risk': fall_risk,
        'occlusion_score': occlusion_score,
        'risk_factors': risk_factors
    }

def process_video(video_path, enhancement_method='deep_learning'):
    """核心视频处理流程：增强、力线绘制、步态评分、风险评估"""
    # 1. 校验视频
    if not os.path.exists(video_path):
        print(f" 视频不存在: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" 无法打开视频")
        return None

    # 2. 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频参数: {width}x{height} @ {fps}fps")

    # 3. 初始化分析器
    gait_analyzer = GaitAnalyzer()
    frame_count = 0
    all_keypoints = []
    all_frames = []

    print(f"开始处理视频，增强方法: {enhancement_method}")

    # 4. 逐帧处理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4.1 保存原始帧
        original_frame = frame.copy()
        
        # 4.2 对帧进行低光增强
        enhanced_frame = enhance_low_light(frame, method=enhancement_method)
        
        # 4.3 尝试检测关键点（在增强后的帧上）
        keypoints = drawer.detect_keypoints(enhanced_frame)
        
        # 4.4 绘制力线
        if keypoints is not None:
            # 检测到关键点，绘制力线
            enhanced_frame, _ = drawer.draw_force_line(enhanced_frame)
            print(f"关键点检测: 成功，数量: {len(keypoints)}")
        else:
            # 未检测到关键点，尝试在原始帧上检测
            print("尝试在原始帧上检测关键点")
            keypoints = drawer.detect_keypoints(original_frame)
            if keypoints is not None:
                # 在原始帧上检测到关键点，使用原始帧
                enhanced_frame = original_frame.copy()
                enhanced_frame, _ = drawer.draw_force_line(enhanced_frame)
                print(f"原始帧上检测到关键点，数量: {len(keypoints)}")
            else:
                # 仍然未检测到，使用原始帧并绘制参考线
                enhanced_frame = original_frame.copy()
                h, w = enhanced_frame.shape[:2]
                for i in range(0, h, 20):
                    cv2.line(enhanced_frame, (w//2, i), (w//2, min(i+10, h)), (0,255,255), 2)
                print("关键点检测: 失败")
        
        # 遮挡检测
        occlusion = detect_occlusion(enhanced_frame)
        
        # 使用处理后的帧
        frame = enhanced_frame

        # 4.4 收集关键点，用于步态分析
        if keypoints is not None:
            gait_analyzer.add_frame(keypoints)
            all_keypoints.append(keypoints)
            print(f"已添加关键点，当前帧数量: {len(gait_analyzer.frames)}")

        # 4.5 保存处理后的帧
        all_frames.append(frame)
        frame_count += 1

        # 进度打印
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count} 帧")

    cap.release()

    # 5. 生成分析结果
    print("\n生成分析报告:")
    gait_score = gait_analyzer.calculate_gait_score()  # 步态评分
    results = {'gait_score': gait_score}

    # 5.1 视角评估
    view = "Unknown"
    condition = "Normal"  # 默认设为Normal，避免误判
    if all_frames:
        # 使用多帧进行预测，提高准确性
        view_counts = {}
        condition_counts = {}
        # 选择前20帧进行预测（更多帧提高准确性）
        sample_frames = all_frames[:min(20, len(all_frames))]
        for frame in sample_frames:
            view_condition = view_classifier.predict(frame)
            if isinstance(view_condition, tuple) and len(view_condition) == 2:
                v, c = view_condition
                view_counts[v] = view_counts.get(v, 0) + 1
                condition_counts[c] = condition_counts.get(c, 0) + 1
        
        # 使用投票结果
        if view_counts:
            view = max(view_counts, key=view_counts.get)
        if condition_counts:
            # 过滤掉不合理的预测结果
            valid_conditions = ['Normal', 'Backpack', 'Bag', 'Coat']
            filtered_counts = {k: v for k, v in condition_counts.items() if k in valid_conditions}
            if filtered_counts:
                # 只有当某个条件的投票数超过半数时才采用
                total_votes = sum(filtered_counts.values())
                condition = max(filtered_counts, key=filtered_counts.get)
                if filtered_counts[condition] < total_votes * 0.6:
                    condition = "Normal"  # 如果没有明显多数，默认设为Normal
        
        # 基于关键点的视角和背包检测
        if all_keypoints and len(all_keypoints) > 0:
            # 使用第一帧的关键点进行分析
            keypoints = all_keypoints[0]
            if keypoints is not None and len(keypoints) >= 33:
                # 计算肩宽和髋宽的比例
                left_shoulder = keypoints[11]
                right_shoulder = keypoints[12]
                left_hip = keypoints[23]
                right_hip = keypoints[24]
                
                if all(len(point) >= 3 for point in [left_shoulder, right_shoulder, left_hip, right_hip]):
                    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
                    hip_width = abs(left_hip[0] - right_hip[0])
                    
                    # 视角判断
                    if shoulder_width < 0.1 and hip_width < 0.1:
                        view = "Side"
                    elif abs(shoulder_width - hip_width) < 0.1:
                        view = "Front"
                    else:
                        view = "45 Degrees"
                    print(f"基于关键点的视角判断: {view}")
                    
                    # 背包检测
                    # 检查肩部和背部区域的关键点高度差异
                    # 背包会导致肩部关键点相对于背部关键点位置异常
                    left_ear = keypoints[7]
                    right_ear = keypoints[8]
                    left_shoulder_y = left_shoulder[1]
                    right_shoulder_y = right_shoulder[1]
                    left_ear_y = left_ear[1]
                    right_ear_y = right_ear[1]
                    
                    # 计算肩耳高度差
                    shoulder_ear_diff_left = abs(left_shoulder_y - left_ear_y)
                    shoulder_ear_diff_right = abs(right_shoulder_y - right_ear_y)
                    
                    # 正常情况下，肩耳高度差应该在合理范围内
                    # 背包会导致肩部变宽或高度异常
                    has_backpack = False
                    if shoulder_width > 0.3:  # 肩宽异常大
                        has_backpack = True
                    elif shoulder_ear_diff_left > 0.2 or shoulder_ear_diff_right > 0.2:  # 肩耳高度差异常
                        has_backpack = True
                    
                    if has_backpack:
                        condition = "Backpack"
                        print("基于关键点检测到背包")
                    else:
                        condition = "Normal"
                        print("基于关键点未检测到背包")
        
        print(f"最终视角状态: {view}, 行走条件: {condition}")
        print(f"条件投票结果: {condition_counts}")
        print(f"视角投票结果: {view_counts}")

    # 5.2 跌倒风险评估
    fall_risk = "Low"
    risk_factors = []
    if all_keypoints:
        fall_risk_result = fall_risk_predictor.calculate_fall_risk(all_keypoints)
        if isinstance(fall_risk_result, dict):
            fall_risk = fall_risk_result.get('risk_level', 'Low')
            risk_factors = fall_risk_result.get('risk_factors', [])
        else:
            fall_risk = fall_risk_result
        print(f"跌倒风险: {fall_risk}")

    # 5.3 遮挡状态
    occlusion_score = 100.0
    if occlusion and isinstance(occlusion, dict):
        occlusion_score = occlusion.get('score', 100.0)
        print(f"遮挡状态: {occlusion_score}")

    # 5.4 保存处理后的视频
    output_video_path = os.path.join('output', 'processed_video.mp4')
    print(f"保存视频: {len(all_frames)} 帧, 尺寸: {width}x{height}, FPS: {fps}")
    if all_frames:
        # 尝试使用不同的编码器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 编码，更广泛支持
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"视频写入器已创建: {out.isOpened()}")
        if not out.isOpened():
            # 如果失败，尝试其他编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"尝试使用mp4v编码器: {out.isOpened()}")
        
        for i, frame in enumerate(all_frames):
            out.write(frame)
            if i % 50 == 0:
                print(f"已写入 {i+1} 帧")
        out.release()
        print(f"处理后的视频已保存到: {output_video_path}")
        # 检查文件大小
        if os.path.exists(output_video_path):
            file_size = os.path.getsize(output_video_path)
            print(f"视频文件大小: {file_size} 字节")
        else:
            print("视频文件未创建")
    else:
        print("没有帧可保存")

    # 5.5 读取处理后的视频并编码为base64
    video_base64 = ""
    if os.path.exists(output_video_path):
        import base64
        with open(output_video_path, 'rb') as f:
            video_base64 = base64.b64encode(f.read()).decode('utf-8')

    # 6. 构建返回结果
    return {
        'success': True,
        'video': video_base64,
        'view': view,
        'condition': condition,
        'gait_score': results.get('gait_score', 0.0),
        'fall_risk': fall_risk,
        'occlusion_score': occlusion_score,
        'risk_factors': risk_factors
    }

# 启动服务
if __name__ == '__main__':
    print("启动后端服务，访问 http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)