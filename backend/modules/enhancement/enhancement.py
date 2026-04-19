import cv2
import numpy as np
import torch
import torch.nn as nn

class DeepEnhancer(nn.Module):
    def __init__(self):
        super(DeepEnhancer, self).__init__()
        # 特征提取
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # 注意力模块
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # 特征提取
        features = self.encoder(x)
        # 注意力机制
        attention = self.attention(features)
        # 特征融合
        enhanced = features * attention
        # 解码
        output = self.decoder(enhanced)
        return output

class LowLightEnhancer:
    def __init__(self):
        # 初始化深度学习模型
        self.deep_model = DeepEnhancer()
        # 获取模型文件的绝对路径
        import os
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "low_light_model.pth")
        try:
            if torch.cuda.is_available():
                self.deep_model.load_state_dict(torch.load(model_path))
                self.deep_model = self.deep_model.cuda()
            else:
                self.deep_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.deep_model.eval()
            self.use_deep = True
        except Exception as e:
            print(f"Deep learning model not available: {e}")
            self.use_deep = False
    
    def enhance_progressive(self, frame):
        """渐进式低光照增强"""
        # 转换为LAB颜色空间
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 增强亮度通道
        l = cv2.equalizeHist(l)
        
        # 合并通道
        enhanced_lab = cv2.merge([l, a, b])
        enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 对比度增强
        alpha = 1.2  # 对比度增益
        beta = 10    # 亮度增益
        enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=alpha, beta=beta)
        
        return enhanced_frame
    
    def enhance_deep_learning(self, frame):
        """深度学习低光照增强"""
        if not self.use_deep:
            return self.enhance_progressive(frame)
        
        try:
            # 保存原始大小
            original_height, original_width = frame.shape[:2]
            
            # 预处理
            frame = frame.astype(np.float32) / 255.0
            frame = cv2.resize(frame, (256, 256))
            frame = frame.transpose(2, 0, 1)
            frame = torch.tensor(frame[np.newaxis, ...], dtype=torch.float32)
            
            # 预测
            if torch.cuda.is_available():
                frame = frame.cuda()
            
            with torch.no_grad():
                output = self.deep_model(frame)
                output = output.cpu().numpy()[0]
                output = output.transpose(1, 2, 0)
                output = (output + 1) / 2  # 反归一化
                output = np.clip(output, 0, 1)
                output = (output * 255).astype(np.uint8)
                output = cv2.resize(output, (original_width, original_height))
            
            return output
        except Exception as e:
            print(f"Error in deep learning enhancement: {e}")
            return self.enhance_progressive(frame)
    
    def enhance(self, frame, method='deep_learning'):
        """增强低光照图像"""
        if method == 'progressive':
            return self.enhance_progressive(frame)
        else:
            return self.enhance_deep_learning(frame)

# 全局增强器实例
enhancer = LowLightEnhancer()

# 直接的函数接口
def enhance_low_light(frame, method='deep_learning'):
    return enhancer.enhance(frame, method)