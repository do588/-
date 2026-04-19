import torch
import torch.nn as nn
import cv2
import numpy as np

class GaitViewConditionModel(nn.Module):
    """旧的3D-CNN模型（保留兼容性）"""
    def __init__(self):
        super(GaitViewConditionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(256)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 32 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        self.view_head = nn.Linear(256, 3)
        self.condition_head = nn.Linear(256, 4)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        view_output = self.view_head(x)
        condition_output = self.condition_head(x)
        output = torch.cat((view_output, condition_output), dim=1)
        return output

class SimpleViewConditionModel(nn.Module):
    """改进的2D-CNN模型（处理数据不平衡）"""
    def __init__(self):
        super(SimpleViewConditionModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 第一层卷积
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 下采样
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 第二层卷积
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 下采样
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 第三层卷积
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 下采样
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 第四层卷积
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4))  # 自适应池化
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),  # 全连接层
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(512, 256),  # 全连接层
            nn.ReLU(),
            nn.Dropout(0.5)  # 防止过拟合
        )
        
        self.view_head = nn.Linear(256, 3)  # 视角分类头
        self.condition_head = nn.Linear(256, 4)  # 行走条件分类头
    
    def forward(self, x):
        # 移除时间维度 (batch, 1, 1, 32, 32) -> (batch, 1, 32, 32)
        if x.dim() == 5:
            x = x.squeeze(1)
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        view_output = self.view_head(x)
        condition_output = self.condition_head(x)
        output = torch.cat((view_output, condition_output), dim=1)
        return output

class ViewConditionModel:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        import os
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "view_condition_model.pth")
        
        try:
            if os.path.exists(model_path):
                # 尝试加载模型检查点
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                
                # 检查是否是新格式（包含model_type）
                if isinstance(checkpoint, dict) and 'model_type' in checkpoint:
                    if checkpoint['model_type'] == 'SimpleViewConditionModel':
                        self.model = SimpleViewConditionModel()
                    else:
                        self.model = GaitViewConditionModel()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # 旧格式，使用3D-CNN模型
                    self.model = GaitViewConditionModel()
                    self.model.load_state_dict(checkpoint)
                
                if torch.cuda.is_available():
                    self.model = self.model.cuda()
                
                self.model.eval()
                self.model_loaded = True
                print("View condition model loaded successfully")
            else:
                print(f"View condition model not found at: {model_path}")
                self.model_loaded = False
        except Exception as e:
            print(f"Error loading view condition model: {e}")
            self.model_loaded = False
        
        self.view_labels = {0: 'Front', 1: '45 Degrees', 2: 'Side'}
        self.condition_labels = {0: 'Normal', 1: 'Backpack', 2: 'Bag', 3: 'Coat'}
    
    def preprocess(self, frame):
        """预处理帧"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 调整大小
        resized = cv2.resize(gray, (32, 32))
        # 归一化
        normalized = resized / 255.0
        # 添加维度
        processed = normalized[np.newaxis, np.newaxis, np.newaxis, ...]
        # 转换为张量
        return torch.tensor(processed, dtype=torch.float32)
    
    def predict(self, frame):
        """预测视角和行走条件"""
        if not self.model_loaded:
            # 模型未加载，返回默认值
            return 'Unknown', 'Unknown'
        
        try:
            # 预处理
            input_tensor = self.preprocess(frame)
            
            # 预测
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            
            with torch.no_grad():
                output = self.model(input_tensor)
                # 分离视角和条件预测
                view_output = output[:, :3]
                condition_output = output[:, 3:]
                
                view_pred = torch.argmax(view_output, dim=1).item()
                condition_pred = torch.argmax(condition_output, dim=1).item()
            
            # 映射到标签
            view = self.view_labels.get(view_pred, 'Unknown')
            condition = self.condition_labels.get(condition_pred, 'Unknown')
            
            return view, condition
        except Exception as e:
            print(f"Error in prediction: {e}")
            return 'Unknown', 'Unknown'