# 🏗️ 技术架构文档

## 概述

本文档详细介绍了中文语音情绪识别系统的技术架构、设计原理和实现细节。

## 目录

- [系统架构](#系统架构)
- [模型架构](#模型架构)
- [数据流程](#数据流程)
- [技术栈](#技术栈)
- [性能优化](#性能优化)
- [扩展性设计](#扩展性设计)
- [安全考虑](#安全考虑)

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    用户界面层                                │
├─────────────────────────────────────────────────────────────┤
│  Web界面 (Gradio)  │  命令行工具  │  Python API              │
├─────────────────────────────────────────────────────────────┤
│                    业务逻辑层                                │
├─────────────────────────────────────────────────────────────┤
│  EmotionRecognitionWebInterface  │  EmotionSpeechRecognition │
├─────────────────────────────────────────────────────────────┤
│                    模型层                                    │
├─────────────────────────────────────────────────────────────┤
│  HubertForSpeechClassification  │  HubertClassificationHead │
├─────────────────────────────────────────────────────────────┤
│                    特征提取层                                │
├─────────────────────────────────────────────────────────────┤
│  Wav2Vec2FeatureExtractor  │  Audio Preprocessing           │
├─────────────────────────────────────────────────────────────┤
│                    基础模型层                                │
├─────────────────────────────────────────────────────────────┤
│  Hubert Model  │  Transformers  │  PyTorch                   │
├─────────────────────────────────────────────────────────────┤
│                    系统层                                    │
├─────────────────────────────────────────────────────────────┤
│  CUDA/CPU  │  Audio Libraries  │  File System                │
└─────────────────────────────────────────────────────────────┘
```

### 模块职责

#### 1. 用户界面层
- **Web界面**: 提供友好的Web交互界面
- **命令行工具**: 支持批量处理和脚本调用
- **Python API**: 提供编程接口

#### 2. 业务逻辑层
- **EmotionRecognitionWebInterface**: Web界面业务逻辑
- **EmotionSpeechRecognition**: 核心识别业务逻辑

#### 3. 模型层
- **HubertForSpeechClassification**: 语音分类模型
- **HubertClassificationHead**: 分类头实现

#### 4. 特征提取层
- **Wav2Vec2FeatureExtractor**: 音频特征提取
- **Audio Preprocessing**: 音频预处理

#### 5. 基础模型层
- **Hubert Model**: 基础语音模型
- **Transformers**: 模型框架
- **PyTorch**: 深度学习框架

## 模型架构

### Hubert模型架构

```
输入音频 → 特征提取 → Hubert编码器 → 分类头 → 情绪输出
    ↓           ↓           ↓           ↓         ↓
  16kHz    Mel频谱    隐藏状态     全连接层    6种情绪
```

#### 详细架构

```
┌─────────────────────────────────────────────────────────────┐
│                        输入层                                │
│  Audio Input (16kHz, mono) → Feature Extraction            │
└─────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│                    Hubert编码器                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Conv1D    │  │ Transformer │  │ Transformer │         │
│  │   Layers    │  │   Blocks    │  │   Blocks    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓               ↓               ↓                  │
│  Feature Maps → Hidden States → Hidden States             │
└─────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│                      分类头                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Mean      │  │   Dense     │  │   Output    │         │
│  │  Pooling    │  │   Layer     │  │   Layer     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│         ↓               ↓               ↓                  │
│  Global Rep → Hidden Rep → Emotion Logits                 │
└─────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│                        输出层                                │
│  Softmax → Emotion Probabilities → Predicted Emotion       │
└─────────────────────────────────────────────────────────────┘
```

### 模型参数

| 参数 | 值 | 说明 |
|------|----|----|
| 输入采样率 | 16kHz | 音频输入采样率 |
| 输入长度 | 6秒 | 最大音频处理长度 |
| 隐藏层维度 | 768 | Hubert隐藏状态维度 |
| 注意力头数 | 12 | Transformer注意力头数 |
| 层数 | 12 | Transformer层数 |
| 输出类别 | 6 | 情绪分类数量 |
| 词汇表大小 | 32,000 | 子词词汇表大小 |

### 情绪分类映射

```python
EMOTION_MAPPING = {
    0: "angry",      # 愤怒
    1: "fear",       # 恐惧  
    2: "happy",      # 快乐
    3: "neutral",    # 中性
    4: "sad",        # 悲伤
    5: "surprise"    # 惊讶
}
```

## 数据流程

### 音频处理流程

```
原始音频 → 预处理 → 特征提取 → 模型推理 → 后处理 → 结果输出
    ↓         ↓         ↓         ↓         ↓         ↓
  任意格式   标准化    Mel频谱    Logits    Softmax   情绪标签
```

#### 1. 音频预处理

```python
def preprocess_audio(audio_path, target_sr=16000):
    # 1. 加载音频
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # 2. 转换为单声道
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 3. 重采样
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    return audio
```

#### 2. 特征提取

```python
def extract_features(audio):
    # 使用Wav2Vec2特征提取器
    inputs = processor(
        audio,
        padding="max_length",
        truncation=True,
        max_length=duration * sample_rate,
        return_tensors="pt",
        sampling_rate=sample_rate
    )
    return inputs
```

#### 3. 模型推理

```python
def predict_emotion(audio_path):
    # 1. 预处理
    audio = preprocess_audio(audio_path)
    
    # 2. 特征提取
    inputs = extract_features(audio)
    
    # 3. 模型推理
    with torch.no_grad():
        logits = model(inputs["input_values"])
        probabilities = F.softmax(logits, dim=1)
    
    # 4. 后处理
    predicted_class = torch.argmax(logits).item()
    confidence = probabilities[0][predicted_class].item()
    
    return {
        "predicted_emotion": id2class(predicted_class),
        "confidence": confidence,
        "all_emotions": get_all_emotions(probabilities)
    }
```

### Web界面数据流

```
用户操作 → 前端界面 → 后端处理 → 模型推理 → 结果展示
    ↓         ↓         ↓         ↓         ↓
  上传音频   Gradio    Python    PyTorch   可视化
```

## 技术栈

### 核心技术

| 技术 | 版本 | 用途 |
|------|----|----|
| Python | 3.8+ | 主要编程语言 |
| PyTorch | 1.9+ | 深度学习框架 |
| Transformers | 4.20+ | 预训练模型框架 |
| Gradio | 3.0+ | Web界面框架 |
| librosa | 0.9+ | 音频处理库 |
| soundfile | 0.10+ | 音频文件读写 |

### 音频处理

| 库 | 用途 |
|----|----|
| librosa | 音频加载、重采样、特征提取 |
| soundfile | 高质量音频文件读写 |
| torchaudio | PyTorch音频处理 |
| numpy | 数值计算 |

### 深度学习

| 组件 | 说明 |
|----|----|
| Hubert | 基础语音模型 |
| Wav2Vec2 | 特征提取器 |
| Transformer | 注意力机制 |
| Softmax | 分类输出 |

### Web框架

| 组件 | 说明 |
|----|----|
| Gradio | 快速Web界面构建 |
| HTML/CSS | 界面样式 |
| JavaScript | 交互逻辑 |

## 性能优化

### 计算优化

#### 1. GPU加速
```python
# 自动设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 批量处理
inputs = {k: v.to(device) for k, v in inputs.items()}
```

#### 2. 内存优化
```python
# 使用上下文管理器
with torch.no_grad():
    logits = model(inputs["input_values"])

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

#### 3. 批处理优化
```python
def batch_predict(audio_paths, batch_size=4):
    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

### 音频处理优化

#### 1. 并行处理
```python
import multiprocessing as mp

def parallel_process(audio_files):
    with mp.Pool() as pool:
        results = pool.map(predict_emotion, audio_files)
    return results
```

#### 2. 缓存机制
```python
# 模型缓存
@lru_cache(maxsize=1)
def load_model():
    return EmotionSpeechRecognition()

# 特征缓存
feature_cache = {}
```

## 扩展性设计

### 模块化设计

#### 1. 插件化架构
```python
class EmotionRecognizerPlugin:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def predict(self, audio):
        raise NotImplementedError
    
    def get_supported_emotions(self):
        raise NotImplementedError

class HubertEmotionRecognizer(EmotionRecognizerPlugin):
    def predict(self, audio):
        # Hubert模型实现
        pass
```

#### 2. 配置驱动
```python
CONFIG = {
    "models": {
        "hubert": {
            "name": "xmj2002/hubert-base-ch-speech-emotion-recognition",
            "type": "hubert",
            "emotions": ["angry", "fear", "happy", "neutral", "sad", "surprise"]
        }
    },
    "audio": {
        "sample_rate": 16000,
        "duration": 6,
        "channels": 1
    }
}
```

### 多模型支持

#### 1. 模型工厂
```python
class ModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == "hubert":
            return HubertEmotionRecognizer()
        elif model_type == "wav2vec2":
            return Wav2Vec2EmotionRecognizer()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
```

#### 2. 模型组合
```python
class EnsembleEmotionRecognizer:
    def __init__(self, models):
        self.models = models
    
    def predict(self, audio):
        predictions = []
        for model in self.models:
            pred = model.predict(audio)
            predictions.append(pred)
        
        # 集成预测结果
        return self.ensemble_predictions(predictions)
```

## 安全考虑

### 数据安全

#### 1. 音频数据保护
```python
# 临时文件处理
import tempfile
import os

def secure_audio_processing(audio_data):
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        temp_file.write(audio_data)
        temp_file.flush()
        result = process_audio(temp_file.name)
    return result
```

#### 2. 模型安全
```python
# 模型完整性检查
import hashlib

def verify_model_integrity(model_path):
    expected_hash = "expected_model_hash"
    with open(model_path, 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    return actual_hash == expected_hash
```

### 隐私保护

#### 1. 数据匿名化
```python
def anonymize_audio(audio_data):
    # 移除音频元数据
    # 添加噪声
    # 时间戳模糊化
    return processed_audio
```

#### 2. 本地处理
```python
# 支持本地模型，不依赖网络
class LocalEmotionRecognizer:
    def __init__(self, local_model_path):
        self.model = load_local_model(local_model_path)
    
    def predict(self, audio):
        # 完全本地处理
        return self.model(audio)
```

## 监控和日志

### 性能监控

```python
import time
import logging

class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def monitor_prediction(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            self.logger.info(f"Prediction time: {end_time - start_time:.2f}s")
            return result
        return wrapper
```

### 错误处理

```python
class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def handle_prediction_error(self, error, audio_path):
        self.logger.error(f"Prediction failed for {audio_path}: {error}")
        
        if isinstance(error, FileNotFoundError):
            return {"error": "Audio file not found"}
        elif isinstance(error, ValueError):
            return {"error": "Invalid audio format"}
        else:
            return {"error": "Internal server error"}
```

---

**注意**: 本技术架构文档基于当前版本，如有更新请参考最新代码。 