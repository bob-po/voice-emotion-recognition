# 📚 API文档

## 概述

本文档详细介绍了中文语音情绪识别系统的API接口，包括核心类、方法和使用示例。

## 目录

- [EmotionSpeechRecognition类](#emotionspeechrecognition类)
- [EmotionRecognitionWebInterface类](#emotionrecognitionwebinterface类)
- [HubertForSpeechClassification类](#hubertforspeechclassification类)
- [HubertClassificationHead类](#hubertclassificationhead类)
- [使用示例](#使用示例)
- [错误处理](#错误处理)

## EmotionSpeechRecognition类

### 类描述
语音情绪识别的核心类，负责加载模型、预处理音频和进行情绪预测。

### 初始化

```python
EmotionSpeechRecognition(model_name="xmj2002/hubert-base-ch-speech-emotion-recognition")
```

#### 参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | `"xmj2002/hubert-base-ch-speech-emotion-recognition"` | Hugging Face模型名称 |

#### 属性
| 属性 | 类型 | 说明 |
|------|------|------|
| `device` | torch.device | 计算设备（CPU/GPU） |
| `duration` | int | 音频处理时长（秒） |
| `sample_rate` | int | 目标采样率 |
| `config` | AutoConfig | 模型配置 |
| `processor` | Wav2Vec2FeatureExtractor | 特征提取器 |
| `model` | HubertForSpeechClassification | 情绪分类模型 |
| `emotion_labels` | Dict[int, str] | 情绪标签映射 |

### 主要方法

#### `predict_emotion(audio_path: str) -> Dict`

预测音频文件的情绪。

**参数:**
- `audio_path` (str): 音频文件路径

**返回:**
```python
{
    "predicted_emotion": "happy",
    "confidence": 0.892,
    "all_emotions": [["happy", 0.892], ["neutral", 0.067], ...],
    "audio_path": "audio.wav"
}
```

**异常:**
- `FileNotFoundError`: 音频文件不存在
- `ValueError`: 音频格式不支持
- `RuntimeError`: 模型预测失败

**示例:**
```python
recognizer = EmotionSpeechRecognition()
result = recognizer.predict_emotion("audio.wav")
print(f"预测情绪: {result['predicted_emotion']}")
print(f"置信度: {result['confidence']:.3f}")
```

#### `predict_emotion_from_array(audio_array: np.ndarray, sample_rate: int = 16000) -> Dict`

从音频数组预测情绪。

**参数:**
- `audio_array` (np.ndarray): 音频数组，形状为 (samples,) 或 (channels, samples)
- `sample_rate` (int): 采样率，默认16000

**返回:** 同 `predict_emotion`

**示例:**
```python
import numpy as np
audio_data = np.random.randn(16000)  # 1秒音频
result = recognizer.predict_emotion_from_array(audio_data, sample_rate=16000)
```

#### `batch_predict(audio_paths: List[str]) -> List[Dict]`

批量预测多个音频文件的情绪。

**参数:**
- `audio_paths` (List[str]): 音频文件路径列表

**返回:**
```python
[
    {
        "predicted_emotion": "happy",
        "confidence": 0.892,
        "all_emotions": [...],
        "audio_path": "audio1.wav"
    },
    {
        "predicted_emotion": "sad",
        "confidence": 0.756,
        "all_emotions": [...],
        "audio_path": "audio2.wav"
    }
]
```

**示例:**
```python
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = recognizer.batch_predict(audio_files)
for result in results:
    print(f"{result['audio_path']}: {result['predicted_emotion']}")
```

#### `preprocess_audio(audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]`

预处理音频文件。

**参数:**
- `audio_path` (str): 音频文件路径
- `target_sr` (int): 目标采样率，默认16000

**返回:**
- `np.ndarray`: 预处理后的音频数组，形状为 (samples,)
- `None`: 预处理失败

**处理步骤:**
1. 加载音频文件
2. 转换为单声道（如果是立体声）
3. 重采样到目标采样率
4. 返回处理后的数组

**示例:**
```python
audio = recognizer.preprocess_audio("audio.wav", target_sr=16000)
if audio is not None:
    print(f"音频长度: {len(audio)} 采样点")
```

#### `extract_features(audio: np.ndarray) -> Dict[str, torch.Tensor]`

提取音频特征。

**参数:**
- `audio` (np.ndarray): 音频数组

**返回:**
```python
{
    "input_values": torch.Tensor,  # 形状: (1, max_length)
    "attention_mask": torch.Tensor  # 形状: (1, max_length)
}
```

**示例:**
```python
audio = recognizer.preprocess_audio("audio.wav")
features = recognizer.extract_features(audio)
print(f"特征形状: {features['input_values'].shape}")
```

#### `id2class(id: int) -> str`

将ID转换为情绪类别名称。

**参数:**
- `id` (int): 情绪ID

**返回:**
- `str`: 情绪类别名称

**情绪映射:**
```python
{
    0: "angry",      # 愤怒
    1: "fear",       # 恐惧
    2: "happy",      # 快乐
    3: "neutral",    # 中性
    4: "sad",        # 悲伤
    5: "surprise"    # 惊讶
}
```

## EmotionRecognitionWebInterface类

### 类描述
Web界面类，提供Gradio界面的创建和管理。

### 初始化

```python
EmotionRecognitionWebInterface()
```

#### 属性
| 属性 | 类型 | 说明 |
|------|------|------|
| `emotion_recognizer` | EmotionSpeechRecognition | 情绪识别器实例 |
| `emotion_colors` | Dict[str, str] | 情绪颜色映射 |
| `chinese_emotions` | Dict[str, str] | 中英文情绪标签映射 |

### 主要方法

#### `load_model() -> str`

加载情绪识别模型。

**返回:**
- `str`: 加载状态消息

**示例:**
```python
interface = EmotionRecognitionWebInterface()
status = interface.load_model()
print(status)  # "✅ 模型加载成功！" 或 "❌ 模型加载失败: ..."
```

#### `create_emotion_chart(emotion_probs: List[Tuple[str, float]]) -> PIL.Image`

创建情绪概率图表。

**参数:**
- `emotion_probs` (List[Tuple[str, float]]): 情绪概率列表，格式为 [("happy", 0.8), ("sad", 0.2)]

**返回:**
- `PIL.Image`: 情绪分布图表

**示例:**
```python
emotions = [("happy", 0.8), ("sad", 0.2)]
chart = interface.create_emotion_chart(emotions)
chart.save("emotion_chart.png")
```

#### `predict_emotion_from_audio(audio_file: str) -> Tuple[str, PIL.Image, str]`

从音频文件预测情绪。

**参数:**
- `audio_file` (str): 音频文件路径

**返回:**
- `Tuple[str, PIL.Image, str]`: (结果文本, 图表, 音频信息)

**示例:**
```python
result_text, chart, audio_info = interface.predict_emotion_from_audio("audio.wav")
print(result_text)
```

#### `predict_emotion_from_microphone(audio_input: Tuple[int, np.ndarray]) -> Tuple[str, PIL.Image, str]`

从麦克风录音预测情绪。

**参数:**
- `audio_input` (Tuple[int, np.ndarray]): (采样率, 音频数据)

**返回:**
- `Tuple[str, PIL.Image, str]`: (结果文本, 图表, 音频信息)

**示例:**
```python
# Gradio Audio组件返回的数据
sample_rate, audio_data = audio_input
result_text, chart, audio_info = interface.predict_emotion_from_microphone((sample_rate, audio_data))
```

#### `create_interface() -> gr.Blocks`

创建Gradio界面。

**返回:**
- `gr.Blocks`: Gradio界面对象

**示例:**
```python
interface = EmotionRecognitionWebInterface()
app = interface.create_interface()
app.launch(server_name="127.0.0.1", server_port=7860)
```

## 使用示例

### 基本使用

```python
from emotion_speech_recognition import EmotionSpeechRecognition

# 初始化模型
recognizer = EmotionSpeechRecognition()

# 预测单个音频文件
result = recognizer.predict_emotion("audio.wav")
print(f"预测情绪: {result['predicted_emotion']}")
print(f"置信度: {result['confidence']:.3f}")

# 显示所有情绪概率
for emotion, prob in result['all_emotions']:
    print(f"{emotion}: {prob:.3f}")
```

### 批量处理

```python
# 批量预测多个文件
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = recognizer.batch_predict(audio_files)

# 统计情绪分布
emotion_counts = {}
for result in results:
    if 'error' not in result:
        emotion = result['predicted_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

print("情绪分布:")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count} 次")
```

### Web界面使用

```python
from web_interface import EmotionRecognitionWebInterface

# 创建界面
interface = EmotionRecognitionWebInterface()

# 加载模型
status = interface.load_model()
print(status)

# 预测音频文件
result_text, chart, audio_info = interface.predict_emotion_from_audio("audio.wav")
print(result_text)

# 启动Web界面
app = interface.create_interface()
app.launch(server_name="127.0.0.1", server_port=7860)
```

## 错误处理

### 常见错误类型

#### 1. 模型加载错误
```python
try:
    recognizer = EmotionSpeechRecognition()
except Exception as e:
    print(f"模型加载失败: {e}")
    # 检查网络连接和模型名称
```

#### 2. 音频处理错误
```python
result = recognizer.predict_emotion("audio.wav")
if "error" in result:
    print(f"处理失败: {result['error']}")
    # 检查音频文件格式和路径
```

#### 3. 内存不足错误
```python
# 强制使用CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

recognizer = EmotionSpeechRecognition()
recognizer.device = torch.device("cpu")
```

### 错误恢复策略

```python
def safe_predict(recognizer, audio_path):
    """安全的预测函数，包含错误处理"""
    try:
        result = recognizer.predict_emotion(audio_path)
        if "error" in result:
            return {"error": result["error"], "audio_path": audio_path}
        return result
    except Exception as e:
        return {"error": str(e), "audio_path": audio_path}

# 使用安全预测
results = []
for audio_file in audio_files:
    result = safe_predict(recognizer, audio_file)
    results.append(result)

# 统计成功和失败
success_count = len([r for r in results if "error" not in r])
error_count = len([r for r in results if "error" in r])
print(f"成功: {success_count}, 失败: {error_count}")
```

## 性能优化

### GPU加速
```python
# 检查GPU可用性
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("使用CPU")
```

### 批量处理优化
```python
# 使用批处理提高效率
def batch_predict_optimized(recognizer, audio_paths, batch_size=4):
    """优化的批量预测"""
    results = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        batch_results = recognizer.batch_predict(batch)
        results.extend(batch_results)
    return results
```

### 内存管理
```python
# 清理GPU内存
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 使用上下文管理器
with torch.no_grad():
    result = recognizer.predict_emotion("audio.wav")
```

---

**注意**: 本API文档基于当前版本，如有更新请参考最新代码。 