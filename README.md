# 🎤 中文语音情绪识别系统

基于Hugging Face中文预训练模型的语音情绪识别系统，支持6种情绪分类：愤怒、恐惧、快乐、悲伤、惊讶和中性。

## ✨ 功能特点

- 🚀 **预训练模型**: 使用Hugging Face上的高质量中文预训练模型
- 🎯 **6种情绪**: 支持愤怒、恐惧、快乐、悲伤、惊讶、中性
- 📁 **多种输入**: 支持音频文件上传和实时录音
- 🌐 **Web界面**: 提供友好的Gradio Web界面
- 💻 **命令行工具**: 支持批量处理和脚本调用
- 📊 **可视化**: 提供情绪概率分布图表
- 🔧 **自动预处理**: 自动音频格式转换和重采样
- 🎨 **中文界面**: 完全中文化的用户界面

## 📋 系统要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）
- 至少4GB内存
- 网络连接（首次运行需要下载模型）

## 🛠️ 安装步骤

### 1. 克隆项目
```bash
git clone https://github.com/example/voice-emotion-recognition.git
cd voice-emotion-recognition
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用conda
conda create -n voice-emotion python=3.9
conda activate voice-emotion

# 或使用venv
python -m venv voice-emotion
source voice-emotion/bin/activate  # Linux/Mac
# voice-emotion\Scripts\activate  # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
```

## 🚀 使用方法

### 方法1: Web界面（推荐）

启动Web界面：
```bash
python web_interface.py
```

然后在浏览器中打开 `http://localhost:7860`

**使用步骤：**
1. 点击"🚀 加载模型"按钮
2. 选择"📁 上传音频文件"或"🎤 实时录音"标签页
3. 上传音频文件或录制音频
4. 点击"🔍 分析"按钮
5. 查看预测结果和情绪分布图

### 方法2: 命令行工具

#### 分析单个音频文件
```bash
python cli_tool.py -f audio.wav
```

#### 批量分析多个音频文件
```bash
python cli_tool.py -f audio1.wav audio2.wav audio3.wav
```

#### 详细输出模式
```bash
python cli_tool.py -f audio.wav -v
```

#### 保存结果到JSON文件
```bash
python cli_tool.py -f audio.wav -o result.json
```

### 方法3: Python API

```python
from emotion_speech_recognition import EmotionSpeechRecognition

# 初始化模型
recognizer = EmotionSpeechRecognition()

# 预测音频文件
result = recognizer.predict_emotion("audio.wav")
print(f"预测情绪: {result['predicted_emotion']}")
print(f"置信度: {result['confidence']:.3f}")

# 从音频数组预测
import numpy as np
audio_array = np.random.randn(16000)  # 示例音频数据
result = recognizer.predict_emotion_from_array(audio_array, sample_rate=16000)
```

## 📁 项目结构

```
voice-emotion-recognition/
├── requirements.txt              # 项目依赖
├── emotion_speech_recognition.py # 核心识别类
├── web_interface.py             # Web界面
├── cli_tool.py                  # 命令行工具
├── run.py                       # 启动脚本
├── test_chinese_model.py        # 中文模型测试
├── test_system.py               # 系统测试
├── emotion_chart.png            # 示例图表
└── README.md                    # 项目说明
```

## 🎯 支持的音频格式

- **输入格式**: WAV, MP3, M4A, FLAC, OGG等
- **采样率**: 自动重采样到16kHz
- **声道**: 自动转换为单声道
- **时长**: 建议3-10秒，支持更长音频

## 🔧 模型信息

- **模型名称**: `xmj2002/hubert-base-ch-speech-emotion-recognition`
- **架构**: Hubert + Transformer
- **训练数据**: CASIA中文语音数据集
- **情绪类别**: 6种（angry, fear, happy, sad, surprise, neutral）
- **语言**: 专门针对中文语音优化

## 📊 输出格式

### 预测结果示例

```json
{
  "predicted_emotion": "happy",
  "confidence": 0.892,
  "all_emotions": [
    ["happy", 0.892],
    ["neutral", 0.067],
    ["surprise", 0.023],
    ["sad", 0.012],
    ["angry", 0.004],
    ["fear", 0.002]
  ],
  "audio_path": "audio.wav"
}
```

## 🎨 情绪标签说明

| 英文标签 | 中文标签 | 描述 | 颜色 |
|---------|---------|------|------|
| angry | 愤怒 | 生气、愤怒的情绪 | 🔴 红色 |
| fear | 恐惧 | 害怕、恐惧的情绪 | 🟣 紫色 |
| happy | 快乐 | 开心、快乐的情绪 | 🟡 金色 |
| sad | 悲伤 | 伤心、悲伤的情绪 | 🔵 蓝色 |
| surprise | 惊讶 | 惊讶、意外的情绪 | 🟠 橙色 |
| neutral | 中性 | 平静、中性的情绪 | ⚪ 灰色 |

## 💡 使用建议

### 提高识别准确率

1. **音频质量**: 使用清晰的语音，避免背景噪音
2. **情绪表达**: 情绪表达越明显，识别准确率越高
3. **音频长度**: 建议3-10秒的语音片段
4. **语言**: 专门针对中文语音优化，支持中文语音识别

### 性能优化

1. **GPU加速**: 如果有NVIDIA GPU，会自动使用CUDA加速
2. **批量处理**: 使用命令行工具进行批量处理
3. **模型缓存**: 首次运行会下载模型，后续运行会使用缓存

## 🔍 故障排除

### 常见问题

#### 1. 模型加载失败
**问题**: `模型加载失败: ConnectionError`
**解决方案**: 检查网络连接，确保可以访问huggingface.co

#### 2. CUDA内存不足
**问题**: `CUDA out of memory`
**解决方案**: 强制使用CPU或减少批处理大小

#### 3. 音频格式不支持
**问题**: `音频预处理错误: Unsupported format`
**解决方案**: 转换为WAV格式或使用支持的音频格式

#### 4. 依赖安装失败
**问题**: `pip install 失败`
**解决方案**: 升级pip，安装编译工具，或使用conda安装

### 调试模式
```bash
# 启用详细日志
python web_interface.py --debug

# 检查模型状态
python -c "
from emotion_speech_recognition import EmotionSpeechRecognition
recognizer = EmotionSpeechRecognition()
print(f'设备: {recognizer.device}')
print(f'模型: {recognizer.model}')
"
```

## 📚 API文档

### EmotionSpeechRecognition类

#### 主要方法

##### `predict_emotion(audio_path: str) -> Dict`
预测音频文件的情绪

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

##### `predict_emotion_from_array(audio_array: np.ndarray, sample_rate: int = 16000) -> Dict`
从音频数组预测情绪

**参数:**
- `audio_array` (np.ndarray): 音频数组
- `sample_rate` (int): 采样率，默认16000

##### `batch_predict(audio_paths: List[str]) -> List[Dict]`
批量预测多个音频文件

**参数:**
- `audio_paths` (List[str]): 音频文件路径列表

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [Hugging Face](https://huggingface.co/) - 提供预训练模型
- [CASIA](http://www.casia.cn/) - 中文语音数据集
- [Gradio](https://gradio.app/) - Web界面框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

**注意**: 本项目仅供学习和研究使用，请勿用于商业用途。 
