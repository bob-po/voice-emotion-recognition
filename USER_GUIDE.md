# 📖 用户指南

## 概述

本指南将帮助您快速上手中文语音情绪识别系统，包括安装、配置、使用和故障排除。

## 目录

- [快速开始](#快速开始)
- [详细安装](#详细安装)
- [Web界面使用](#web界面使用)
- [命令行使用](#命令行使用)
- [Python API使用](#python-api使用)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)
- [故障排除](#故障排除)

## 快速开始

### 1. 环境准备

确保您的系统满足以下要求：
- Python 3.8 或更高版本
- 至少 4GB 内存
- 网络连接（用于下载模型）
- 可选：NVIDIA GPU（用于加速）

### 2. 安装系统

```bash
# 克隆项目
git clone https://github.com/example/voice-emotion-recognition.git
cd voice-emotion-recognition

# 创建虚拟环境
python -m venv voice-env
source voice-env/bin/activate  # Linux/Mac
# voice-env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 首次运行

```bash
# 启动Web界面
python web_interface.py
```

在浏览器中打开 `http://localhost:7860`，您将看到语音情绪识别系统的界面。

## 详细安装

### Windows 安装

#### 方法1: 使用Anaconda（推荐）

```bash
# 安装Anaconda后，打开Anaconda Prompt
conda create -n voice-emotion python=3.9
conda activate voice-emotion

# 安装PyTorch
conda install pytorch torchaudio -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

#### 方法2: 使用Python虚拟环境

```bash
# 确保已安装Python 3.8+
python --version

# 创建虚拟环境
python -m venv voice-env
voice-env\Scripts\activate

# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

### macOS 安装

```bash
# 使用Homebrew安装Python（如果需要）
brew install python@3.9

# 创建虚拟环境
python3 -m venv voice-env
source voice-env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### Linux 安装

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# 创建虚拟环境
python3 -m venv voice-env
source voice-env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 验证安装

```bash
# 检查Python版本
python --version

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 检查CUDA（如果有GPU）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查其他依赖
python -c "import transformers, librosa, gradio; print('所有依赖安装成功')"
```

## Web界面使用

### 界面概览

Web界面包含以下主要区域：
1. **模型加载区域**: 加载和检查模型状态
2. **输入区域**: 上传音频文件或录制音频
3. **结果区域**: 显示预测结果和可视化图表

### 使用步骤

#### 步骤1: 加载模型

1. 点击"🚀 加载模型"按钮
2. 等待模型下载和加载完成
3. 查看模型状态显示"✅ 模型加载成功！"

**注意**: 首次运行需要下载模型，可能需要几分钟时间。

#### 步骤2: 选择输入方式

**方式A: 上传音频文件**
1. 点击"📁 上传音频文件"标签页
2. 点击文件选择区域
3. 选择支持的音频文件（WAV、MP3、M4A等）

**方式B: 实时录音**
1. 点击"🎤 实时录音"标签页
2. 点击录音按钮开始录制
3. 说话后点击停止按钮

#### 步骤3: 分析音频

1. 点击"🔍 分析音频文件"或"🔍 分析录音"按钮
2. 等待处理完成
3. 查看预测结果

#### 步骤4: 查看结果

结果包含以下信息：
- **主要情绪**: 预测的主要情绪类型
- **置信度**: 预测的置信度百分比
- **所有情绪概率**: 每种情绪的概率分布
- **情绪分布图**: 可视化的概率图表

### 界面功能详解

#### 模型信息区域
- 显示使用的模型名称
- 支持的音频格式
- 技术规格信息

#### 音频信息显示
- 显示音频文件路径或录音信息
- 音频处理状态

#### 结果展示
- 中英文情绪标签
- 置信度百分比
- 颜色编码的情绪图表

## 命令行使用

### 基本命令

#### 分析单个音频文件
```bash
python cli_tool.py -f audio.wav
```

#### 批量分析多个文件
```bash
python cli_tool.py -f audio1.wav audio2.wav audio3.wav
```

#### 详细输出模式
```bash
python cli_tool.py -f audio.wav -v
```

#### 保存结果到文件
```bash
python cli_tool.py -f audio.wav -o result.json
```

### 命令行参数

| 参数 | 短参数 | 说明 | 示例 |
|------|--------|------|------|
| `--files` | `-f` | 音频文件路径（必需） | `-f audio.wav` |
| `--output` | `-o` | 输出JSON文件路径 | `-o result.json` |
| `--verbose` | `-v` | 详细输出模式 | `-v` |
| `--model` | `-m` | 指定模型名称 | `-m model_name` |

### 输出示例

```bash
🚀 正在加载模型...
✅ 模型加载完成

📁 处理文件 1/1: audio.wav
🎯 预测情绪: happy
📊 置信度: 0.892 (89.2%)

📈 所有情绪概率:
  - happy: 0.892 (89.2%)
  - neutral: 0.067 (6.7%)
  - surprise: 0.023 (2.3%)
  - sad: 0.012 (1.2%)
  - angry: 0.004 (0.4%)
  - fear: 0.002 (0.2%)

💾 结果已保存到: result.json

📋 处理总结:
  - 总文件数: 1
  - 成功处理: 1
  - 处理失败: 0
```

### 批量处理脚本

创建批量处理脚本 `batch_process.sh`:

```bash
#!/bin/bash
# 批量处理音频文件

# 设置输入目录
INPUT_DIR="audio_files"
OUTPUT_DIR="results"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 处理所有WAV文件
for file in $INPUT_DIR/*.wav; do
    if [ -f "$file" ]; then
        echo "处理文件: $file"
        python cli_tool.py -f "$file" -o "$OUTPUT_DIR/$(basename "$file" .wav).json"
    fi
done

echo "批量处理完成！"
```

## Python API使用

### 基本使用

```python
from emotion_speech_recognition import EmotionSpeechRecognition

# 初始化模型
recognizer = EmotionSpeechRecognition()

# 预测单个文件
result = recognizer.predict_emotion("audio.wav")
print(f"预测情绪: {result['predicted_emotion']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 批量处理

```python
# 批量预测
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
results = recognizer.batch_predict(audio_files)

# 分析结果
for result in results:
    if 'error' not in result:
        print(f"{result['audio_path']}: {result['predicted_emotion']} ({result['confidence']:.3f})")
```

### 实时处理

```python
import numpy as np
import sounddevice as sd

# 录制音频
duration = 5  # 录制5秒
sample_rate = 16000
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

# 预测情绪
result = recognizer.predict_emotion_from_array(audio_data.flatten(), sample_rate)
print(f"实时预测: {result['predicted_emotion']}")
```

### 自定义处理

```python
# 预处理音频
audio = recognizer.preprocess_audio("audio.wav", target_sr=16000)
if audio is not None:
    # 提取特征
    features = recognizer.extract_features(audio)
    
    # 手动预测
    with torch.no_grad():
        logits = recognizer.model(features["input_values"])
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits).item()
    
    print(f"预测类别: {recognizer.id2class(predicted_class)}")
```

## 最佳实践

### 音频质量要求

#### 推荐的音频规格
- **格式**: WAV（无损）或高质量MP3
- **采样率**: 16kHz或更高
- **声道**: 单声道（系统会自动转换）
- **时长**: 3-10秒
- **质量**: 清晰语音，最小背景噪音

#### 音频准备技巧
1. **环境**: 在安静环境中录制
2. **距离**: 麦克风距离嘴部15-20厘米
3. **音量**: 保持适中的音量水平
4. **清晰度**: 发音清晰，语速适中

### 情绪表达建议

#### 提高识别准确率
1. **情绪强度**: 情绪表达越明显，识别越准确
2. **语音特征**: 注意语调、语速、音量变化
3. **语言**: 使用中文语音，系统针对中文优化
4. **内容**: 情绪相关的内容有助于识别

#### 情绪表达示例
- **快乐**: 语调上扬，语速较快，音量较大
- **悲伤**: 语调低沉，语速较慢，音量较小
- **愤怒**: 语调尖锐，语速快，音量很大
- **恐惧**: 语调颤抖，语速不稳定，音量变化大
- **惊讶**: 语调突然变化，语速快，音量突然增大
- **中性**: 语调平稳，语速适中，音量稳定

### 性能优化

#### 系统优化
1. **GPU加速**: 如果有NVIDIA GPU，系统会自动使用
2. **内存管理**: 处理大文件时注意内存使用
3. **批量处理**: 使用命令行工具进行批量处理
4. **模型缓存**: 首次运行后模型会缓存，后续运行更快

#### 代码优化
```python
# 使用上下文管理器减少内存使用
with torch.no_grad():
    result = recognizer.predict_emotion("audio.wav")

# 批量处理提高效率
results = recognizer.batch_predict(audio_files)

# 清理GPU内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 常见问题

### Q1: 模型加载失败怎么办？

**A**: 检查以下几点：
1. 网络连接是否正常
2. 是否有足够的磁盘空间
3. 防火墙是否阻止了下载
4. 尝试使用代理或VPN

### Q2: 音频文件无法识别怎么办？

**A**: 检查音频文件：
1. 文件格式是否支持
2. 文件是否损坏
3. 文件路径是否正确
4. 尝试转换为WAV格式

### Q3: 预测结果不准确怎么办？

**A**: 尝试以下方法：
1. 使用更清晰的音频
2. 确保情绪表达明显
3. 使用中文语音
4. 调整音频长度（3-10秒）

### Q4: 系统运行缓慢怎么办？

**A**: 优化建议：
1. 使用GPU加速
2. 减少同时处理的文件数量
3. 关闭其他占用资源的程序
4. 增加系统内存

### Q5: 如何批量处理大量文件？

**A**: 使用命令行工具：
```bash
# 创建批处理脚本
python cli_tool.py -f *.wav -o results.json

# 或使用Python脚本
import glob
audio_files = glob.glob("*.wav")
results = recognizer.batch_predict(audio_files)
```

## 故障排除

### 错误诊断

#### 1. 模型加载错误
```bash
# 检查网络连接
ping huggingface.co

# 检查磁盘空间
df -h

# 检查Python环境
python -c "import transformers; print('OK')"
```

#### 2. 音频处理错误
```bash
# 检查音频文件
file audio.wav

# 转换音频格式
ffmpeg -i input.mp3 output.wav

# 检查音频信息
python -c "
import librosa
audio, sr = librosa.load('audio.wav')
print(f'采样率: {sr}, 长度: {len(audio)/sr:.2f}秒')
"
```

#### 3. 内存不足错误
```bash
# 检查内存使用
free -h

# 强制使用CPU
export CUDA_VISIBLE_DEVICES=""
python web_interface.py
```

#### 4. 依赖安装错误
```bash
# 升级pip
pip install --upgrade pip

# 重新安装依赖
pip uninstall torch torchaudio
pip install torch torchaudio

# 使用conda安装
conda install pytorch torchaudio -c pytorch
```

### 调试模式

#### 启用详细日志
```bash
# 设置环境变量
export PYTHONPATH=.
export TRANSFORMERS_VERBOSITY=info

# 运行程序
python web_interface.py --debug
```

#### 检查系统状态
```python
# 检查PyTorch和CUDA
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")

# 检查模型状态
from emotion_speech_recognition import EmotionSpeechRecognition
recognizer = EmotionSpeechRecognition()
print(f"设备: {recognizer.device}")
print(f"模型: {recognizer.model}")
```

### 性能监控

#### 监控资源使用
```bash
# 监控CPU和内存
htop

# 监控GPU使用（如果有）
nvidia-smi

# 监控磁盘I/O
iotop
```

#### 性能测试
```python
import time
from emotion_speech_recognition import EmotionSpeechRecognition

# 性能测试
recognizer = EmotionSpeechRecognition()
start_time = time.time()
result = recognizer.predict_emotion("test_audio.wav")
end_time = time.time()

print(f"处理时间: {end_time - start_time:.2f}秒")
print(f"预测结果: {result['predicted_emotion']}")
```

---

**注意**: 本指南基于当前版本，如有更新请参考最新文档。如有问题，请查看故障排除部分或联系技术支持。 