#!/usr/bin/env python3
"""
语音情绪识别系统测试脚本
"""

import os
import sys
import numpy as np
import tempfile
import soundfile as sf
from emotion_speech_recognition import EmotionSpeechRecognition

def create_test_audio(duration=3, sample_rate=16000, frequency=440):
    """
    创建测试音频文件
    
    Args:
        duration: 音频时长（秒）
        sample_rate: 采样率
        frequency: 频率（Hz）
        
    Returns:
        音频文件路径
    """
    # 生成正弦波
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # 添加一些随机噪声模拟真实语音
    noise = np.random.normal(0, 0.01, len(audio))
    audio = audio + noise
    
    # 创建临时文件
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio, sample_rate)
    
    return temp_file.name

def test_model_loading():
    """
    测试模型加载
    """
    print("🧪 测试1: 模型加载")
    try:
        recognizer = EmotionSpeechRecognition()
        print("✅ 模型加载成功")
        return recognizer
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_audio_preprocessing(recognizer):
    """
    测试音频预处理
    """
    print("\n🧪 测试2: 音频预处理")
    try:
        # 创建测试音频
        test_audio_path = create_test_audio()
        print(f"📁 创建测试音频: {test_audio_path}")
        
        # 测试预处理
        audio = recognizer.preprocess_audio(test_audio_path)
        if audio is not None:
            print(f"✅ 音频预处理成功，长度: {len(audio)} 采样点")
        else:
            print("❌ 音频预处理失败")
        
        # 清理临时文件
        os.unlink(test_audio_path)
        return test_audio_path
        
    except Exception as e:
        print(f"❌ 音频预处理测试失败: {e}")
        return None

def test_emotion_prediction(recognizer, audio_path):
    """
    测试情绪预测
    """
    print("\n🧪 测试3: 情绪预测")
    try:
        # 重新创建测试音频（因为之前删除了）
        test_audio_path = create_test_audio()
        
        # 预测情绪
        result = recognizer.predict_emotion(test_audio_path)
        
        if "error" in result:
            print(f"❌ 情绪预测失败: {result['error']}")
            return False
        
        print("✅ 情绪预测成功")
        print(f"🎯 预测情绪: {result['predicted_emotion']}")
        print(f"📊 置信度: {result['confidence']:.3f}")
        
        print("📈 所有情绪概率:")
        for emotion, prob in result['all_emotions']:
            print(f"  - {emotion}: {prob:.3f} ({prob*100:.1f}%)")
        
        # 清理临时文件
        os.unlink(test_audio_path)
        return True
        
    except Exception as e:
        print(f"❌ 情绪预测测试失败: {e}")
        return False

def test_array_prediction(recognizer):
    """
    测试数组预测
    """
    print("\n🧪 测试4: 数组预测")
    try:
        # 创建测试音频数组
        sample_rate = 16000
        duration = 3
        audio_array = np.random.randn(sample_rate * duration) * 0.1
        
        # 预测情绪
        result = recognizer.predict_emotion_from_array(audio_array, sample_rate)
        
        if "error" in result:
            print(f"❌ 数组预测失败: {result['error']}")
            return False
        
        print("✅ 数组预测成功")
        print(f"🎯 预测情绪: {result['predicted_emotion']}")
        print(f"📊 置信度: {result['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数组预测测试失败: {e}")
        return False

def test_batch_prediction(recognizer):
    """
    测试批量预测
    """
    print("\n🧪 测试5: 批量预测")
    try:
        # 创建多个测试音频文件
        audio_paths = []
        for i in range(3):
            audio_path = create_test_audio(duration=2)
            audio_paths.append(audio_path)
        
        # 批量预测
        results = recognizer.batch_predict(audio_paths)
        
        print(f"✅ 批量预测成功，处理了 {len(results)} 个文件")
        
        for i, result in enumerate(results):
            if "error" not in result:
                print(f"  文件 {i+1}: {result['predicted_emotion']} (置信度: {result['confidence']:.3f})")
            else:
                print(f"  文件 {i+1}: 处理失败 - {result['error']}")
        
        # 清理临时文件
        for audio_path in audio_paths:
            os.unlink(audio_path)
        
        return True
        
    except Exception as e:
        print(f"❌ 批量预测测试失败: {e}")
        return False

def test_system_info():
    """
    测试系统信息
    """
    print("\n🧪 测试6: 系统信息")
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA版本: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            print(f"✅ 当前GPU: {torch.cuda.get_device_name()}")
        
        import transformers
        print(f"✅ Transformers版本: {transformers.__version__}")
        
        import librosa
        print(f"✅ Librosa版本: {librosa.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ 系统信息测试失败: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("🎤 语音情绪识别系统测试")
    print("=" * 50)
    
    # 测试计数器
    passed_tests = 0
    total_tests = 6
    
    # 测试1: 模型加载
    recognizer = test_model_loading()
    if recognizer is not None:
        passed_tests += 1
    
    if recognizer is None:
        print("\n❌ 模型加载失败，无法继续测试")
        return
    
    # 测试2: 音频预处理
    audio_path = test_audio_preprocessing(recognizer)
    if audio_path is not None:
        passed_tests += 1
    
    # 测试3: 情绪预测
    if test_emotion_prediction(recognizer, audio_path):
        passed_tests += 1
    
    # 测试4: 数组预测
    if test_array_prediction(recognizer):
        passed_tests += 1
    
    # 测试5: 批量预测
    if test_batch_prediction(recognizer):
        passed_tests += 1
    
    # 测试6: 系统信息
    if test_system_info():
        passed_tests += 1
    
    # 测试总结
    print("\n" + "=" * 50)
    print("📋 测试总结")
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    print(f"❌ 失败测试: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！系统运行正常。")
        print("\n🚀 现在可以运行以下命令启动Web界面:")
        print("   python web_interface.py")
        print("\n💻 或者使用命令行工具:")
        print("   python cli_tool.py -f your_audio.wav")
    else:
        print("⚠️  部分测试失败，请检查安装和配置。")
        print("\n🔧 建议检查:")
        print("   1. 网络连接是否正常")
        print("   2. 依赖包是否正确安装")
        print("   3. 是否有足够的磁盘空间")
        print("   4. Python版本是否兼容")

if __name__ == "__main__":
    main() 