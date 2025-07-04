#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中文语音情绪识别模型
"""

import os
import sys
from emotion_speech_recognition import EmotionSpeechRecognition

def test_model_loading():
    """
    测试模型加载
    """
    print("🔧 正在加载中文语音情绪识别模型...")
    try:
        emotion_recognizer = EmotionSpeechRecognition()
        print("✅ 模型加载成功！")
        return emotion_recognizer
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def test_audio_prediction(emotion_recognizer, audio_path):
    """
    测试音频预测
    
    Args:
        emotion_recognizer: 情绪识别器实例
        audio_path: 音频文件路径
    """
    if not os.path.exists(audio_path):
        print(f"❌ 音频文件不存在: {audio_path}")
        return
    
    print(f"🎵 正在分析音频文件: {audio_path}")
    try:
        result = emotion_recognizer.predict_emotion(audio_path)
        
        if "error" in result:
            print(f"❌ 预测失败: {result['error']}")
            return
        
        print("✅ 预测成功！")
        print(f"🎯 主要情绪: {result['predicted_emotion']}")
        print(f"📊 置信度: {result['confidence']:.3f} ({result['confidence']*100:.1f}%)")
        print("\n📈 所有情绪概率:")
        
        for emotion, prob in result['all_emotions']:
            percentage = prob * 100
            print(f"  - {emotion}: {prob:.3f} ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")

def main():
    """
    主函数
    """
    print("🚀 开始测试中文语音情绪识别模型")
    print("=" * 50)
    
    # 测试模型加载
    emotion_recognizer = test_model_loading()
    if emotion_recognizer is None:
        print("❌ 无法继续测试，模型加载失败")
        return
    
    print("\n" + "=" * 50)
    
    # 测试音频文件（如果有的话）
    test_audio_path = "C:/Users/guoji/Music/天不生我毋忠良-气势.wav"
    
    if os.path.exists(test_audio_path):
        test_audio_prediction(emotion_recognizer, test_audio_path)
    else:
        print("⚠️  没有找到测试音频文件，跳过音频预测测试")
        print(f"   预期路径: {test_audio_path}")
        print("💡 你可以将音频文件放在上述路径，然后重新运行测试")
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！")

if __name__ == "__main__":
    main() 