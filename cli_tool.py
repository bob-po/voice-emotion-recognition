#!/usr/bin/env python3
"""
语音情绪识别命令行工具
"""

import argparse
import os
import sys
from emotion_speech_recognition import EmotionSpeechRecognition
import json

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description="语音情绪识别命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python cli_tool.py -f audio.wav                    # 分析单个音频文件
  python cli_tool.py -f audio1.wav audio2.wav        # 批量分析多个音频文件
  python cli_tool.py -f audio.wav -o result.json     # 保存结果到JSON文件
  python cli_tool.py -f audio.wav -v                 # 详细输出模式
        """
    )
    
    parser.add_argument(
        "-f", "--files",
        nargs="+",
        required=True,
        help="音频文件路径（支持多个文件）"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="输出JSON文件路径"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="harshit345/xlsr-wav2vec-speech-emotion-recognition",
        help="Hugging Face模型名称"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for file_path in args.files:
        if not os.path.exists(file_path):
            print(f"❌ 错误: 文件不存在 - {file_path}")
            sys.exit(1)
    
    try:
        # 初始化模型
        print("🚀 正在加载模型...")
        emotion_recognizer = EmotionSpeechRecognition(args.model)
        print("✅ 模型加载完成")
        
        results = []
        
        # 处理每个音频文件
        for i, file_path in enumerate(args.files, 1):
            print(f"\n📁 处理文件 {i}/{len(args.files)}: {file_path}")
            
            # 预测情绪
            result = emotion_recognizer.predict_emotion(file_path)
            
            if "error" in result:
                print(f"❌ 处理失败: {result['error']}")
                results.append({
                    "file": file_path,
                    "error": result["error"]
                })
                continue
            
            # 显示结果
            predicted_emotion = result["predicted_emotion"]
            confidence = result["confidence"]
            
            print(f"🎯 预测情绪: {predicted_emotion}")
            print(f"📊 置信度: {confidence:.3f} ({confidence*100:.1f}%)")
            
            if args.verbose:
                print("📈 所有情绪概率:")
                for emotion, prob in result["all_emotions"]:
                    print(f"  - {emotion}: {prob:.3f} ({prob*100:.1f}%)")
            
            results.append(result)
        
        # 保存结果到JSON文件
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\n💾 结果已保存到: {args.output}")
            except Exception as e:
                print(f"❌ 保存结果失败: {e}")
        
        # 显示总结
        print(f"\n📋 处理总结:")
        print(f"  - 总文件数: {len(args.files)}")
        print(f"  - 成功处理: {len([r for r in results if 'error' not in r])}")
        print(f"  - 处理失败: {len([r for r in results if 'error' in r])}")
        
        if len(results) > 1:
            print(f"\n🏆 情绪分布:")
            emotion_counts = {}
            for result in results:
                if 'error' not in result:
                    emotion = result['predicted_emotion']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len([r for r in results if 'error' not in r])) * 100
                print(f"  - {emotion}: {count} 次 ({percentage:.1f}%)")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"❌ 程序执行错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 