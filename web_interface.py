import gradio as gr
import numpy as np
import tempfile
import os
from emotion_speech_recognition import EmotionSpeechRecognition
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
# import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号

class EmotionRecognitionWebInterface:
    """
    语音情绪识别Web界面
    """
    
    def __init__(self):
        """
        初始化Web界面
        """
        self.emotion_recognizer = None
        self.emotion_colors = {
            "angry": "#FF6B6B",      # 红色
            "fear": "#9370DB",       # 紫色
            "happy": "#FFD700",      # 金色
            "sad": "#4682B4",        # 钢蓝色
            "surprise": "#FF69B4",   # 粉红色
            "neutral": "#808080"     # 灰色
        }
        
        # 中文情绪标签
        self.chinese_emotions = {
            "angry": "愤怒",
            "fear": "恐惧",
            "happy": "快乐",
            "sad": "悲伤",
            "surprise": "惊讶",
            "neutral": "中性"
        }
    
    def load_model(self):
        """
        加载模型
        """
        if self.emotion_recognizer is None:
            try:
                self.emotion_recognizer = EmotionSpeechRecognition()
                return "✅ 模型加载成功！"
            except Exception as e:
                return f"❌ 模型加载失败: {str(e)}"
        return "✅ 模型已加载"
    
    def create_emotion_chart(self, emotion_probs):
        """
        创建情绪概率图表
        
        Args:
            emotion_probs: 情绪概率列表
            
        Returns:
            PIL.Image 对象
        """
        emotions = [item[0] for item in emotion_probs]
        probabilities = [item[1] for item in emotion_probs]
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        colors = [self.emotion_colors.get(emotion, "#808080") for emotion in emotions]
        
        bars = plt.bar(emotions, probabilities, color=colors, alpha=0.8)
        plt.xlabel('情绪类型', fontsize=12)
        plt.ylabel('概率', fontsize=12)
        plt.title('语音情绪识别结果', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表到内存并返回PIL.Image对象
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        return img
    
    def predict_emotion_from_audio(self, audio_file):
        """
        从音频文件预测情绪
        
        Args:
            audio_file: 上传的音频文件路径
            
        Returns:
            预测结果
        """
        if self.emotion_recognizer is None:
            return "请先加载模型", None, None
        
        if audio_file is None:
            return "请上传音频文件", None, None
        
        try:
            # 预测情绪 - audio_file 是字符串路径
            result = self.emotion_recognizer.predict_emotion(audio_file)
            
            if "error" in result:
                return f"❌ 预测失败: {result['error']}", None, None
            
            # 获取预测结果
            predicted_emotion = result["predicted_emotion"]
            confidence = result["confidence"]
            all_emotions = result["all_emotions"]
            
            # 创建结果文本
            chinese_emotion = self.chinese_emotions.get(predicted_emotion, predicted_emotion)
            result_text = f"""
## 🎯 预测结果

**主要情绪**: {chinese_emotion} ({predicted_emotion})
**置信度**: {confidence:.3f} ({confidence*100:.1f}%)

## 📊 所有情绪概率

"""
            
            for emotion, prob in all_emotions:
                chinese_name = self.chinese_emotions.get(emotion, emotion)
                percentage = prob * 100
                result_text += f"- **{chinese_name}** ({emotion}): {prob:.3f} ({percentage:.1f}%)\n"
            
            # 创建图表
            chart_path = self.create_emotion_chart(all_emotions)
            
            return result_text, chart_path, audio_file
            
        except Exception as e:
            return f"❌ 处理过程中出错: {str(e)}", None, None
    
    def predict_emotion_from_microphone(self, audio_input):
        """
        从麦克风录音预测情绪
        
        Args:
            audio_input: Gradio Audio组件返回的元组 (sample_rate, audio_data)
            
        Returns:
            预测结果
        """
        if self.emotion_recognizer is None:
            return "请先加载模型", None, None
        
        if audio_input is None:
            return "请录制音频", None, None
        
        try:
            # 解包音频数据 - Gradio返回的是 (sample_rate, audio_data)
            sample_rate, audio_data = audio_input
            
            # 调试信息
            print(f"音频数据类型: {type(audio_data)}")
            print(f"音频数据形状: {getattr(audio_data, 'shape', 'No shape')}")
            print(f"采样率: {sample_rate}")
            
            # 确保音频数据是numpy数组
            if not isinstance(audio_data, np.ndarray):
                if isinstance(audio_data, (list, tuple)):
                    audio_data = np.array(audio_data)
                    print(f"转换为numpy数组，形状: {audio_data.shape}")
                else:
                    print(f"无法处理的数据类型: {type(audio_data)}")
                    return "❌ 音频数据格式错误", None, None
            
            # 预测情绪
            result = self.emotion_recognizer.predict_emotion_from_array(audio_data, sample_rate)
            
            if "error" in result:
                return f"❌ 预测失败: {result['error']}", None, None
            
            # 获取预测结果
            predicted_emotion = result["predicted_emotion"]
            confidence = result["confidence"]
            all_emotions = result["all_emotions"]
            
            # 创建结果文本
            chinese_emotion = self.chinese_emotions.get(predicted_emotion, predicted_emotion)
            result_text = f"""
## 🎯 预测结果

**主要情绪**: {chinese_emotion} ({predicted_emotion})
**置信度**: {confidence:.3f} ({confidence*100:.1f}%)

## 📊 所有情绪概率

"""
            
            for emotion, prob in all_emotions:
                chinese_name = self.chinese_emotions.get(emotion, emotion)
                percentage = prob * 100
                result_text += f"- **{chinese_name}** ({emotion}): {prob:.3f} ({percentage:.1f}%)\n"
            
            # 创建图表
            chart_path = self.create_emotion_chart(all_emotions)
            
            return result_text, chart_path, "麦克风录音"
            
        except Exception as e:
            return f"❌ 处理过程中出错: {str(e)}", None, None
    
    def create_interface(self):
        """
        创建Gradio界面
        
        Returns:
            Gradio界面
        """
        with gr.Blocks(title="语音情绪识别系统") as interface:
            gr.Markdown("""
            # 🎤 语音情绪识别系统
            
            使用Hugging Face中文预训练模型进行语音情绪识别，支持6种情绪分类：
            - 😠 愤怒 (Angry)
            - 😨 恐惧 (Fear)
            - 😊 快乐 (Happy)
            - 😢 悲伤 (Sad)
            - 😲 惊讶 (Surprise)
            - 😐 中性 (Neutral)
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 第一步：加载模型")
                    load_btn = gr.Button("🚀 加载模型", variant="primary")
                    model_status = gr.Textbox(label="模型状态", value="模型未加载", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 模型信息")
                    gr.Markdown("""
                    - **模型**: xmj2002/hubert-base-ch-speech-emotion-recognition
                    - **支持格式**: WAV, MP3, M4A等
                    - **采样率**: 自动重采样到16kHz
                    - **设备**: 自动检测CPU/GPU
                    - **语言**: 中文语音
                    """)
            
            gr.Markdown("---")
            
            with gr.Tab("📁 上传音频文件"):
                gr.Markdown("### 第二步：上传音频文件")
                file_input = gr.Audio(
                    label="选择音频文件",
                    type="filepath"
                )
                file_btn = gr.Button("🔍 分析音频文件", variant="primary")
            
            with gr.Tab("🎤 实时录音"):
                gr.Markdown("### 第二步：录制音频")
                mic_input = gr.Audio(
                    label="录制音频",
                    type="numpy"
                )
                mic_btn = gr.Button("🔍 分析录音", variant="primary")
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 第三步：查看结果")
                    result_output = gr.Markdown(label="预测结果")
                    audio_info = gr.Textbox(label="音频信息", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### 📊 情绪概率图表")
                    chart_output = gr.Image(label="情绪分布图")
            
            # 绑定事件
            load_btn.click(
                fn=self.load_model,
                outputs=model_status
            )
            
            file_btn.click(
                fn=self.predict_emotion_from_audio,
                inputs=file_input,
                outputs=[result_output, chart_output, audio_info]
            )
            
            mic_btn.click(
                fn=self.predict_emotion_from_microphone,
                inputs=mic_input,
                outputs=[result_output, chart_output, audio_info]
            )
            
            gr.Markdown("---")
            gr.Markdown("""
            ### 💡 使用提示
            
            1. **音频质量**: 建议使用清晰的语音，避免背景噪音
            2. **音频长度**: 建议3-10秒的语音片段
            3. **语言支持**: 专门针对中文语音优化
            4. **情绪表达**: 情绪表达越明显，识别准确率越高
            
            ### 🔧 技术说明
            
            - 使用Hubert模型进行中文语音特征提取
            - 基于Transformer架构进行情绪分类
            - 支持实时处理和批量处理
            - 自动音频预处理和格式转换
            - 在CASIA中文语音数据集上训练
            """)
        
        return interface

def main():
    """
    主函数
    """
    # 创建界面
    interface = EmotionRecognitionWebInterface()
    app = interface.create_interface()
    
    # 启动应用
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 禁用分享链接
        debug=True
    )

if __name__ == "__main__":
    main() 