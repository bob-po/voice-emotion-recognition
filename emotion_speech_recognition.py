import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union
import os

class HubertClassificationHead(nn.Module):
    """
    Hubert分类头，用于情绪分类
    """
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_class)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class HubertForSpeechClassification(HubertPreTrainedModel):
    """
    用于语音分类的Hubert模型
    """
    def __init__(self, config):
        super().__init__(config)
        self.hubert = HubertModel(config)
        self.classifier = HubertClassificationHead(config)
        self.init_weights()

    def forward(self, x):
        outputs = self.hubert(x)
        hidden_states = outputs[0]
        x = torch.mean(hidden_states, dim=1)
        x = self.classifier(x)
        return x

class EmotionSpeechRecognition:
    """
    语音情绪识别类，使用中文语音情绪识别模型
    """
    
    def __init__(self, model_name: str = "xmj2002/hubert-base-ch-speech-emotion-recognition"):
        """
        初始化语音情绪识别模型
        
        Args:
            model_name: Hugging Face模型名称
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 模型参数
        self.duration = 6
        self.sample_rate = 16000
        
        # 加载配置
        self.config = AutoConfig.from_pretrained(model_name)
        
        # 加载特征提取器
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # 加载模型
        self.model = HubertForSpeechClassification.from_pretrained(
            model_name,
            config=self.config,
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 情绪标签映射（根据中文模型调整）
        self.emotion_labels = {
            0: "angry",      # 愤怒
            1: "fear",       # 恐惧
            2: "happy",      # 快乐
            3: "neutral",    # 中性
            4: "sad",        # 悲伤
            5: "surprise"    # 惊讶
        }
        
        print(f"中文语音情绪识别模型 {model_name} 加载完成")
    
    def id2class(self, id):
        """
        将ID转换为情绪类别名称
        """
        return self.emotion_labels.get(id, f"class_{id}")
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
        """
        预处理音频文件
        
        Args:
            audio_path: 音频文件路径
            target_sr: 目标采样率
            
        Returns:
            预处理后的音频数组
        """
        try:
            # 加载音频文件
            if audio_path.endswith('.wav'):
                audio, sr = sf.read(audio_path)
            else:
                audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # 如果是立体声，转换为单声道
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # 重采样到目标采样率
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            return audio
            
        except Exception as e:
            print(f"音频预处理错误: {e}")
            return None
    
    def extract_features(self, audio: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        提取音频特征
        
        Args:
            audio: 音频数组
            
        Returns:
            特征张量字典
        """
        # 使用模型的特征提取器，按照中文模型的要求处理
        inputs = self.processor(
            audio, 
            padding="max_length", 
            truncation=True, 
            max_length=self.duration * self.sample_rate,
            return_tensors="pt", 
            sampling_rate=self.sample_rate
        )
        
        return inputs
    
    def predict_emotion(self, audio_path: str) -> Dict:
        """
        预测音频的情绪
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            包含预测结果的字典
        """
        try:
            # 预处理音频
            audio = self.preprocess_audio(audio_path)
            if audio is None:
                return {"error": "音频预处理失败"}
            
            # 提取特征
            inputs = self.extract_features(audio)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 进行预测
            with torch.no_grad():
                logits = self.model(inputs["input_values"])
                probabilities = F.softmax(logits, dim=1)
                
                # 获取预测结果
                predicted_class = torch.argmax(logits).cpu().numpy()
                confidence = probabilities[0][int(predicted_class)].cpu().numpy()
                
                # 获取所有情绪的概率
                emotion_probs = {}
                for i, prob in enumerate(probabilities[0]):
                    emotion_probs[self.id2class(i)] = prob.cpu().numpy()
                
                # 按概率排序
                sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                
                result = {
                    "predicted_emotion": self.id2class(int(predicted_class)),
                    "confidence": confidence,
                    "all_emotions": sorted_emotions,
                    "audio_path": audio_path
                }
                
                return result
                
        except Exception as e:
            return {"error": f"预测过程中出错: {str(e)}"}
    
    def predict_emotion_from_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        从音频数组预测情绪
        
        Args:
            audio_array: 音频数组
            sample_rate: 采样率
            
        Returns:
            包含预测结果的字典
        """
        try:
            # 确保音频数据是numpy数组
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            # 确保音频数据是浮点数类型
            if audio_array.dtype != np.float32 and audio_array.dtype != np.float64:
                audio_array = audio_array.astype(np.float32)
            
            # 确保音频是单声道
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # 重采样到16kHz
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # 提取特征
            inputs = self.extract_features(audio_array)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 进行预测
            with torch.no_grad():
                logits = self.model(inputs["input_values"])
                probabilities = F.softmax(logits, dim=1)
                
                # 获取预测结果
                predicted_class = torch.argmax(logits).cpu().numpy()
                confidence = probabilities[0][int(predicted_class)].cpu().numpy()
                
                # 获取所有情绪的概率
                emotion_probs = {}
                for i, prob in enumerate(probabilities[0]):
                    emotion_probs[self.id2class(i)] = prob.cpu().numpy()
                
                # 按概率排序
                sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
                
                result = {
                    "predicted_emotion": self.id2class(int(predicted_class)),
                    "confidence": confidence,
                    "all_emotions": sorted_emotions
                }
                
                return result
                
        except Exception as e:
            return {"error": f"预测过程中出错: {str(e)}"}
    
    def batch_predict(self, audio_paths: List[str]) -> List[Dict]:
        """
        批量预测多个音频文件的情绪
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            预测结果列表
        """
        results = []
        for audio_path in audio_paths:
            result = self.predict_emotion(audio_path)
            results.append(result)
        return results

# 使用示例
if __name__ == "__main__":
    # 初始化模型
    emotion_recognizer = EmotionSpeechRecognition()
    
    # 测试音频文件路径（需要替换为实际路径）
    test_audio_path = "C:/Users/guoji/Music/天不生我毋忠良-气势.wav"
    
    if os.path.exists(test_audio_path):
        # 预测情绪
        result = emotion_recognizer.predict_emotion(test_audio_path)
        print("预测结果:", result)
    else:
        print(f"测试音频文件 {test_audio_path} 不存在") 