import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import warnings

# 忽略 FP16 警告
warnings.filterwarnings("ignore", category=UserWarning)

def record_audio(filename="sample.wav", duration=5, fs=16000):
    print(f"🎙️ 開始錄音（{duration} 秒）...")
    print("請開始說話...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # 等待錄音完成
    write(filename, fs, recording)  # 儲存為 wav 檔
    print("✅ 錄音完成，已儲存為", filename)

# 匯入 OpenAI Whisper 模型套件
import whisper

# 定義一個函式，用來進行語音辨識
def transcribe_audio(file_path: str):
    print("🔍 載入模型...")
    model = whisper.load_model("base")# 載入 Whisper 的 "base" 模型（還有 tiny、small、medium、large 可選）

    print("✅ 模型載入完成。開始辨識音訊檔...")

    result = model.transcribe(file_path)# 使用模型對輸入的音訊檔案進行轉錄（語音 → 文字）
    print("\n📝 辨識結果：")
    print(result["text"])

if __name__ == "__main__":
    audio_path = "T2.wav"
    record_audio(audio_path, duration=5)  # 錄音 5 秒
    transcribe_audio(audio_path)
