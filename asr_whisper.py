import whisper

def transcribe_audio(file_path: str):
    print("🔍 載入模型...")
    model = whisper.load_model("base")
    print("✅ 模型載入完成。開始辨識音訊檔...")

    result = model.transcribe(file_path)

    print("\n📝 辨識結果：")
    print(result["text"])

if __name__ == "__main__":
    audio_path = "sample.wav"
    transcribe_audio(audio_path)
