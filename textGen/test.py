import whisper

# 加載模型
model = whisper.load_model("medium")

# 語音轉文字，包含時間戳
wav_path = r"C:\Users\jason\OneDrive\Desktop\python_project\113_AI\audio\1.wav"
result = model.transcribe(wav_path, task="transcribe", word_timestamps=True)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"[{word['start']:.2f} - {word['end']:.2f}] {word['text']}")