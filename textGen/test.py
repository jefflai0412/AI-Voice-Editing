import whisper

model = whisper.load_model("base")
wav_path = r"C:\Users\jason\Desktop\python_project\AI_final\textGen\audio\Demo.wav"
result = model.transcribe(wav_path, task="transcribe", word_timestamps=True)

for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")