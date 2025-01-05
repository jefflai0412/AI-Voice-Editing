import speech_recognition as sr

def transcribe_audio(wav_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        print(f"Processing audio file: {wav_file}...")
        audio = recognizer.record(source)  # 讀取整個音訊檔案
    try:
        # 使用 Google Speech API 辨識中文（繁體）
        text = recognizer.recognize_google(audio, language="zh-TW")  # 改成 zh-CN 辨識簡體中文
        return text
    except sr.UnknownValueError:
        return "無法辨識音訊，可能音質不清楚。"
    except sr.RequestError as e:
        return f"無法請求結果：{e}"

if __name__ == "__main__":
    wav_path = r"C:\Users\jason\OneDrive\Desktop\python_project\113_AI\audio\1.wav"  # 替換成你的 .wav 檔案路徑
    transcription = transcribe_audio(wav_path)
    print("Transcription Result:")
    print(transcription)
