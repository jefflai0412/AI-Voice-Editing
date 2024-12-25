import librosa
import librosa.display

# 讀取音訊檔案
wav_path = r"C:\Users\jason\OneDrive\Desktop\python_project\113_AI\audio\1.wav"
y, sr = librosa.load(wav_path, sr=None)

# 檢測語音段落
intervals = librosa.effects.split(y, top_db=20)

# 輸出每段的時間範圍
for start, end in intervals:
    print(f"Start: {start / sr:.2f}s, End: {end / sr:.2f}s")