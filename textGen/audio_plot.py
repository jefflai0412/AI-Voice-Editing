import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

def resample_audio(y, orig_sr, target_sr):
    """
    將音訊重新取樣到目標取樣率
    :param y: 音訊數據
    :param orig_sr: 原始取樣率
    :param target_sr: 目標取樣率
    :return: 重新取樣後的音訊數據與新的取樣率
    """
    y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return y_resampled, target_sr

def plot_audio_waveform_and_spectrogram(wav_file, target_sr=16000):
    # 讀取音訊檔案
    y, sr = librosa.load(wav_file, sr=None)  # y 是音訊數據，sr 是原始取樣率
    print(f"原始取樣率: {sr} Hz")
    
    # 將音訊重新取樣
    if sr != target_sr:
        y, sr = resample_audio(y, orig_sr=sr, target_sr=target_sr)
        print(f"重新取樣後的取樣率: {sr} Hz")
    
    # 繪製聲波圖（Waveform）
    plt.figure(figsize=(14, 5))
    plt.title("Waveform")
    librosa.display.waveshow(y, sr=sr)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
    
    # 計算與繪製頻譜圖（Spectrogram）
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)  # 計算短時傅里葉變換 (STFT)
    plt.figure(figsize=(14, 5))
    plt.title("Spectrogram")
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()
    
    # 儲存重新取樣後的音訊（可選）
    output_path = wav_file.replace(".wav", f"_resampled_{target_sr}Hz.wav")
    sf.write(output_path, y, sr)
    print(f"重新取樣後的音訊已儲存至: {output_path}")

if __name__ == "__main__":
    wav_path = r"C:\Users\jason\OneDrive\Desktop\python_project\113_AI\audio\1.wav"  # 替換成你的 .wav 檔案路徑
    plot_audio_waveform_and_spectrogram(wav_path, target_sr=16000)  # 指定目標取樣率，例如 16kHz

