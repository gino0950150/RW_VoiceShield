import os
import librosa
import matplotlib.pyplot as plt
import numpy as np

def draw_spectrogram(audio_file, output_path):
    # 讀取音檔
    y, sr = librosa.load(audio_file)
    
    # 生成頻譜圖
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.savefig(output_path,bbox_inches='tight')
    plt.close()

def process_audio_folder(folder_path):

    # 處理每個音檔
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):  # 假設所有音檔都是wav格式
            audio_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}.jpg")
            draw_spectrogram(audio_file, output_file)

if __name__ == "__main__":
    paths = ["/homes/jinyu/RW_VoiceShield/docs/samples/blackbox", "/homes/jinyu/RW_VoiceShield/docs/samples/whitebox"]
    for path in paths:
        p = os.listdir(path)
        for i in p:
            process_audio_folder(os.path.join(path, i))