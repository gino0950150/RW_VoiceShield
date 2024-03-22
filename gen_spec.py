import os
import librosa
import matplotlib.pyplot as plt

def draw_spectrogram(audio_file, output_path):
    # 讀取音檔
    y, sr = librosa.load(audio_file)
    
    # 生成頻譜圖
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(librosa.stft(y), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(output_path)
    plt.close()

def process_audio_folder(folder_path):
    # 確保輸出目錄存在
    if not os.path.exists("output"):
        os.makedirs("output")
    
    # 處理每個音檔
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".wav"):  # 假設所有音檔都是wav格式
            audio_file = os.path.join(folder_path, file_name)
            output_file = os.path.join("output", f"{os.path.splitext(file_name)[0]}.jpg")
            draw_spectrogram(audio_file, output_file)

if __name__ == "__main__":
    audio_folder = input("請輸入音檔資料夾的路徑：")
    process_audio_folder(audio_folder)
    print("頻譜圖已生成並保存在output文件夾中。")