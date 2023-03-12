# ref: https://qiita.com/kotai2003/items/69638e18b6d542fb275e

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

voice_file = "./voice.wav"
y, sr = librosa.load(voice_file)

plt.figure(figsize=(16,6))
librosa.display.waveshow(y=y, sr=sr) # 時間軸

D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) # 周波数軸
    # n_fft(窓関数のサイズ)
    # hop_length(窓のスライド幅)

DB = librosa.amplitude_to_db(D, ref=np.max) # スペクトログラム（フォルマント）
#librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
#plt.colorbar()
#plt.show()
