# ref: https://qiita.com/kotai2003/items/69638e18b6d542fb275e

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

def preEmphasis(wave, p=0.97): # 高音強調
    return scipy.signal.lfilter([1.0, -p], 1, wave)

voice_file = "./voice.wav"
y, sr = librosa.load(voice_file)

y = y/max(abs(y))
p = 0.97
y = preEmphasis(y, p)

plt.figure(figsize=(16,6))
librosa.display.waveshow(y=y, sr=sr) # 時間軸

D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) # 周波数軸
    # n_fft(窓関数のサイズ)
    # hop_length(窓のスライド幅)

DB = librosa.amplitude_to_db(D, ref=np.max) # スペクトログラム（フォルマント）
#librosa.display.specshow(DB, sr=sr, hop_length=512, x_axis='time', y_axis='log')
#plt.colorbar()
#plt.show()

order = 20
frame_length = 320
length=y.shape[0]
n_frame =length // frame_length

worN = 513
envelope = np.zeros((worN, n_frame))
eps=1e-3

for k in range(n_frame):
    slc = slice(k*n_frame, (k+1)*n_frame)
    try:
        a=librosa.lpc(y[slc], order)
        freqs, h = scipy.signal.freqz(1.0, a, worN=worN)
        envelope[:, k] = np.abs(h)
    except FloatingPointError:
        pass

env_min = -120
with np.errstate(divide='ignore'):
    envelope = 20 * np.log10(envelope/np.max(envelope))
envelope[envelope<env_min] = env_min

plt.clf()
librosa.display.specshow(
    envelope, 
    sr=sr, 
    hop_length=frame_length,
    x_axis='time', y_axis='log'
)
plt.colorbar(format='%+2.0f dB')
plt.show()

