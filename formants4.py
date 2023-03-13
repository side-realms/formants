import numpy as np
import librosa
from scipy import signal
import matplotlib.pyplot as plt

x = np.array([1,2,3,-2,-3,-4,4,-1], dtype='float64')
N=len(x)

y_sound, sr = librosa.load('./aaa.wav')

def levinson(y, p, r=None):
    if r is None:
        autocorr = np.correlate(y,y,mode='full')
        return levinson(y, p, autocorr[len(y)-1:len(y)+p])
    
    # 初期条件を以下のように設定する
    # a1 = -r(1)/r(0)
    # E1 = r(0) + r(1)a1
    if p == 1:
        a = np.array([1, -r[1]/r[0]])
        E = a.dot(r[:2])
    
    # そうでなければ，過去の値から予測する
    else:
        aa, EE = levinson(y, p-1, r)
        kk = -(aa.dot(r[p:0:-1])) / EE
        U = np.append(aa,0) # 0 を追加
        V=U[::-1] # 転置
        a=U+kk*V # A=U+kV
        E=EE*(1-kk*kk)
    
    return a, E

a, E = levinson(y_sound/32768, 20)

w, h = signal.freqz(1, a)
w = sr*w/2/np.pi
h = 20 * np.log10(np.abs(h))
plt.plot(w, h)
plt.grid()
plt.show()