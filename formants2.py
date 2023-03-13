import numpy as np

x = np.array([1,2,3,-2,-3,-4,4,-1], dtype='float64')
N=len(x) # 8

xpad = np.pad(x,[0,N-1],'constant') #[1,2,3,-2,-3,-4,4,-1,0,0,0,0,0,0,0]

# 自己相関関数を求める
# 時間 t だけずらして自身と積をとった値（時間平均）
corr = [np.sum(xpad[0:N] * xpad[lag:lag+N]) for lag in range(N)]
print(corr)

autocorr = np.correlate(x, x, mode='full')
print(autocorr)
