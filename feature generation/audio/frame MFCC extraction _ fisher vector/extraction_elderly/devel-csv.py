from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import numpy as np
import scipy.io.wavfile as wav
import os
path = r'/Users/jcy/Documents/vscode/mfcc_elderly/wav/test'
res = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
# print(len(res[0]))

list_0 = os.listdir(path)
list_0.sort()
print(len(list_0))

for i in list_0:
    # print("wav/devel/"+i)
    (rate,sig) = wav.read("/Users/jcy/Documents/vscode/mfcc_elderly/wav/test/"+i)
    mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
    d_mfcc_feat = delta(mfcc_feat, 2)
    a_mfcc_feat = delta(d_mfcc_feat, 2)
    feat = np.hstack([mfcc_feat, d_mfcc_feat, a_mfcc_feat])
    feat = feat[0:498]
    print(len(feat))
    res = np.vstack([res,feat])
# print(res)
print(len(res))
# np.savetxt('MFCC_test.csv', res, delimiter = ',')