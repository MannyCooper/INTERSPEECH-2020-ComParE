from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import numpy as np
import scipy.io.wavfile as wav
#mfcc(signal,samplerate=16000,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)

(rate,sig) = wav.read("wav/devel/devel_00001.wav")
mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
# fbank_feat = logfbank(sig,rate,winlen=0.025,winstep=0.01,
# 	nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

#print(len(mfcc_feat))
#print(len(mfcc_feat[0]))
#print(mfcc_feat[:,:])
#print(fbank_feat[1:3,:])

d_mfcc_feat = delta(mfcc_feat, 2)
a_mfcc_feat = delta(d_mfcc_feat, 2)

feat = np.hstack([mfcc_feat, d_mfcc_feat, a_mfcc_feat])

#print(len(feat))
#print(len(feat[0]))
#print(feat)

csv = [i for item in feat for i in item]
print(len(csv))
# print(csv)

res = []
res.append(csv)
print(res)