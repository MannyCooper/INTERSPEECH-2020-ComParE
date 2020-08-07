import csv
from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import numpy as np
import scipy.io.wavfile as wav
import os

res = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
print(len(res[0]))


csvFile = open("/Users/jcy/Documents/vscode/mfcc/labels.csv", "r")
reader = csv.reader(csvFile)
clear = []
mask = []
clearnum = []
masknum = []
for item in reader:
    # 忽略第一行
    # if reader.line_num == 1:
    #     continue
    # result[item[0]] = item[1]
    if item[1] == 'clear':
        clear.append(item[0])
        # print(item[0])
    if item[1] == 'mask':
        mask.append(item[0])
        # print(item)
csvFile.close()



# print(list_0)
# (rate,sig)= wav.read("/Users/jcy/Documents/vscode/mfcc/wav/mix/train_00001.wav")


for i in mask:
    # print(i)
    (rate,sig) = wav.read("/Users/jcy/Documents/vscode/mfcc/wav/mix/"+i)
    mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.01,numcep=13,nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
    d_mfcc_feat = delta(mfcc_feat, 2)
    a_mfcc_feat = delta(d_mfcc_feat, 2)
    feat = np.hstack([mfcc_feat, d_mfcc_feat, a_mfcc_feat])
    res = np.vstack([res,feat])
print(len(res))
np.savetxt('MFCC_mask.csv', res, delimiter = ',')


# print(clear)
# print(mask)