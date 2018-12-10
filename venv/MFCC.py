import numpy as np
import librosa
import tensorflow as tf

# all of the test records are sampled with rate 22050
y, sr = librosa.load('sa1.wav')
mfccs = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13)
print(mfccs.shape)