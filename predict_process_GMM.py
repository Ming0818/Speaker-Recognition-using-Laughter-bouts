# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:52:37 2018

@author: Oyeyemi Damilola
https://github.com/abhijeet3922/Speaker-identification-using-GMMs
"""

#path to training data
 

# read the laughter file

import os
import pickle 
from feature_extraction import extract_features
from scipy.io.wavfile import read
import numpy as np

model_path = 'models'

path_of_test_laugh = r"C:\Users\USER\Speaker_Recognition_Laugther\test.wav"

rate,sig = read(path_of_test_laugh)

gmm_files = [os.path.join(model_path,laugher) for laugher in 
              os.listdir(model_path) if laugher.endswith('.gmm')]

models    = [pickle.load(open(laugher,'rb')) for laugher in gmm_files]

speakers   = [laugher.split(".gmm")[0].split('\\')[1] for laugher 
              in gmm_files]

#%%
# predict the speaker

rate,audio = read(path_of_test_laugh)
vector   = extract_features(audio,rate)
    
log_likelihood = np.zeros(len(models)) 
    
for i in range(len(models)):
    gmm = models[i]         #checking with each model one by one
    scores = np.array(gmm.score(vector))
    log_likelihood[i] = scores.sum()
    
winner = np.argmax(log_likelihood)
print ("\tdetected as - ", speakers[winner])
#time.sleep(1.0)