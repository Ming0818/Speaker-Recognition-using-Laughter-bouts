# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 08:54:34 2018

@author: Oyeyemi Damilola
references from https://github.com/abhijeet3922/Speaker-identification-using-GMMs
"""

import os
import pickle 
from feature_extraction import extract_features
from scipy.io.wavfile import read
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 
from collections import deque
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    
    # reading the files from each directory
    
    main_dir = 'laughter'
    model_destination = 'models' 
    os.mkdir(model_destination)
    
    for speaker in os.listdir(main_dir):
        
        # extracting the audio files for each speaker
        
        features = np.asarray(())
        
        speaker_laughter_path = os.path.join(main_dir,speaker)       
        
        laughter_paths = [os.path.join(speaker_laughter_path,audio_laughter)for audio_laughter in os.listdir(speaker_laughter_path) if os.path.isfile(os.path.join(speaker_laughter_path,audio_laughter))]
        
        speaker_laughter = deque(laughter_paths)
        
        # appending the feature for each speaker
        for i in range(len(laughter_paths)):
            
            rate,sig = read(speaker_laughter.popleft())
            feature_vector = extract_features(sig,rate)
            
            if features.size == 0:                
                features = feature_vector
            else:                
                features = np.vstack((features,feature_vector))
                
        gmm = GMM(n_components = 16, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
        
        speaker_model = speaker+ ".gmm"
        with open(model_destination+'\\'+speaker_model,'wb') as file:
            pickle.dump(gmm,file)
            
        print("Training for speaker %s has been completed with data points = (%d,%d)" %(speaker,features.shape[0],features.shape[1]))
        
     