import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import os
import argparse
import librosa
import csv
import h5py
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

sample_rate = 22050
window_size = 1024
overlap = 336   
seq_len = 127 
mel_bins = 64
ham_win = np.hamming(window_size)
path = 'ybss-audios/'

def audio_to_spectrogram(output_dir):
    i=0
    c = 0
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    with h5py.File(output_dir+'/'+'Ybss-spectrograms.h5', 'w') as f:
        
        train_spec = f.create_dataset('train_spec', (1599, seq_len, mel_bins), dtype=np.float32)
        train_labs = f.create_dataset('train_labs', (1599,), dtype=np.float32)
        
        test_spec = f.create_dataset('test_spec', (400, seq_len, mel_bins), dtype=np.float32)
        test_labs = f.create_dataset('test_labs', (400,), dtype=np.float32)
        
        for folder in sorted(os.listdir(path)):
          if not folder.startswith('.'):
            for folder2 in sorted(os.listdir(path+folder+'/')):
                if not folder2.startswith('.') and folder2 == 'test' :
                    for file in sorted(os.listdir(path+folder+'/'+folder2+'/')):
                        audio, sr = librosa.load(path+folder+'/'+folder2+'/'+file)
                        # Log mel-spectogram
                        mel = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=50., fmax=sample_rate // 2).T

                        [f, t, x] = signal.spectrogram(audio, window=ham_win, nperseg=window_size, noverlap=overlap, detrend=False, return_onesided=True, mode='magnitude') 

                        x = x.T

                           
                        x = np.dot(x, mel)

                        
                        x = np.log(x + 1e-8)

                        x = x.astype(np.float32)
                        test_spec[i] = np.resize(x, (seq_len, mel_bins))
                        label_map = {
                            'auto_rikshaw': 0,
                            'cricket_crowd': 1,
                            'electronic_stapler': 2,
                            'formula_1': 3,
                            'grass_cutting': 4,
                            'guitar': 5,
                            'helicoptor': 6,
                            'sewing_machine': 7,
                            'tap_water': 8,
                            'traffic': 9
                        }
                        test_labs[i] = label_map[folder]
                        i+=1
                elif not folder2.startswith('.') and not folder2 == 'test' : 
                    for file in sorted(os.listdir(path+folder+'/'+folder2+'/')):
                        audio, sr = librosa.load(path+folder+'/'+folder2+'/'+file)
                        if len(audio) < window_size:
                            print(folder+' '+folder2+' '+file)
                            print(ValueError('Input signal is too short'))
                        else:
                            # Log mel-spectrograms
                            mel = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=50., fmax=sample_rate // 2).T

                            [f, t, x] = signal.spectrogram(audio, window=ham_win, nperseg=window_size, noverlap=overlap, detrend=False, return_onesided=True, mode='magnitude') 

                            x = x.T

                                
                            x = np.dot(x, mel)

                            
                            x = np.log(x + 1e-8)

                            x = x.astype(np.float32)

                            train_spec[c] = np.resize(x, (seq_len, mel_bins))
                            label_map = {
                                'auto_rikshaw': 0,
                                'cricket_crowd': 1,
                                'electronic_stapler': 2,
                                'formula_1': 3,
                                'grass_cutting': 4,
                                'guitar': 5,
                                'helicoptor': 6,
                                'sewing_machine': 7,
                                'tap_water': 8,
                                'traffic': 9
                            }
                            train_labs[c] = label_map[folder]
                            c+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory containing audio files to be converted to spectrograms")
    args = parser.parse_args()
    audio_to_spectrogram(args.output_dir)
