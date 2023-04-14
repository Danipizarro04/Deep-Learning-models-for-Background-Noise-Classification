import soundata
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
count = 0

def create_log_spectrograms(output_dir):
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with h5py.File(output_dir+'/'+'Urban_spectograms.h5', 'w') as f:
        
        
        n_mels = 64
        dataset = soundata.initialize('urbansound8k')
        ids = dataset.clip_ids
        spectrograms = f.create_dataset('spectrograms', (len(ids), seq_len, n_mels), dtype=np.float32)
        labels = f.create_dataset('labels', (len(ids),), dtype=np.float32)
        # iterate over the dataset and compute mel-spectrograms
        for i, clip_id in enumerate(ids):
            clip = dataset.clip(clip_id)
            audio, sr = clip.audio
            label = clip.tags.labels[0]
            if librosa.get_duration(audio) > 2:
                # Log mel-spectrogram
                mel = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=50., fmax=sample_rate // 2).T

                [f, t, x] = signal.spectrogram(audio, window=ham_win, nperseg=window_size, noverlap=overlap, detrend=False, return_onesided=True, mode='magnitude') 

                x = x.T


                x = np.dot(x, mel)


                x = np.log(x + 1e-8)

                x = x.astype(np.float32)

                spectrograms[i] = np.resize(x, (seq_len, n_mels))
                label_map = {
                    'air_conditioner': 0,
                    'car_horn': 1,
                    'children_playing': 2,
                    'dog_bark': 3,
                    'drilling': 4,
                    'engine_idling': 5,
                    'gun_shot': 6,
                    'jackhammer': 7,
                    'siren': 8,
                    'street_music': 9
                }
                labels[i] = label_map[label]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory where we save the audio spectograms")
    args = parser.parse_args()
    create_log_spectrograms(args.output_dir)