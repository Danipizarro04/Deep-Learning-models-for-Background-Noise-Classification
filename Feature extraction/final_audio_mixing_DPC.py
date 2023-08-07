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
import random
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

sample_rate = 22050
window_size = 1024
overlap = 336   
seq_len = 127 
mel_bins = 64
ham_win = np.hamming(window_size)
path_noise = 'noise_dataset/'
path_audio = 'LibriSpeech/train-clean-100/412/126975/'

def mix_audio(audio, noise, mixing_parameter):
    mixed_audio = (1 - mixing_parameter) * audio + mixing_parameter * noise
    return mixed_audio

        
def create_log_spectrograms(output_dir):
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with h5py.File(output_dir+'/'+'DPC.h5', 'w') as f:
        
        
        n_mels = 64
        count = 0
        count2 = 0
        count_1 = 0
        count_5 = 0
        count_10 = 0
        count_20 = 0
        spectrograms_mix = f.create_dataset('spectrograms_mix', (800, seq_len, n_mels), dtype=np.float32)
        labels_mix = f.create_dataset('labels_mix', (800,), dtype=np.float32)
        spectrograms_mix_1 = f.create_dataset('spectrograms_mix_1', (200, seq_len, n_mels), dtype=np.float32)
        labels_mix_1 = f.create_dataset('labels_mix_1', (200,), dtype=np.float32)
        spectrograms_mix_5 = f.create_dataset('spectrograms_mix_5', (200, seq_len, n_mels), dtype=np.float32)
        labels_mix_5 = f.create_dataset('labels_mix_5', (200,), dtype=np.float32)
        spectrograms_mix_10 = f.create_dataset('spectrograms_mix_10', (200, seq_len, n_mels), dtype=np.float32)
        labels_mix_10 = f.create_dataset('labels_mix_10', (200,), dtype=np.float32)
        spectrograms_mix_20 = f.create_dataset('spectrograms_mix_20', (200, seq_len, n_mels), dtype=np.float32)
        labels_mix_20 = f.create_dataset('labels_mix_20', (200,), dtype=np.float32)
        # iterate over the dataset and compute mel-spectrograms
        for folder in sorted(os.listdir(path_noise)):
            if not folder.startswith('.'):
                for file in sorted(os.listdir(path_noise+folder+'/')):
                    noise, sr = librosa.load(path_noise+folder+'/'+file)
                    audio_path = None
                    while audio_path is None:
                        audio_path = os.listdir('LibriSpeech/train-clean-100/412/126975')[count2]
                        audio, sr = librosa.load(path_audio + audio_path, sr = sample_rate)
                         # We only take human voices with duratioin > than 7 seconds. Otherwise, we discard them.
                        if len(audio)/sample_rate < 7:
                            audio_path = None
                            
                    duration = len(noise)
                    if len(audio) > duration:
                        audio = audio[:duration]
                    else:
                        audio = librosa.util.pad_center(audio, duration)
                    mixed_audio = mix_audio(audio, noise, 0.6)
                    folder_path = os.path.join(output_dir, folder)
                    if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                    sf.write(folder_path+'/DPC'+str(count+1)+'.wav',mixed_audio,sample_rate)
                        # Log mel-spectrogram
                    mel = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=50., fmax=sample_rate // 2).T

                    [f, t, x] = signal.spectrogram(mixed_audio, window=ham_win, nperseg=window_size, noverlap=overlap, detrend=False, return_onesided=True, mode='magnitude') 

                    x = x.T


                    x = np.dot(x, mel)


                    x = np.log(x + 1e-8)

                    x = x.astype(np.float32)

                    spectrograms_mix[count] = np.resize(x, (seq_len, n_mels))
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
                    labels_mix[count] = label_map[folder]
                    count+=1
                    count2+=1
                    if count2 >= 40:
                        count2 = 0
                        
                    if duration / sample_rate < 2:
                        folder_path = os.path.join(output_dir+'_1_sec', folder)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        sf.write(folder_path+'/DPC'+str(count_1+1)+'.wav',mixed_audio,sample_rate)
                        spectrograms_mix_1[count_1] = np.resize(x, (seq_len, n_mels))
                        labels_mix_1[count_1] = label_map[folder]
                        count_1+=1
                    
                    if duration / sample_rate > 2 and duration/sample_rate < 7:
                        folder_path = os.path.join(output_dir+'_5_sec', folder)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        sf.write(folder_path+'/DPC'+str(count_5+1)+'.wav',mixed_audio,sample_rate)
                        spectrograms_mix_5[count_5] = np.resize(x, (seq_len, n_mels))
                        labels_mix_5[count_5] = label_map[folder]
                        count_5+=1
                    
                    if duration / sample_rate > 7 and duration/sample_rate < 13:
                        folder_path = os.path.join(output_dir+'_10_sec', folder)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        sf.write(folder_path+'/DPC'+str(count_10+1)+'.wav',mixed_audio,sample_rate)
                        spectrograms_mix_10[count_10] = np.resize(x, (seq_len, n_mels))
                        labels_mix_10[count_10] = label_map[folder]
                        count_10+=1
                        
                    if duration / sample_rate > 13:
                        folder_path = os.path.join(output_dir+'_20_sec', folder)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        sf.write(folder_path+'/DPC'+str(count_20+1)+'.wav',mixed_audio,sample_rate)
                        spectrograms_mix_20[count_20] = np.resize(x, (seq_len, n_mels))
                        labels_mix_20[count_20] = label_map[folder]
                        count_20+=1


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory where we save the audio spectograms")
    args = parser.parse_args()
    create_log_spectrograms(args.output_dir)