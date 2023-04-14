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
path_audio = 'artic_cmu/'
def mix_audio(audio, noise, mixing_parameter):
    mixed_audio = (1 - mixing_parameter) * audio + mixing_parameter * noise
    return mixed_audio

        
def create_log_spectrograms(output_dir):
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with h5py.File(output_dir+'/'+'final_dataset.h5', 'w') as f:
        
        
        n_mels = 64
        count = 0
        spectrograms_mix = f.create_dataset('spectrograms_mix', (200, seq_len, n_mels), dtype=np.float32)
        labels_mix = f.create_dataset('labels_mix', (200,), dtype=np.float32)
        # iterate over the dataset and compute mel-spectrograms
        for folder in sorted(os.listdir(path_noise)):
            if not folder.startswith('.'):
                for file in sorted(os.listdir(path_noise+folder+'/')):
                    noise, sr = librosa.load(path_noise+folder+'/'+file)
                    # We only take noises with duratioin > than 2 seconds. Otherwise, we discard them.
                    if librosa.get_duration(noise) > 2:
                        # We load human voices dataset (common mozilla voice) and we mix it with our own noise dataset
                        audio_path = None
                        while audio_path is None:
                            audio_path = os.listdir('artic_cmu')[count]
                            audio, sr = librosa.load(path_audio + audio_path, sr = sample_rate)
                            # We only take human voices with duratioin > than 2 seconds. Otherwise, we discard them.
                            if librosa.get_duration(audio) < 2:
                                audio_path = None
                        # We check audio durations and we truncate the one longer to match the sorter one.
                        # find the minimum duration of the two audio signals
                        duration = min(len(noise), len(audio))
                        if duration<22000:
                            print('Warning')
                        # truncate or pad the audio signals to the minimum duration
                        if len(noise) > duration:
                            noise = noise[:duration]
                        else:
                            noise = librosa.util.pad_center(noise, duration)

                        if len(audio) > duration:
                            audio = audio[:duration]
                        else:
                            audio = librosa.util.pad_center(audio, duration)
                        mixed_audio = mix_audio(audio, noise, 0.2)
                        sf.write(output_dir+'/final_mixed_audio'+str(count)+'.wav',mixed_audio,sample_rate)
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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory where we save the audio spectograms")
    args = parser.parse_args()
    create_log_spectrograms(args.output_dir)