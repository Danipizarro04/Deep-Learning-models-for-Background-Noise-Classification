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


def mix_audio(audio, noise, mixing_parameter):
    mixed_audio = (1 - mixing_parameter) * audio + mixing_parameter * noise
    return mixed_audio

        
def create_log_spectrograms(output_dir):
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    with h5py.File(output_dir+'/'+'Urban_spectograms_augmented.h5', 'w') as f:
        
        
        n_mels = 64
        dataset = soundata.initialize('urbansound8k')
        ids = dataset.clip_ids
        spectrograms_aug = f.create_dataset('spectrograms_aug', (len(ids), seq_len, n_mels), dtype=np.float32)
        labels_aug = f.create_dataset('labels_aug', (len(ids),), dtype=np.float32)
        # iterate over the dataset and compute mel-spectrograms
        for i, clip_id in enumerate(ids):
            clip = dataset.clip(clip_id)
            noise, sr = clip.audio
            label = clip.tags.labels[0]
            # We only take noises with duratioin > than 1 second. Otherwise, we discard them.
            if librosa.get_duration(noise) > 2:
                # We load human voices dataset (artic_cmu) and we mix it with urbanSound8K
                audio_path = None
                while audio_path is None:
                    random_audio = np.random.choice(os.listdir('artic_cmu'))
                    audio_path = os.path.join('artic_cmu', random_audio)
                    audio, sr = librosa.load(audio_path, sr = sample_rate)
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
                sf.write(output_dir+'/mixed_audio'+str(i)+'.wav',mixed_audio,sample_rate)
                # Log mel-spectrogram
                mel = librosa.filters.mel(sr=sample_rate, n_fft=window_size, n_mels=mel_bins, fmin=50., fmax=sample_rate // 2).T

                [f, t, x] = signal.spectrogram(mixed_audio, window=ham_win, nperseg=window_size, noverlap=overlap, detrend=False, return_onesided=True, mode='magnitude') 

                x = x.T


                x = np.dot(x, mel)


                x = np.log(x + 1e-8)

                x = x.astype(np.float32)

                spectrograms_aug[i] = np.resize(x, (seq_len, n_mels))
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
                labels_aug[i] = label_map[label]
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory where we save the audio spectograms")
    args = parser.parse_args()
    create_log_spectrograms(args.output_dir)