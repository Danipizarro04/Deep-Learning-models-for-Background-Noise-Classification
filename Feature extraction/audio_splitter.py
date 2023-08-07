import argparse
import os
from pydub import AudioSegment
import math

def split_audio(input_file, duration):
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Calculate the number of chunks
    num_chunks = math.ceil(len(audio) / duration)

    # Split the audio into equal chunks
    chunks = []
    for i in range(num_chunks):
        start_time = i * duration  # Convert to milliseconds
        end_time = (i + 1) * duration
        chunk = audio[start_time:end_time]
        chunks.append(chunk)

    # Create a folder to store the audio chunks
    folder_name = 'formula1'
    os.makedirs(folder_name, exist_ok=True)

    # Export each chunk to a separate file in the folder
    for i, chunk in enumerate(chunks):
        if i == 20:
            break;
        output_file = os.path.join(folder_name, f"{i+61}.wav")
        chunk.export(output_file, format="wav")
        print(f"Exported {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a WAV file into equal chunks of specified duration.')
    parser.add_argument('input_file', help='Path to the input WAV file')
    parser.add_argument('duration', type=int, help='Duration of each split audio in seconds')

    args = parser.parse_args()
    input_file = args.input_file
    duration = args.duration

    split_audio(input_file, duration)
