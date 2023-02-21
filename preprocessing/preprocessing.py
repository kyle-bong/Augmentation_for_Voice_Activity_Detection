import librosa
import os
import numpy as np
import noisereduce as nr
import soundfile as sf
import random
from pydub import AudioSegment

# Preprocessing
def nr_and_trimming(file):
    """파일의 노이즈를 감쇄시키고, 무음 구간을 제거합니다.

    Args:
        file (str): 오디오 파일의 경로

    Returns:
        trimmed (numpy.ndarray): 노이즈 감쇄 및 무음 구간이 적용된 ndarray
    """

    audio_data, sample_rate = librosa.load(file, 16000)

    # Noise reduction
    noisy_part = audio_data[0:5000]
    reduced_noise = nr.reduce_noise(y=audio_data, y_noise=noisy_part, sr=16000)

    # Trimming. 무음구간을 제거합니다.
    trimmed, index = librosa.effects.trim(reduced_noise, top_db=10, hop_length=256, frame_length=512)

    return trimmed


# Slicing
def slicing(trimmed_path, sliced_file_path):
    """파일을 256ms 길이의 파일로 짧게 쪼개어 별도의 경로에 저장합니다.

    Args:
        file (str) : 쪼갤 파일의 경로
        sliced_file_path (str): 쪼개진 파일을 저장할 경로
    """
    # Slicing unit
    MS = 256

    

    # 무음구간을 삭제하고 남은 길이가 1024 이상인 파일만 slicing하여 저장합니다.
    for file in os.listdir(trimmed_path):
        if file.endswith('.wav'):
            trimmed = AudioSegment.from_wav(os.path.join(trimmed_path, file))
            if len(trimmed) >= 1024:
                for i in range(0, len(trimmed), MS):
                    trimmed[i:i+MS].export(os.path.join(sliced_file_path, file.split('.')[-2]+'_'+str(i)+'_sliced.wav'), format='wav')


if __name__ == "__main__":
    speaking_file_path = r'../dataset/speaking'
    speaking_trimmed_file_path = r'../dataset/speaking/trimmed'
    speaking_sliced_file_path = r'../dataset/speaking/sliced'

    other_sound_file_path = r'../dataset/other_sound'
    other_sound_trimmed_file_path = r'../dataset/other_sound/trimmed'
    other_sound_sliced_file_path = r'../dataset/other_sound/sliced'

    # Trimming
    for file in os.listdir(speaking_file_path):
        if file.endswith('.wav'):
            trimmed = nr_and_trimming(os.path.join(speaking_file_path, file))
            sf.write(os.path.join(speaking_trimmed_file_path, file.split('.')[-2]+'_trimmed.wav'), trimmed, 16000, format='wav')

    for file in os.listdir(other_sound_file_path):
        if file.endswith('.wav'):
            trimmed = nr_and_trimming(os.path.join(other_sound_file_path, file))
            sf.write(os.path.join(other_sound_trimmed_file_path, file.split('.')[-2]+'_trimmed.wav'), trimmed, 16000, format='wav')

    # Slicing
    # for file in os.listdir(speaking_trimmed_file_path):
    #     if file.endswith('.wav'):
    #         slicing(os.path.join(speaking_trimmed_file_path, file), speaking_sliced_file_path)

    # for file in os.listdir(other_sound_trimmed_file_path):
    #     if file.endswith('.wav'):
    #         slicing(os.path.join(other_sound_trimmed_file_path, file), other_sound_sliced_file_path)

    slicing(speaking_trimmed_file_path, speaking_sliced_file_path)
    slicing(other_sound_trimmed_file_path, other_sound_sliced_file_path)
