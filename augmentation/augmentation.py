import os
import wave
import librosa
import numpy as np
import soundfile as sf
import random
import librosa.display
import matplotlib.pyplot as plt
from room_impulse_response import *

# Time Stretching
# 소리를 빠르거나 느리게 변형합니다.

class Augmentation:
    def __init__(self):
        self.ir_path = r'../dataset/IR'
        random.seed = 42
    
    def impulse_response(self, path, output_path):
        # Trimming 및 Slicing된 오디오 파일과 Impulse Response를 Convolution하여 새로운 반향이 적용된 오디오 파일을 생성합니다.
        ir_path = self.ir_path

        ir_files = os.listdir(ir_path)

        # 오디오 파일에 무작위 음향 환경 적용
        for file in os.listdir(path):
            if file.endswith('.wav'):
                
                # ir sampling
                ir_file = random.sample(ir_files, 1)[0]

                # convolution
                convolution_reverb(os.path.join(path, file), 
                                os.path.join(ir_path, ir_file),
                                os.path.join(output_path, file[:-4]+'_rir.wav'))
            
    # 소리 속도 변형
    def speed_shifting(self, path, output_path, factor=[0.8, 0.9, 1.1, 1.2]):

        for file in os.listdir(path):
            if file.endswith('.wav'):
                random_speed = random.sample(factor, 1)[0]
                data, sr = librosa.load(os.path.join(path, file))
                speed_data = librosa.effects.time_stretch(data, random_speed)
                save_path = os.path.join(output_path, file[:-4] + '_speed_shifted.wav')
                sf.write(save_path, speed_data, sr, format='wav')

    # 소리 음높이 변형
    def pitch_shifting(self, path, output_path, factor=[-2, -1, 1, 2]):

        for file in os.listdir(path):
            if file.endswith('.wav'):
                random_pitch = random.sample(factor, 1)[0]
                data, sr = librosa.load(os.path.join(path, file))
                pitch_data = librosa.effects.pitch_shift(data,sr,n_steps=random_pitch)
                save_path = os.path.join(output_path, file[:-4] + '_pitch_shifted.wav')
                sf.write(save_path, pitch_data, sr, format='wav')


if __name__ == '__main__':
    speaking_sliced_file_path = r'../dataset/speaking/sliced'
    other_sound_sliced_file_path = r'../dataset/other_sound/sliced'
    speaking_augmentation_path = r'../dataset/speaking/augmentation'
    other_sound_augmentation_path = r'../dataset/other_sound/augmentation'

    aug = Augmentation()
    # random room impulse response
    aug.impulse_response(speaking_sliced_file_path, speaking_augmentation_path)
    aug.impulse_response(other_sound_sliced_file_path, other_sound_augmentation_path)

    # 소리 속도 변형
    aug.speed_shifting(speaking_sliced_file_path, speaking_augmentation_path)
    aug.speed_shifting(other_sound_sliced_file_path, other_sound_augmentation_path)

    # 소리 음높이 변형
    aug.pitch_shifting(speaking_sliced_file_path, speaking_augmentation_path)
    aug.pitch_shifting(other_sound_sliced_file_path, other_sound_augmentation_path)

    