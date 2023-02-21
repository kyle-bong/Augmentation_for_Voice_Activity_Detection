# Dataset Augmentation for Voice Activity Detection

VAD task를 위한 Dataset augmentation 기법을 모은 repository입니다. <br>
VAD(Voice Activity Detection)는 특정 오디오 구간이 사람의 말소리인지 아닌지를 분류하는 task입니다. <br>
따라서 VAD 모델 학습을 위해서는 '사람의 말소리' 데이터와 '기타 소리' 데이터가 필요합니다. <br>
이 repository의 코드에서는 '사람의 말소리'를 'speaking'으로, '기타 소리'를 'other_sound'로 지칭합니다. 

## Install
```
cd install
bash install.sh
```

## Preprocessing
먼저, 오디오 데이터에서 노이즈 및 무음 구간을 제거하고, 256ms 단위로 자릅니다.
노이즈 및 무음 구간이 제거된 파일은 `dataset/{각 클래스의 경로/trimmed`에 저장되며, 256ms 단위로 잘린 파일은 `dataset/{각 클래스의 경로/sliced}`에 저장됩니다. 
```
cd preprocessing
python preprocessing.py
```

## Augmentation
Augmentation 순서는 다음과 같습니다. <br>
첫째, 다양한 반향이 적용된 데이터를 증강하기 위해 무작위 Room impulse response를 생성합니다. <br>
무작위로 생성된 반향 파일들은 `dataset/IR`에 저장됩니다.
```
cd augmentation
python room_impulse_response.py
```

둘째, 원본 오디오 데이터에 무작위 반향을 적용하고, 음높이를 무작위로 조절하고, 속도를 무작위로 조절합니다. 이 과정은 `augmentation.py`로 진행합니다. <br>
증강된 파일들은 `dataset/{각 클래스의 경로/augmentation}`에 저장됩니다.
```
python augmentation.py
```

