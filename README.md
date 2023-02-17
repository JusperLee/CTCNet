# An Audio-Visual Speech Separation Model Inspired by Cortico-Thalamo-Cortical Circuits

#### Kai Li, Fenghua Xie, Hang Chen, Kexin Yuan, and Xiaolin Hu | Tsinghua University

PyTorch Implementation of [CTCNet (Arxiv'22)](https://arxiv.org/pdf/2212.10744v1.pdf): An Audio-Visual Speech Separation Model Inspired by Cortico-Thalamo-Cortical Circuits.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.10744v1) [![GitHub Stars](https://img.shields.io/github/stars/JusperLee/CTCNet?style=social)](https://github.com/JusperLee/CTCNet) ![visitors](https://visitor-badge.laobi.icu//badge?page_id=JusperLee/CTCNet)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-audio-visual-speech-separation-model/speech-separation-on-lrs2)](https://paperswithcode.com/sota/speech-separation-on-lrs2?p=an-audio-visual-speech-separation-model) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-audio-visual-speech-separation-model/speech-separation-on-lrs3)](https://paperswithcode.com/sota/speech-separation-on-lrs3?p=an-audio-visual-speech-separation-model) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/an-audio-visual-speech-separation-model/speech-separation-on-voxceleb2)](https://paperswithcode.com/sota/speech-separation-on-voxceleb2?p=an-audio-visual-speech-separation-model)

## Audio-visual demos

https://user-images.githubusercontent.com/33806018/208616615-dab6ab87-def1-405a-897e-a3c1decb790a.mp4

## Key points

- The performance of multimodal speech separation is greatly improved.
- Incorporating brain inspiration into network design to improve model performance.
- For real scenes can still get better results.

## Quick Started

### Datasets and Pretrained Models 

This method involves using the LRS2, LRS3, and Vox2 datasets to create a multimodal speech separation dataset. The corresponding folders [Datasets/](Datasets/) in the provided GitHub repository contain the files necessary to build the datasets, and the code in the [repository](https://github.com/JusperLee/LRS3-For-Speech-Separation) can be used to construct the multimodal datasets.

The generated datasets (LRS2-2Mix, LRS3-2Mix, and VoxCeleb2-2Mix) can be downloaded at the links below.

| Datasets |  Links  | Pretrained Models  |
| ------------ | ------------ |------------ |
| LRS2-2Mix  | [Baidu Driver](https://pan.baidu.com/s/1FejWqmaYMejOt_8W1TVW4A) Password: **v6bi**  | [Google Driver](https://drive.google.com/file/d/1WtcpYYr8nMiIpJ1epnuGNk2DtiacUXDf/view?usp=sharing)|
| LRS3-2Mix  |  [Baidu Driver](https://pan.baidu.com/s/1FejWqmaYMejOt_8W1TVW4A) Password: **v6bi** |[Google Driver](https://drive.google.com/file/d/19OqNxY1jaO8USM-KnAIJ9B0Rh_Uwh5Ji/view?usp=sharing)|
| VoxCeleb2-2Mix |  [Baidu Driver](https://pan.baidu.com/s/1FejWqmaYMejOt_8W1TVW4A) Password: **v6bi** |[Google Driver](https://drive.google.com/file/d/1jFHC6R51tpqyUd81LYM8Tg45NbbjWkwV/view?usp=sharing) |

### Video Pretrain model

This pre-trained model is a lip-reading model trained only on videos, and it achieves an accuracy of 84% on the LRW dataset.

| Datasets |  Links  | Pretrained Models  |
| ------------ | ------------ |------------ |
| LRS2-2Mix  | [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) | [Google Driver](https://drive.google.com/file/d/13-T3nBnf21-lMKrV_XbH6Lf4vK2xU7lS/view?usp=sharing)|

### Dependencies

- torch   1.13.1+cu116
- torchaudio              0.13.1+cu116
- torchvision             0.14.1+cu116
- pytorch-lightning       1.8.4.post0
- torch-mir-eval          0.4
- torch-optimizer         0.3.0
- fast-bss-eval           0.1.4
- pandas                  1.5.1
- rich                    10.16.2
- opencv-python           4.6.0.66

### Preprocess

```shell
python preprocess_lrs2.py --in_audio_dir audio/wav16k/min --in_mouth_dir mouths --out_dir data
```

### Training Pipeline

#### Training on the LRS2
```shell
python train.py -c local/lrs2_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml
```

#### Training on the LRS3
```shell
python train.py -c local/lrs3_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml
```

#### Training on the VoxCeleb2
```shell
python train.py -c local/vox2_conf_64_64_3_adamw_1e-1_blocks16_pretrain.yml
```

### Testing Pipeline
```shell
python eval.py --test=local/data/tt --conf_dir=exp/lrs2_64_64_3_adamw_1e-1_blocks8_pretrain/conf.yml
```

### Testing Your Own Videos
```shell
ffmpeg -i ./test_videos/interview.mp4 -filter:v fps=fps=25 ./test_videos/interview25fps.mp4
mv ./test_videos/interview25fps.mp4 ./test_videos/interview.mp4
python ./utils/detectFaces.py --video_input_path ./test_videos/interview.mp4 --output_path ./test_videos/interview/ --number_of_speakers 2 --scalar_face_detection 1.5 --detect_every_N_frame 8
ffmpeg -i ./test_videos/interview.mp4 -vn -ar 16000 -ac 1 -ab 192k -f wav ./test_videos/interview/interview.wav
python ./utils/crop_mouth_from_video.py --video-direc ./test_videos/interview/faces/ --landmark-direc ./test_videos/interview/landmark/ --save-direc ./test_videos/interview/mouthroi/ --convert-gray --filename-path ./test_videos/interview/filename_input/interview.csv
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos: [Asteroid](https://github.com/asteroid-team/asteroid) as described in our code.

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{li2022audio,
  title={An Audio-Visual Speech Separation Model Inspired by Cortico-Thalamo-Cortical Circuits},
  author={Li, Kai and Xie, Fenghua and Chen, Hang and Yuan, Kexin and Hu, Xiaolin},
  journal={arXiv preprint arXiv:2212.10744},
  year={2022}
}
```
