from model import SpeakerEncoder
from util import random_batch, get_split_mels, split_audio, test_random_batch
from data_preprocess import get_feature
import librosa
import numpy as np
from train import load_checkpoint
import torch
import matplotlib.pyplot as plt

speaker = 30
utter = 10
speaker_encoder = SpeakerEncoder(40, n=speaker, m=utter)
checkpoint = torch.load('/home/zeng/work/mywork/GE2E/checkpoints/checkpoint_step000112000_ema.pth')
audio_path = '/home/zeng/work/data/VCTK-Corpus/wav48/p364/p364_001.wav'
speaker1_path = '/home/zeng/work/data/VCTK-Corpus/wav48/p230/p230_008.wav'
speaker2_path = '/home/zeng/work/data/VCTK-Corpus/wav48/p225/p225_011.wav'
speaker3_path = '/home/zeng/work/data/VCTK-Corpus/wav48/p226/p226_022.wav'
speaker4_path = '/home/zeng/work/data/VCTK-Corpus/wav48/p345/p345_018.wav'
wav_path = '/home/zeng/work/data/VCTK-Corpus/wav48/'
speaker_encoder.load_state_dict(checkpoint["state_dict"])


def test_tisv():
    x = random_batch(speaker_num=10, utter_num=10, train=False)
    print(x.shape)
    x = torch.from_numpy(x).float()
    d_vector, sim_matrix = speaker_encoder(x)
    print(d_vector)
    plt.figure(figsize=(8, 4))
    plt.imshow(sim_matrix.data, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('similarity_matrix.png')
    plt.show()


def test_audio_speaker_encoder(sr=8000):
    mels = []
    paths = [speaker1_path, speaker2_path, speaker3_path, speaker4_path]
    for audio_path in paths:
        x, sr = librosa.load(audio_path, sr=sr)
        x, index = librosa.effects.trim(x, 10)
        a = (160 * 0.01 + 0.025) / 4
        audios = split_audio(x, sr=sr,seg_length=a)
        mels += get_split_mels(audios, sr=sr,)[:6]

    print(mels)
    mels = np.stack(mels, axis=0)
    mels = np.transpose(mels, [0, 2, 1])
    mels = mels.transpose(1, 0, 2)

    speaker_encoder = SpeakerEncoder(40, 4, 6)
    mels = torch.from_numpy(mels).float()
    d_vector, sim_matrix = speaker_encoder(mels)
    plt.figure(figsize=(8, 4))
    plt.imshow(sim_matrix.data, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('similarity_matrix.png')
    plt.show()


def test_get_test_random_batch():
    mels = test_random_batch(wav_path, speaker_num=speaker,
                             utter_num=utter, hop_lenght=0.005,tisv_frame=80)
    mels = mels.transpose(0, 2, 1)
    mels = mels.transpose(1, 0, 2)
    x = torch.from_numpy(mels).float()
    print(x.shape)
    d_vector, sim_matrix = speaker_encoder(x)
    print(d_vector)
    plt.figure(figsize=(8, 4))
    plt.imshow(sim_matrix.data, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('similarity_matrix.png')
    plt.show()


# test_audio_speaker_encoder()
# test_tisv()
# x = get_feature(audio_path)
# print(len(x))
test_get_test_random_batch()
# test_audio_speaker_encoder()