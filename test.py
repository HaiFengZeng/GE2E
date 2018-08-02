import librosa
import torch
import matplotlib.pyplot as plt
from model import SpeakerEncoder
from torch.nn import functional as F
from torch.utils import data as data_utils
from nnmnkwii.datasets import vctk
from util import get_split_mels, split_audio
import random


def test1():
    x_size = 10
    x = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x1 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x2 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x3 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x4 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x5 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x6 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x7 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x8 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    x9 = torch.randn(1, 12).repeat(x_size, 1) + torch.randn(x_size, 12) * 0.05
    y = torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9])
    yy = y.unsqueeze(0).repeat(10, 1, 1)
    c = torch.stack(y.split([x_size] * 10), 0).mean(1, keepdim=True)
    cc = c.repeat(1, x_size * 10, 1)
    cc = cc.permute(1, 0, 2)
    yy = yy.permute(1, 0, 2)
    xx = F.cosine_similarity(cc, yy, dim=-1)
    print(xx)

    se = SpeakerEncoder(12, 10, 10)
    print(se.similarity_matrix(y))
    x = se.similarity_matrix(y)
    plt.figure(figsize=(8, 4))
    plt.imshow(x.data, cmap='gray', interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title('similarity_matrix.png')
    plt.show()


def test2():
    import os
    path = '/home/zeng/work/data/VCTK-Corpus/txt'
    paths = os.listdir(path)
    paths = list(map(lambda x: os.path.join(path, x), paths))
    lens = list(map(lambda x: len(os.listdir(x)), paths))
    print(lens)


class SpeakerDatasets(data_utils.Dataset):
    def __init__(self, speaker_id, paths,seg_length):
        super(SpeakerDatasets, self).__init__()
        self.speaker_id = speaker_id
        self.paths = paths

    def __getitem__(self, index):
        audio,sr = librosa.load(self.paths[index],sr=8000)
        x, index = librosa.effects.trim(audio, top_db=10)
        audios = split_audio(x,seg_length=0.6)
        mels = get_split_mels(audios)
        _id = random.randint(0, len(mels)-1)
        return mels[_id]

    def __len__(self):
        return len(self.paths)


def test3():
    class VCTKDatasets():

        def __init__(self, data_root='/home/zeng/work/data/VCTK-Corpus/'):
            super(VCTKDatasets, self).__init__()
            data_sources = vctk.WavFileDataSource(data_root)
            self.paths = data_sources.collect_files()
            data_sources.speakers.sort()
            self.speakers = data_sources.speakers[:10]
            self.train_speaker = self.speakers[:int(len(self.speakers) * 0.9)]
            self.test_speaker = self.speakers[int(len(self.speakers) * 0.9):]
            self.datasets = {_id: SpeakerDatasets(_id, list(filter(lambda x: 'p' + _id in x, self.paths))) for _id in
                             self.speakers}

        def sample_speakers(self, N, train=True):
            speakers = self.train_speaker if train else self.test_speaker
            speaker_selected = random.sample(speakers, N)
            return [self.datasets[_id] for _id in speaker_selected]

    vctk_datasets = VCTKDatasets()
    dsets = vctk_datasets.sample_speakers(3)[0]
    print(dsets)


data_sources = vctk.WavFileDataSource(data_root='/home/zeng/work/data/VCTK-Corpus/')
speakers = data_sources.speakers
files = data_sources.collect_files()
print(files)
speaker_path = list(filter(lambda x: 'p295' in x, files))
sd = SpeakerDatasets('p295',speaker_path)
