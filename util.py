# coding: utf-8
from __future__ import with_statement, print_function, absolute_import
from ge2e_hparams import hparams
import os
import numpy as np
import librosa
import random
from os.path import join


def _assert_valid_input_type(s):
    assert s == "mulaw-quantize" or s == "mulaw" or s == "raw"


def is_mulaw_quantize(s):
    _assert_valid_input_type(s)
    return s == "mulaw-quantize"


def is_mulaw(s):
    _assert_valid_input_type(s)
    return s == "mulaw"


def is_raw(s):
    _assert_valid_input_type(s)
    return s == "raw"


def is_scalar_input(s):
    return is_raw(s) or is_mulaw(s)


def keyword_spot(spec):
    """ Keyword detection for data preprocess
        For VTCK data I truncate last 80 frames of trimmed audio - "Call Stella"
    :return: 80 frames spectrogram
    """
    return spec[:, -hparams.tdsv_frame:]


def random_batch(speaker_num=hparams.N,
                 utter_num=hparams.M,
                 train=True,
                 shuffle=True, noise_filenum=None, utter_start=0):
    """ Generate 1 batch.
        For TD-SV, noise is added to each utterance.
        For TI-SV, random frame length is applied to each batch of utterances (140-180 frames)
        speaker_num : number of speaker of each batch
        utter_num : number of utterance per speaker of each batch
        shuffle : random sampling or not
        noise_filenum : specify noise file or not (TD-SV)
        utter_start : start point of slicing (TI-SV)
    :return: 1 random numpy batch (frames x batch(NM) x n_mels)
    """

    # data path
    if train:
        path = hparams.train_path
    else:
        path = hparams.test_path

    # TD-SV
    if hparams.mode == 'TD-SV':
        np_file = os.listdir(path)[0]
        path = os.path.join(path, np_file)  # path of numpy file
        utters = np.load(path)  # load text specific utterance spectrogram
        if shuffle:
            np.random.shuffle(utters)  # shuffle for random sampling
        utters = utters[:speaker_num]  # select N speaker

        # concat utterances (M utters per each speaker)
        # ex) M=2, N=2 => utter_batch = [speaker1, speaker1, speaker2, speaker2]
        utter_batch = np.concatenate([np.concatenate([utters[i]] * utter_num, axis=1) for i in range(speaker_num)],
                                     axis=1)

        if noise_filenum is None:
            noise_filenum = np.random.randint(0, hparams.noise_filenum)  # random selection of noise
        noise = np.load(os.path.join(hparams.noise_path, "noise_%d.npy" % noise_filenum))  # load noise

        utter_batch += noise[:, :utter_batch.shape[1]]  # add noise to utterance

        utter_batch = np.abs(utter_batch) ** 2
        mel_basis = librosa.filters.mel(sr=hparams.sr, n_fft=hparams.nfft, n_mels=40)
        utter_batch = np.log10(np.dot(mel_basis, utter_batch) + 1e-6)  # log mel spectrogram of utterances

        utter_batch = np.array([utter_batch[:, hparams.tdsv_frame * i:hparams.tdsv_frame * (i + 1)]
                                for i in range(speaker_num * utter_num)])  # reshape [batch, n_mels, frames]

    # TI-SV
    else:
        np_file_list = os.listdir(path)
        total_speaker = len(np_file_list)

        if shuffle:
            selected_files = random.sample(np_file_list, speaker_num)  # select random N speakers
        else:
            selected_files = np_file_list[:speaker_num]  # select first N speakers

        utter_batch = []
        for file in selected_files:
            utters = np.load(os.path.join(path, file))  # load utterance spectrogram of selected speaker
            if shuffle:
                utter_index = np.random.randint(0, utters.shape[0], utter_num)  # select M utterances per speaker
                utter_batch.append(utters[utter_index])  # each speakers utterance [M, n_mels, frames] is appended
            else:
                utter_batch.append(utters[utter_start: utter_start + utter_num])

        utter_batch = np.concatenate(utter_batch, axis=0)  # utterance batch [batch(NM), n_mels, frames]

        if train:
            frame_slice = np.random.randint(140, 181)  # for train session, random slicing of input batch
            utter_batch = utter_batch[:, :, :frame_slice]
        else:
            utter_batch = utter_batch[:, :, :160]  # for train session, fixed length slicing of input batch

    utter_batch = np.transpose(utter_batch, axes=(2, 0, 1))  # transpose [frames, batch, n_mels]

    return utter_batch


def test_random_batch(audio_path, speakers=None, sentences=[], speaker_num=5, tisv_frame=100,
                      sr=8000, utter_num=15, window=0.025, n_fft=512,
                      hop_lenght=0.01, mel=40, same_sentence=False):
    seg_len = (tisv_frame * hop_lenght + window) / 2
    folders = []
    for i, folder in enumerate(os.listdir(audio_path)):
        folders.append(folder)
    choices = random.sample(folders, speaker_num)
    audios_file = list(map(lambda x: join(audio_path, x), choices))
    speaker_paths = []
    for path in audios_file:
        # print(os.listdir(path))
        audios_path = list(map(lambda x: join(path, x), os.listdir(path)))
        # print(audios_path)
        selected_path = audios_path if same_sentence else list(np.random.permutation(audios_path))
        speaker_paths.append(selected_path)
    # print(len(speaker_paths))
    mel_rs = []
    # perm = np.random.permutation(range(len(speaker_paths[0])))
    # print(perm)
    for speaker in speaker_paths:
        _mel = []
        for i in range(len(speaker)):
            path =  speaker[i]
            # print(path)
            x, sr = librosa.load(path, sr)
            x, index = librosa.effects.trim(x, top_db=10)
            audios = split_audio(x, sr, seg_len)
            mels = get_split_mels(audios, sr, n_fft=n_fft, win_length=window, hop_length=hop_lenght, mel=mel)
            _mel += mels
            if len(_mel) > utter_num:
                _mel = _mel[:utter_num]
                mel_rs += _mel
                break
    mel_rs = np.stack(mel_rs, 0)
    return mel_rs


def get_split_mels(splited_audios, sr=22050, n_fft=512, win_length=0.025, hop_length=0.01, mel=40):
    log_mels = []
    for audio in splited_audios:
        S = librosa.core.stft(y=audio, n_fft=n_fft, win_length=int(win_length * sr), hop_length=int(sr * hop_length))
        S = np.abs(S) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)
        log_mels.append(S)
    return log_mels


def split_audio(x, sr=22050, seg_length=0.8, pad=False):
    l = x.shape[0] / sr
    L = int(l / seg_length)
    audio_list = []
    for i in range(L - 1):
        audio_list.append(x[int(i * seg_length * sr):int((i + 2) * seg_length * sr)])
    return audio_list


if __name__ == '__main__':
    batch = random_batch(4, 5)
    print(batch.shape)
