# coding:utf-8
from os.path import join
import os


def parse_speaker_info(data_root=''):
    speaker_info = join(data_root, 'SPEAKERS.TXT')
    with open(speaker_info) as f:
        lines = f.readlines()
        lines = list(filter(lambda x: '|' in x and ';' not in x and len(x.strip('\n')) > 0, lines))
        lines = list(map(lambda x: x.strip('\n'), lines))
        datasets = []
        for line in lines:
            info = line.split('|')
            ID, sex, dataset, minutes, speaker_name = info[0], info[1], info[2], info[3], ' '.join(info[4:])
            datasets.append(dataset)
        datasets = list(set(datasets))
        datasets = list(map(lambda x: x.strip(' '), datasets))
        current_datasets = None
        for dataset in datasets:
            if os.path.exists(join(data_root, dataset)):
                current_datasets = dataset
        if not current_datasets:
            raise Exception("bad datasets")

        return join(data_root, current_datasets)


def parse_text_audio_path(train_rate=0.9, val_rate=0.05,
                          train='train.txt',
                          val='val.txt',
                          test='test.txt'):
    path = parse_speaker_info()
    speakers = os.listdir(path)
    texts = list(map(lambda x: list(map(lambda y: x + '-' + y + '.trans.txt', os.listdir(join(path, x)))), speakers))
    i = 0
    train_list, val_list, test_list = [], [], []
    for txts in texts:
        if i < len(speakers) * train_rate:
            for txt in txts:
                p = '/'.join(txt.split('.')[0].split('-'))
                with open(join(path, p + '/' + txt)) as f:
                    lines = f.readlines()
                    lines = list(map(lambda x: join(path, p)+'/' + x[:len(p) + 5] + '.flac|' + x[len(p) + 6:], lines))
                    train_list += lines
        elif i < len(speakers) * (train_rate + val_rate):
            print('validate ', txts)
            for txt in txts:
                p = '/'.join(txt.split('.')[0].split('-'))
                with open(join(path, p + '/' + txt)) as f:
                    lines = f.readlines()
                    lines = list(map(lambda x: join(path, p)+'/' + x[:len(p) + 5] + '.flac|' + x[len(p) + 6:], lines))
                    val_list += lines
        else:
            print('test ', txts)
            for txt in txts:
                p = '/'.join(txt.split('.')[0].split('-'))
                with open(join(path, p + '/' + txt)) as f:
                    lines = f.readlines()
                    lines = list(map(lambda x: join(path, p)+'/' + x[:len(p) + 5] + '.flac|' + x[len(p) + 6:], lines))
                    test_list += lines
        i += 1
    with open(train, 'w') as f:
        f.writelines(train_list)
    with open(test, 'w') as f:
        f.writelines(test_list)
    with open(val, 'w') as f:
        f.writelines(val_list)


parse_text_audio_path()
