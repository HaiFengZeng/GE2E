"""Trainining script for WaveNet vocoder

usage: train.py [options]

options:
    --data-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys
from os.path import dirname, join
from tqdm import tqdm, trange
from datetime import datetime
from model import SpeakerEncoder, GE2ELoss

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler
import numpy as np

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii.datasets import vctk
from os.path import join, expanduser
import random
import librosa.display
from matplotlib import pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn

from util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input, random_batch

import audio
from ge2e_hparams import hparams, hparams_debug_string

fs = hparams.sample_rate

global_step = 0
global_test_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False


def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=0)
    return x


class VCTK():
    def __init__(self, vctk_path='/home/zeng/work/data/VCTK-Corpus/',
                 data_root='data',
                 mode='TD-SV'):
        self.data_root = data_root
        self.speakers = vctk.WavFileDataSource(data_root=vctk_path)
        speakers = self.speakers.labelmap.keys()
        self.speakerDataSource = {'p' + speaker_id: PyTorchDataset_TISV(
            _NPYDataSource(data_root=data_root,
                           col='',
                           speaker_id='p' + speaker_id,
                           text_index=None if mode == 'TD-SV' else '001',
                           train=True,
                           test_size=0.1
                           )) for speaker_id in speakers}
        self.speakerDataSource_index = {}
        self.speakerDataLoader = {
            _id: data_utils.DataLoader(dataset=self.speakerDataSource[_id], batch_size=hparams.M,
                                       shuffle=True)
            for _id in self.speakerDataSource.keys()
        }
        for index, key in enumerate(self.speakerDataSource.keys()):
            self.speakerDataSource_index[index] = key

    def sample_speakers(self, N=hparams.N):
        # sample N speakers
        x = range(len(self.speakerDataSource_index.keys()))
        speakers = np.random.choice(x, size=N)
        return [self.speakerDataSource[self.speakerDataSource_index[x]] for x in speakers]


class _NPYDataSource(FileDataSource):
    def __init__(self, data_root,
                 col,
                 speaker_id=None,
                 text_index='001',  # key words
                 train=True,
                 test_size=0.05,
                 test_num_samples=None,
                 random_state=1234):
        self.data_root = data_root
        self.text_index = text_index
        self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = True
        self.speaker_ids = None
        self.train = train
        self.test_size = test_size
        self.test_num_samples = test_num_samples
        self.random_state = random_state

    def interest_indices(self, paths):
        indices = np.arange(len(paths))
        if self.test_size is None:
            test_size = self.test_num_samples / len(paths)
        else:
            test_size = self.test_size
        train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=self.random_state)
        return train_indices if self.train else test_indices

    def collect_files(self):
        meta = join(self.data_root, "train.txt")
        with open(meta, "r") as f:
            lines = f.readlines()
        lines = list(filter(lambda x: self.speaker_id in x, lines))
        if self.text_index:
            lines = list(filter(lambda x: self.speaker_id in x and self.text_index in x, lines))
        self.lengths = list(map(lambda l: int(l.split("|")[1]), lines))
        paths = list(map(lambda l: join(self.data_root, str(l).split('|')[0]), lines))
        if self.text_index: return paths
        # Filter by train/test
        indices = self.interest_indices(paths)
        paths = list(np.array(paths)[indices])
        self.lengths = list(np.array(self.lengths)[indices])
        self.lengths = list(map(int, self.lengths))

        return paths

    def collect_features(self, path):
        return np.load(path)


class RawAudioDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(RawAudioDataSource, self).__init__(data_root, 0, **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, data_root, **kwargs):
        super(MelSpecDataSource, self).__init__(data_root, 1, **kwargs)


class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randmoized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batchs
    """

    def __init__(self, lengths, batch_size=16, batch_group_size=None, permutate=True):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))
        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 32, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0
        self.permutate = permutate

    def __iter__(self):
        indices = self.sorted_indices.clone()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            random.shuffle(indices[s:e])

        # Permutate batches
        if self.permutate:
            perm = np.arange(len(indices[:e]) // self.batch_size)
            random.shuffle(perm)
            indices[:e] = indices[:e].view(-1, self.batch_size)[perm, :].view(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            random.shuffle(indices[s:])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)


# There two datasets for different mode TD-SV OR TI-SV
class PyTorchDataset_TISV(object):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        raw_audio = self.X[idx]
        return raw_audio

    def __len__(self):
        return len(self.X)


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand, requires_grad=False)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


# https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
# https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
class ExponentialMovingAverage(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, x):
        assert name in self.shadow
        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta


def clone_as_averaged_model(model, ema):
    assert ema is not None
    averaged_model = SpeakerEncoder(input_size=hparams.num_mels)
    if use_cuda:
        averaged_model = averaged_model.cuda()
    averaged_model.load_state_dict(model.state_dict())
    for name, param in averaged_model.named_parameters():
        if name in ema.shadow:
            param.data = ema.shadow[name].clone()
    return averaged_model


def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)


def assert_ready_for_upsampling(x, c):
    assert len(x) % len(c) == 0 and len(x) // len(c) == audio.get_hop_size()


def get_audio(x, length=1.6, sr=22050):
    L = int(length * sr)
    s = np.random.randint(0, len(x) - L)
    return x[s:s + L]


def get_log_mels(raw_audio, n_fft=1024, sr=22050, window=100, hop=128, mels=40, n_frames=180):
    S = librosa.core.stft(y=raw_audio, n_fft=n_fft, win_length=window, hop_length=hop)
    S = np.abs(S) ** 2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mels)
    S = np.log10(np.dot(mel_basis, S) + 1e-6)
    S = S[:n_frames]
    return S


def collate_fn(batch):
    """Create batch
    what a batch look like?
    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,) audio
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """
    # handle every batch
    new_batch = []
    for idx in range(len(batch)):
        x = batch[idx]
        x = audio.trim(x)  # we hope the length should be longer than 1.6s
        x = get_audio(x)
        x = get_log_mels(x)
        new_batch.append(x)
    batch = new_batch
    # (B, T, C)

    return batch


def eval_model(global_step, writer, model, y, c, g, input_lengths, eval_dir, ema=None):
    if ema is not None:
        print("Using averaged model for evaluation")
        model = clone_as_averaged_model(model, ema)

    model.eval()
    idx = np.random.randint(0, len(y))
    length = input_lengths[idx].data.cpu().numpy()[0]

    # (T,)
    y_target = y[idx].view(-1).data.cpu().numpy()[:length]

    if c is not None:
        c = c[idx, :, :length].unsqueeze(0)
        assert c.dim() == 3
        print("Shape of local conditioning features: {}".format(c.size()))
    if g is not None:
        # TODO: test
        g = g[idx]
        print("Shape of global conditioning features: {}".format(g.size()))

    # Dummy silence
    if is_mulaw_quantize(hparams.input_type):
        initial_value = P.mulaw_quantize(0, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        initial_value = P.mulaw(0.0, hparams.quantize_channels)
    else:
        initial_value = 0.0
    print("Intial value:", initial_value)

    # (C,)
    if is_mulaw_quantize(hparams.input_type):
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = Variable(torch.from_numpy(initial_input)).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = Variable(torch.zeros(1, 1, 1).fill_(initial_value))
    initial_input = initial_input.cuda() if use_cuda else initial_input
    y_hat = model.incremental_forward(
        initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
        log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
        y_target = P.inv_mulaw_quantize(y_target, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
        y_target = P.inv_mulaw(y_target, hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    # Save audio
    os.makedirs(eval_dir, exist_ok=True)
    path = join(eval_dir, "step{:09d}_predicted.wav".format(global_step))
    librosa.output.write_wav(path, y_hat, sr=hparams.sample_rate)
    path = join(eval_dir, "step{:09d}_target.wav".format(global_step))
    librosa.output.write_wav(path, y_target, sr=hparams.sample_rate)

    # save figure
    path = join(eval_dir, "step{:09d}_waveplots.png".format(global_step))


def save_states(global_step, writer, similarity_matrix, checkpoint_dir=None, step=0):
    print("Save intermediate states at step {}".format(global_step))
    plt.figure(figsize=(12, 6))
    plt.imshow(similarity_matrix, aspect='auto')
    plt.colorbar()
    plt.title('similarity_matrix.png')
    if writer:
        import io
        from PIL import Image
        buff = io.BytesIO()
        plt.savefig(buff, format='png')
        buff.seek(0)
        im = np.array(Image.open(buff))
        writer.add_image('similarity_matrix', im, global_step=step)
    # plt.savefig('similarity_matrix.png')
    plt.close()


def __train_step(phase, epoch, global_step, global_test_step,
                 model, optimizer, writer, criterion, x,
                 checkpoint_dir, eval_dir=None, do_eval=False, ema=None, scheduler=None):
    train = (phase == "train")
    clip_thresh = hparams.clip_thresh
    if train:
        model.train()
        step = global_step
    else:
        model.eval()
        step = global_test_step

    # scheduler.step()
    optimizer.zero_grad()

    d_vector, similarity_matrix = model(x)

    loss = criterion(similarity_matrix)

    if train and step > 0 and step % 200 == 0:
        print(model.w, model.b)
        save_states(step, writer, similarity_matrix.cpu().data, checkpoint_dir, global_step)
    if train and step > 0 and step % hparams.checkpoint_interval == 0:
        save_states(step, writer, similarity_matrix.cpu().data, checkpoint_dir, global_step)
        save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, ema)

    if do_eval and False:
        # NOTE: use train step (i.e., global_step) for filename
        pass
        # eval_model(global_step, writer, model, x, eval_dir, ema)

    # Update
    if train:
        loss.backward()
        if clip_thresh > 0:
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), clip_thresh)
        optimizer.step()
        # update moving average
        if ema is not None:
            for name, param in model.named_parameters():
                if name in ema.shadow:
                    ema.update(name, param.data)

    # Logs
    writer.add_scalar("{} loss".format(phase), float(loss.data[0]), step)
    if train:
        if clip_thresh > 0:
            pass
            # writer.add_scalar("gradient norm", grad_norm, step)
    print('step {},loss = {}'.format(global_step, loss.item()))
    return loss.data[0]


def train_loop(model, optimizer, writer, data_loaders=None, scheduler=None, checkpoint_dir=None):
    if use_cuda:
        model = model.cuda()

    criterion = GE2ELoss()

    if hparams.exponential_moving_average:
        ema = ExponentialMovingAverage(hparams.ema_decay)
        for name, param in model.named_parameters():
            if param.requires_grad:
                ema.register(name, param.data)
    else:
        ema = None
    phase = 'train'
    global global_step, global_epoch, global_test_step
    while global_epoch < hparams.nepochs:
        train = (phase == "train")
        running_loss = 0.
        test_evaluated = False
        for i in range(10000):
            x = torch.from_numpy(random_batch()).float()
            if use_cuda:
                x = x.cuda()
            do_eval = False
            eval_dir = join(checkpoint_dir, "{}_eval".format(phase))
            # Do eval per eval_interval for train
            if train and global_step > 0 \
                    and global_step % hparams.train_eval_interval == 0:
                do_eval = True

            if not train and not test_evaluated \
                    and global_epoch % hparams.test_eval_epoch_interval == 0:
                do_eval = True
                test_evaluated = True
            if do_eval:
                print("[{}] Eval at train step {}".format(phase, global_step))

            # Do step
            running_loss += __train_step(
                phase, global_epoch, global_step, global_test_step, model,
                optimizer, writer, criterion, x,
                checkpoint_dir, eval_dir, do_eval, ema, scheduler)

            # update global state
            if train:
                global_step += 1
            else:
                global_test_step += 1

        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    if ema is not None:
        averaged_model = clone_as_averaged_model(model, ema)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "state_dict": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    global_test_step = checkpoint.get("global_test_step", 0)

    return model


# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = torch.load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))




if __name__ == "__main__":
    # args = docopt(__doc__)
    args = {
        "--checkpoint-dir": 'checkpoints',
        "--checkpoint": None,
        "--restore-parts": None,
        "--mode": 'TD-SV',
        "--data-root": None,
        "--log-event-path": None,
        "--reset-optimizer": True,
        "--hparams": '',
    }
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_restore_parts = args["--restore-parts"]

    data_root = args["--data-root"]
    if data_root is None:
        data_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    print(hparams_debug_string())

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json

        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Model
    model = SpeakerEncoder(input_size=hparams.num_mels)
    print(model)
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams.steps)

    if checkpoint_restore_parts is not None:
        restore_parts(checkpoint_restore_parts, model)

    # Load checkpoints
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test"
    print("Los event path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)

    # Train!
    try:
        train_loop(model, optimizer, writer, checkpoint_dir=checkpoint_dir, scheduler=scheduler)
    except KeyboardInterrupt:
        save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)
