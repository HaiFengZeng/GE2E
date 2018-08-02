import tensorflow as tf
import numpy as np

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    name="GE2E",

    # Presets known to work good.
    # NOTE: If specified, override hyper parameters with preset
    preset="",
    presets={},

    # Input type:
    input_type="raw",
    # Train and test
    train_path='/home/zeng/work/mywork/Speaker_Verification/train_tisv',
    test_path='/home/zeng/work/mywork/Speaker_Verification/test_tisv',
    tdsv_frame=80,
    tisv_frame=180,
    # Audio:
    sample_rate=22050,
    # this is only valid for mulaw is True
    silence_threshold=2,
    num_mels=40,
    fmin=125,
    fmax=7600,
    fft_size=512,
    # shift can be specified by either hop_size or frame_shift_ms
    hop=100,
    frame_shift_ms=None,
    min_level_db=-100,
    ref_level_db=20,
    window=0.025,

    rescaling=True,
    rescaling_max=0.999,
    # mel-spectrogram is normalized to [0, 1] for each utterance and clipping may
    # happen depends on min_level_db and ref_level_db, causing clipping noise.
    # If False, assertion is added to ensure no clipping happens.
    allow_clipping_in_normalization=True,

    # Mixture of logistic distributions:
    log_scale_min=float(np.log(1e-14)),

    # Model:
    N=20,
    M=10,
    mode='TI-SV',

    hidden_size_tisv=768,
    project_size_tisv=256,
    hidden_size_tidv=128,
    project_size_tidv=64,
    # If True, apply weight normalization as same as DeepVoice3
    weight_normalization=True,
    # this should only be enabled for multi-speaker dataset
    n_speakers=7,  # 7 for CMU ARCTIC
    # Data loader
    pin_memory=True,
    num_workers=2,

    # train/test
    # test size can be specified as portion or num samples
    test_size=0.0441,  # 50 for CMU ARCTIC single speaker
    test_num_samples=None,
    random_state=1234,

    # Loss
    loss_type='softmax',
    # Training:
    batch_size=1,  # real batch_size = N*M
    steps=3 * 1e6,
    initial_learning_rate=1e-2,
    # see lrschedule.py for available lr_schedule
    nepochs=2000,
    learning_rate_decay=0.5,
    clip_thresh=3,
    # max time steps can either be specified as sec or steps
    # This is needed for those who don't have huge GPU memory...
    # if both are None, then full audio samples are used
    max_time_sec=None,
    max_time_steps=8000,
    # Hold moving averaged parameters and use them for evaluation
    exponential_moving_average=True,
    # averaged = decay * averaged + (1 - decay) * x
    ema_decay=0.9999,

    # Save
    # per-step intervals
    checkpoint_interval=1000,
    train_eval_interval=1000,
    # per-epoch interval
    test_eval_epoch_interval=5,
    save_optimizer_state=True,

    # Eval:
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
