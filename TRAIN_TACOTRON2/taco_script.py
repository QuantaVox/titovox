import os
get_ipython().system('nvidia-smi')

from numba import cuda 
device = cuda.get_current_device()
device.reset()


from pydub import AudioSegment
import os
from os.path import exists
import sys
get_ipython().system('pip install phonemizer')
get_ipython().system('pip install ffmpeg-normalize')
get_ipython().system('pip install git+https://github.com/wkentaro/gdown.git')
get_ipython().system('git clone -q https://github.com/rmcpantoja/tacotron2.git')
sys.path.append('tacotron2')
get_ipython().run_line_magic('cd', 'content/tacotron2')
get_ipython().system('git clone -q --recursive https://github.com/SortAnon/hifi-gan')
sys.path.append('hifi-gan')
get_ipython().system('pip install git+https://github.com/savoirfairelinux/num2words')
get_ipython().system('git submodule init')
get_ipython().system('git submodule update')
get_ipython().system('pip install matplotlib numpy inflect librosa scipy unidecode pillow tensorboardX')
get_ipython().system('apt-get install pv')
get_ipython().system('apt-get -qq install sox')
get_ipython().system('apt-get install jq')
get_ipython().system('wget https://raw.githubusercontent.com/tonikelope/megadown/master/megadown -O megadown.sh')
get_ipython().system('chmod 755 megadown.sh')

get_ipython().run_line_magic('matplotlib', 'inline')

import IPython.display as ipd
import json
from layers import TacotronSTFT
from audio_processing import griffin_lim
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import Generator
from denoiser import Denoiser
import resampy
import scipy.signal

import os
if os.getcwd() != '/content/tacotron2':
    os.chdir('tacotron2')
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
 
import random
import numpy as np

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence
from math import e
#from tqdm import tqdm # Terminal
#from tqdm import tqdm_notebook as tqdm # Legacy Notebook TQDM
from tqdm.notebook import tqdm # Modern Notebook TQDM
from distutils.dir_util import copy_tree
import matplotlib.pylab as plt

get_ipython().run_line_magic('cd', '/content/')
def get_hifigan(MODEL_ID, conf_name):
    # Download HiFi-GAN
    hifigan_pretrained_model = 'hifimodel_' + conf_name
    #gdown.download(d+MODEL_ID, hifigan_pretrained_model, quiet=False)

    if MODEL_ID == "universal":
      get_ipython().system('wget "https://github.com/johnpaulbin/tacotron2/releases/download/Main/g_02500000" -O $hifigan_pretrained_model')
    else:
      get_ipython().system('gdown --id "$MODEL_ID" -O $hifigan_pretrained_model')

    # Load HiFi-GAN
    conf = os.path.join("hifi-gan", conf_name + ".json")
    with open(conf) as f:
        json_config = json.loads(f.read())
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    hifigan = Generator(h).to(torch.device("cuda"))
    state_dict_g = torch.load(hifigan_pretrained_model, map_location=torch.device("cuda"))
    hifigan.load_state_dict(state_dict_g["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()
    denoiser = Denoiser(hifigan, mode="normal")
    return hifigan, h, denoiser
 
# Download character HiFi-GAN
hifigan, h, denoiser = get_hifigan("universal", "config_v1")
# Download super-resolution HiFi-GAN
hifigan_sr, h2, denoiser_sr = get_hifigan("14fOprFAIlCQkVRxsfInhEPG0n-xN4QOa", "config_32k")

get_ipython().run_line_magic('cd', '/content/tacotron2')

def download_from_google_drive(file_id, file_name):
  # download a file from the Google Drive link
  get_ipython().system('rm -f ./cookie')
  get_ipython().system('curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id={file_id}" > /dev/null')
  confirm_text = get_ipython().getoutput("awk '/download/ {print $NF}' ./cookie")
  confirm_text = confirm_text[0]
  get_ipython().system('curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm={confirm_text}&id={file_id}" -o {file_name}')

def create_mels():
    print("Generating Mels")
    stft = layers.TacotronSTFT(
                hparams.filter_length, hparams.hop_length, hparams.win_length,
                hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
                hparams.mel_fmax)
    def save_mel(filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != stft.sampling_rate:
            raise ValueError("{} {} SR does not match the objective {} SR".format(filename, 
                sampling_rate, stft.sampling_rate))
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).cpu().numpy()
        np.save(filename.replace('.wav', ''), melspec)

    import glob
    wavs = glob.glob('wavs/*.wav')
    for i in tqdm(wavs):
        save_mel(i)


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(trainset, num_workers=1, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    try:
        torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)
    except KeyboardInterrupt:
        print("interrupt received while saving, waiting for save to complete.")
        torch.save({'iteration': iteration,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),'learning_rate': learning_rate}, filepath)
    print("Model Saved")

def plot_alignment(alignment, info=None):
    get_ipython().run_line_magic('matplotlib', 'inline')
    fig, ax = plt.subplots(figsize=(int(alignment_graph_width/100), int(alignment_graph_height/100)))
    im = ax.imshow(alignment, cmap='inferno', aspect='auto', origin='lower',
                   interpolation='none')
    ax.autoscale(enable=True, axis="y", tight=True)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    fig.canvas.draw()
    plt.show()

def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank, epoch, start_eposh, learning_rate, sample_interbal, save_audio = False, audio_path = None):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Epoch: {} Validation loss {}: {:9f}  Time: {:.1f}m LR: {:.6f}".format(epoch, iteration, val_loss,(time.perf_counter()-start_eposh)/60, learning_rate))
        logger.log_validation(val_loss, model, y, y_pred, iteration)
        if hparams.show_alignments:
            get_ipython().run_line_magic('matplotlib', 'inline')
            _, mel_outputs, gate_outputs, alignments = y_pred
            idx = random.randint(0, alignments.size(0) - 1)
            plot_alignment(alignments[idx].data.cpu().numpy().T)

    dv = epoch/sample_interbal
    if dv.is_integer():
      print(f"Generación de muestras... \n{sampletext}")
      for i in [x for x in sampletext.split("\n") if len(x)]:
          if i[-1] != ";": i=i+";" 
          with torch.no_grad():
              sequence = np.array(text_to_sequence(i, ['basic_cleaners']))[None, :]
              sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
              mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
              y_g_hat = hifigan(mel_outputs_postnet.float())
              audio = y_g_hat.squeeze()
              audio = audio * MAX_WAV_VALUE
              audio_denoised = denoiser(audio.view(1, -1), strength=35)[:, 0]
 
              # Resample to 32k
              audio_denoised = audio_denoised.cpu().numpy().reshape(-1)
 
              normalize = (MAX_WAV_VALUE / np.max(np.abs(audio_denoised))) ** 0.9
              audio_denoised = audio_denoised * normalize
              wave = resampy.resample(
                  audio_denoised,
                  h.sampling_rate,
                  h2.sampling_rate,
                  filter="sinc_window",
                  window=scipy.signal.windows.hann,
                  num_zeros=8,
              )
              wave_out = wave.astype(np.int16)
 
              # HiFi-GAN super-resolution
              wave = wave / MAX_WAV_VALUE
              wave = torch.FloatTensor(wave).to(torch.device("cuda"))
              new_mel = mel_spectrogram(
                  wave.unsqueeze(0),
                  h2.n_fft,
                  h2.num_mels,
                  h2.sampling_rate,
                  h2.hop_size,
                  h2.win_size,
                  h2.fmin,
                  h2.fmax,
              )
              y_g_hat2 = hifigan_sr(new_mel)
              audio2 = y_g_hat2.squeeze()
              audio2 = audio2 * MAX_WAV_VALUE
              audio2_denoised = denoiser(audio2.view(1, -1), strength=35)[:, 0]
                  
              # High-pass filter, mixing and denormalizing
              audio2_denoised = audio2_denoised.cpu().numpy().reshape(-1)
              b = scipy.signal.firwin(
                  101, cutoff=10500, fs=h2.sampling_rate, pass_zero=False
              )
              y = scipy.signal.lfilter(b, [1.0], audio2_denoised)
              y *= 0
              y_out = y.astype(np.int16)
              y_padded = np.zeros(wave_out.shape)
              y_padded[: y_out.shape[0]] = y_out
              sr_mix = wave_out + y_padded
              sr_mix = sr_mix / normalize

              print("")
              ipd.display(ipd.Audio(sr_mix.astype(np.int16), rate=h2.sampling_rate))
              if save_audio:
                if not os.path.isdir(audio_path+"/audio samples"):
                  get_ipython().system('os.makedirs(audio_path+"/audio samples")')
                scipy.io.wavfile.write(audio_path+"/audio samples/_"+epoch+"test.wav", h2.sampling_rate, sr_mix.astype(np.int16))
                wav2mp3(audio_path+"/audio samples/_"+epoch+"test.wav")
