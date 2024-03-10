import argparse
import os
import time
from functools import partial

import torch
import pickle
import numpy as np
import sys
sys.path.append("FreeVC")
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from nnAudio import Spectrogram

import model.utils as model_utils
import utils
from data.dataset import VCTKDataset
from model.waveunet import Waveunet
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder
import multiprocessing
from FreeVC.speaker_encoder.params_data import *
from tqdm import tqdm
from FreeVC.models import SynthesizerTrn
from FreeVC import utils as vc_utils
import soundfile as sf
import random
import glob
from resemblyzer import VoiceEncoder
from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

def main(args):
    smodel = SpeakerEncoder('/homes/jinyu/attack_freevc/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').cuda()
    hps = vc_utils.get_hparams_from_file(args.hpfile)
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = vc_utils.load_checkpoint(args.ptfile, net_g, None, True)
    cmodel = vc_utils.get_cmodel(0)
    emb_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    root = "/homes/jinyu/dataset/VCTK/wav16/"
    with open("/homes/jinyu/dataset/VCTK/speaker-info.txt", "r") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    lines = lines[1:-1]
    spk_info = {} 
    for line in lines:
        L = line.split()
        spk_info[L[0]] = L[2]

    print(spk_info)
    m_spks, f_spks = [], []
    with open("/homes/jinyu/attack_vc/data/test.txt", "r") as f:
        lines = f.readlines()
    lines = [line.rstrip() for line in lines]
    for line in lines:
        spk = line.split("/")[-2]
        if spk_info[spk] == "M" and (spk not in m_spks):
            m_spks.append(spk)
        elif spk_info[spk] == "F" and (spk not in f_spks):
            f_spks.append(spk)
    print(len(m_spks), len(f_spks))
    
    m_spk = random.choice(f_spks)
    f_spk = random.choice(f_spks)
    
    print(m_spk)
    print(f_spk)

    m_wavs = glob.glob(os.path.join(root, m_spk) + '/*.wav')
    f_wavs = glob.glob(os.path.join(root, f_spk) + '/*.wav')
    random.shuffle(m_wavs)
    i = 0
    for tar_path in m_wavs:
        for j in range(10):
            while True:
                src_path = random.choice(lines)
                if src_path.split("/")[-2] != spk:
                    break
            out = utils.vc_infer(src_path, tar_path, smodel, cmodel, net_g, hps, vc_utils)
            sf.write('out.wav', out, args.sr)
            if utils.sv(tar_path, 'out.wav', emb_model):
                adv_path = random.choice(f_wavs)
                with open("/homes/jinyu/attack_vc/data/test_pairs_tsne_f_f.txt", "a") as f:
                    f.writelines([f"{src_path} {tar_path} {adv_path}\n"])
                i = i + 1
                break
        if i >= 50:
            break
                
            
        
        
if __name__ == '__main__':
    ## TRAIN PARAMETERS
    parser = argparse.ArgumentParser()
    parser.add_argument('--instruments', type=str, nargs='+', default=["adv_noise"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loader worker threads (default: 1)')
    parser.add_argument('--features', type=int, default=32,
                        help='Number of feature channels per layer')
    parser.add_argument('--log_dir', type=str, default='logs/waveunet',
                        help='Folder to write logs into')
    parser.add_argument('--dataset_dir', type=str, default="/mnt/windaten/Datasets/MUSDB18HQ",
                        help='Dataset path')
    parser.add_argument('--hdf_dir', type=str, default="hdf",
                        help='Dataset path')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/waveunet',
                        help='Folder to write checkpoints into')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Reload a previously trained model (whole task model)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate in LR cycle (default: 1e-3)')
    parser.add_argument('--min_lr', type=float, default=5e-5,
                        help='Minimum learning rate in LR cycle (default: 5e-5)')
    parser.add_argument('--cycles', type=int, default=2,
                        help='Number of LR cycles per epoch')
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number of DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=16000,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=1,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=1.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping on validation set")
    parser.add_argument('--example_freq', type=int, default=200,
                        help="Write an audio summary into Tensorboard logs every X training iterations")
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--conv_type', type=str, default="gn",
                        help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser.add_argument('--res', type=str, default="fixed",
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=0,
                        help="Train separate model for each source (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default="double",
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")
    parser.add_argument('--start_delta', type=float, default=0.1,
                        help="level of start noise")
    parser.add_argument("--hpfile", type=str, default="FreeVC/configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="FreeVC/checkpoints/freevc.pth", help="path to pth file")
    parser.add_argument("--exp_dir", type=str, default="exps/1105", help="path to pth file")
    args = parser.parse_args()

    main(args)
