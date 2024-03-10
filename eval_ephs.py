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
from pesq import pesq
from pystoi import stoi

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
import librosa
from test import predict
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import shutil
def SNR(inputSig, outputSig):
    noise = outputSig-inputSig
    
    powS = signalPower(outputSig)
    powN = signalPower(noise)
    
    if(powS-powN) >= 0:
        return 10*np.log10((powS/powN)**2)
    else:
        return -1* 10*np.log10((powS/powN)**2)

def signalPower(x):
    return np.mean(x**2)**0.5

def main(args):
    multiprocessing.set_start_method('spawn')
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    print("Loading speaker encoder...")
    vc_smodel = SpeakerEncoder('/homes/jinyu/attack_freevc/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').cuda()
    smodel = SpeakerEncoder('/mnt/sda/jinyu/attack_vc/spk_encoder/ckpt/blackbox_backups/blackbox_bak_435000.pt').cuda()
    model = model_utils.DataParallel(model)
    print("move model to gpu")
    model.cuda()

    print('model: ', model)
    print('parameter count: ', str(sum(p.numel() for p in model.parameters())))
    hps = vc_utils.get_hparams_from_file(args.hpfile)
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    _ = vc_utils.load_checkpoint(args.ptfile, net_g, None, True)
    cmodel = vc_utils.get_cmodel(0)

    ### DATASET
    # musdb = get_musdb_folds(args.dataset_dir)
    # # If not data augmentation, at least crop targets to fit model output shape
    # crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training

    ##### TRAINING ####

    emb_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf,
             "delta": args.start_delta}
    
    sr = 16000

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)
    
    model_iter_list = [790000, 1030000, 1140000, 1220000, 1360000, 1420000]

    for iter in model_iter_list:
        ckpt = os.path.join(args.exp_dir, f"{str(iter)}/ckpt_{str(iter)}")
        state = model_utils.load_model(model, optimizer, ckpt, args.cuda)
        delta = state["delta"]
        with open("/homes/jinyu/attack_vc/data/test_pairs.txt", "r") as f:
            lines = f.readlines()
        lines = [line.rstrip().split() for line in lines] 
        i = 0
        root = os.path.join(args.exp_dir, "result_eph")
        L_attack_fail = []
        L_preserve_success = []
        L_SNR = []
        L_pesq = []
        L_stoi = []
        
        for src_path, tar_path, adv_path in tqdm(lines):
            
            dir = os.path.join(root, str(i))
            src, _= librosa.load(src_path, sr = args.sr)
            tar, _= librosa.load(tar_path, sr = args.sr)
            adv, _= librosa.load(adv_path, sr = args.sr)
            
            src_emb = smodel.embed_utterance(src)
            src_emb = torch.from_numpy(src_emb).cuda()
            tar_emb = smodel.embed_utterance(tar)
            tar_emb = torch.from_numpy(tar_emb).cuda()
            adv_emb = smodel.embed_utterance(adv)
            adv_emb = torch.from_numpy(adv_emb).cuda()
            
            m = np.abs(tar).max()
            tar =  tar / m * 0.9
            tar = np.expand_dims(tar, axis = 0)
            adv_wav, m = predict(tar, tar_emb.unsqueeze(0), adv_emb.unsqueeze(0), delta, model)
            adv_wav = adv_wav["adv_noise"]
            os.makedirs(dir, exist_ok= True)
            sf.write(os.path.join(dir, "adv_input.wav"), adv_wav.squeeze(), args.sr)
            sf.write(os.path.join(dir, "ori_input.wav"), tar.squeeze()/m, args.sr)
            sf.write(os.path.join(dir, "adv_spk.wav"), adv, args.sr)
            sf.write(os.path.join(dir, "content.wav"), src, args.sr)
            before = utils.vc_infer(src_path, os.path.join(dir, f'ori_input.wav'), vc_smodel, cmodel, net_g, hps, vc_utils)
            after = utils.vc_infer(src_path, os.path.join(dir, f'adv_input.wav'), vc_smodel, cmodel, net_g, hps, vc_utils)
            sf.write(os.path.join(dir, 'before.wav'), before, args.sr)
            sf.write(os.path.join(dir, 'after.wav'), after, args.sr)
            sf.write(os.path.join(dir, 'noise.wav'), adv_wav.squeeze() - tar.squeeze()/m, args.sr)
            
            attack_fail = utils.sv(os.path.join(dir, 'after.wav'), os.path.join(dir, "ori_input.wav"), emb_model)
            preserve_success = utils.sv(os.path.join(dir, 'adv_input.wav'), os.path.join(dir, 'ori_input.wav'), emb_model)
            clean, _= librosa.load(os.path.join(dir, "ori_input.wav"), sr = sr)
            noisy, _= librosa.load(os.path.join(dir, "adv_input.wav"), sr = sr)
            L_SNR.append(SNR(clean, noisy))
            L_stoi.append(stoi(clean, noisy, 16000, extended=True))
            L_pesq.append(pesq(sr, clean, noisy, 'wb'))
            
            L_attack_fail.append(attack_fail)
            L_preserve_success.append(preserve_success)
            if attack_fail or not preserve_success:
                shutil.rmtree(dir)
            i += 1
        print("attak success rate : ", 1 - np.array(L_attack_fail).mean())
        print("preserve success rate : ", np.array(L_preserve_success).mean())
        with open(os.path.join(args.exp_dir, "ephs_results_ecapa.txt"), "a") as f:
            f.writelines([f"iter {str(state['step'])} delta {str(state['delta'])} \
                attak_success_rate {str(1 - np.array(L_attack_fail).mean())} preserve_success_rate {np.array(L_preserve_success).mean()}\
                SNR {np.array(L_SNR).mean()} PESQ {np.array(L_pesq).mean()} STOI {np.array(L_stoi).mean()}\n"])
        
        
        
        
        
    
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
    parser.add_argument('--start_delta', type=float, default=20,
                        help="level of start noise")
    parser.add_argument("--hpfile", type=str, default="FreeVC/configs/freevc.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="FreeVC/checkpoints/freevc.pth", help="path to pth file")
    parser.add_argument("--exp_dir", type=str, default="exps/1105", help="path to pth file")
    args = parser.parse_args()

    main(args)