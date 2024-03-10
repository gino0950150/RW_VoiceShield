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
from data.dataset import VCTKFeedBackDataset
from model.waveunet import Waveunet
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder
import multiprocessing
from FreeVC.speaker_encoder.params_data import *
from tqdm import tqdm
from FreeVC.models import SynthesizerTrn
from FreeVC import utils as vc_utils
import soundfile as sf
import random
def main(args):
    #torch.backends.cudnn.benchmark=True # This makes dilated conv much faster for CuDNN 7.5

    # MODEL
    multiprocessing.set_start_method('spawn')
    num_features = [args.features*i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
                   [args.features*2**i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)
    model = Waveunet(args.channels, num_features, args.channels, args.instruments, kernel_size=args.kernel_size,
                     target_output_size=target_outputs, depth=args.depth, strides=args.strides,
                     conv_type=args.conv_type, res=args.res, separate=args.separate)

    print("Loading speaker encoder...")
    smodel = SpeakerEncoder('/homes/jinyu/attack_freevc/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt').cuda()
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
    cmodel.train()

    ### DATASET
    # musdb = get_musdb_folds(args.dataset_dir)
    # # If not data augmentation, at least crop targets to fit model output shape
    # crop_func = partial(crop_targets, shapes=model.shapes)
    # Data augmentation function for training
    train_data = VCTKFeedBackDataset("/homes/jinyu/attack_vc/data/train.txt", args.instruments, args.sr, args.channels, model.shapes, True, smodel)
    val_data = VCTKFeedBackDataset("/homes/jinyu/attack_vc/data/val.txt", args.instruments, args.sr, args.channels, model.shapes, True, smodel)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)
    val_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=utils.worker_init_fn)

    ##### TRAINING ####

    # Set up the loss function
    # if args.loss == "L1":
    #     criterion = nn.L1Loss()
    # elif args.loss == "L2":
    #     criterion = nn.MSELoss()
    # else:
    #     raise NotImplementedError("Couldn't find this loss!")
    criterion = torch.nn.CosineEmbeddingLoss()

    # Set up optimiser
    optimizer = Adam(params=model.parameters(), lr=args.lr)

    # Set up training state dict that will also be saved into checkpoints
    state = {"step" : 0,
             "worse_epochs" : 0,
             "epochs" : 0,
             "best_loss" : np.Inf,
             "delta": args.start_delta}

    # LOAD MODEL CHECKPOINT IF DESIRED
    if args.load_model is not None:
        print("Continuing training full model from checkpoint " + str(args.load_model))
        state = model_utils.load_model(model, optimizer, args.load_model, args.cuda)

    
    delta = state["delta"]
    
    print('TRAINING START')
    while state["worse_epochs"] < args.patience:
        print("Training one epoch from iteration " + str(state["step"]))
        model.train()
        smodel.train()
        net_g.train()
        cmodel.train()
        losses = []
        for example_num, (x, src_emb, tar_emb, c_wav, src_spk, tar_spk) in tqdm(enumerate(train_dataloader)):
            # Set LR for this iteration
            utils.set_cyclic_lr(optimizer, example_num, len(train_data) // args.batch_size, args.cycles, args.min_lr, args.lr)

            optimizer.zero_grad()
            x = x.cuda()
            c_wav = c_wav.cuda()
            noise = model(x, delta = delta, src_emb = src_emb, tar_emb = tar_emb)["adv_noise"].squeeze()
            noise = noise - noise.mean(dim = 1).unsqueeze(1).repeat(1,noise.shape[1])
            m = torch.abs(noise).max(dim = 1)[0].unsqueeze(1).repeat(1,noise.shape[1])
            noise = torch.div(noise, m)*delta

            # print("max:", noise[0,:].max(), "min", noise[0,:].min())
            x = x.squeeze()
            x_adv = x[:,model.shapes["output_start_frame"]:model.shapes["output_end_frame"]] + noise
            mel = Spectrogram.MelSpectrogram(
                sr=sampling_rate,
                n_fft=int(sampling_rate * mel_window_length / 1000),
                hop_length=int(sampling_rate * mel_window_step / 1000),
                n_mels=mel_n_channels,
                verbose=False
            ).cuda()
            mel_x = mel(x_adv)
            
            mel_x = torch.transpose(mel_x, 1, 2)
            adv_emb = smodel.mel_embed_utterance(mel_x)
            c = cmodel.extract_features(c_wav.squeeze(1))[0]
            c = c.transpose(1, 2)
            y = net_g.infer(c, g=adv_emb)
            
            mel_y = mel(y)
            mel_y = torch.transpose(mel_y, 1, 2)
            adv_emb_feedback = smodel.mel_embed_utterance(mel_y)
            
            loss = criterion(adv_emb_feedback, tar_emb, torch.ones(adv_emb_feedback.shape[0]).cuda()) + 0.1*criterion(adv_emb_feedback, src_emb, -1*torch.ones(adv_emb_feedback.shape[0]).cuda())
            loss.backward()
            optimizer.step()
            state["step"] += 1
            losses.append(loss.item())
            if state["step"] % 10000 == 0:
                dir = os.path.join(args.exp_dir, str(state["step"]))
                os.makedirs(dir, exist_ok= True)
                src = x[0,model.shapes["output_start_frame"]:model.shapes["output_end_frame"]].cpu().numpy()
                sf.write(os.path.join(dir, f'ori_input.wav'), src, args.sr)
                adv = x_adv[0,:].detach().cpu().numpy()
                sf.write(os.path.join(dir, 'adv_input.wav'), adv, args.sr)
                sf.write(os.path.join(dir, 'noise.wav'), noise[0,:].detach().cpu().numpy(), args.sr)
                c = random.choice(train_data.audiolist)
                before = utils.vc_infer(c, os.path.join(dir, f'ori_input.wav'), smodel, cmodel, net_g, hps, vc_utils)
                after = utils.vc_infer(c, os.path.join(dir, 'adv_input.wav'), smodel, cmodel, net_g, hps, vc_utils)
                sf.write(os.path.join(dir, 'before.wav'), before, args.sr)
                sf.write(os.path.join(dir, 'after.wav'), after, args.sr)
                print("loss: ", np.array(losses).mean(), noise.max(), noise.min())  
                with open(os.path.join(args.exp_dir, "record.txt"), "a") as f:
                    f.writelines([f"iter {state['step']} delta {delta} loss: {np.array(losses).mean()},{noise[0,:].max()},{noise[0,:].min()}\n"])
                losses = []
                state["delta"] = delta
                model_utils.save_model(model, optimizer, state, os.path.join(dir, f'ckpt_{state["step"]}'))
                
            if state["step"] % 100000 == 0 and (delta - 0.025) > 0.01:
                delta -= 0.025
                with open(os.path.join(args.exp_dir, "record.txt"), "a") as f:
                    f.writelines([f"delta reduced to {delta}\n"])
            elif state["step"] % 100000 == 0 and (delta - 0.025) <= 0.01: 
                delta = 0.0100
                with open(os.path.join(args.exp_dir, "record.txt"), "a") as f:
                    f.writelines([f"delta reduced to {delta}\n"])
                
                
                
                
                    


        # # VALIDATE
        # val_loss = validate(args, model, criterion, val_data)
        # print("VALIDATION FINISHED: LOSS: " + str(val_loss))
        # writer.add_scalar("val_loss", val_loss, state["step"])

        # # EARLY STOPPING CHECK
        # checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_" + str(state["step"]))
        # if val_loss >= state["best_loss"]:
        #     state["worse_epochs"] += 1
        # else:
        #     print("MODEL IMPROVED ON VALIDATION SET!")
        #     state["worse_epochs"] = 0
        #     state["best_loss"] = val_loss
        #     state["best_checkpoint"] = checkpoint_path

        # state["epochs"] += 1
        # # CHECKPOINT
        # print("Saving model...")
        # model_utils.save_model(model, optimizer, state, checkpoint_path)

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
    parser.add_argument("--exp_dir", type=str, default="/mnt/sda/jinyu/attack_vc/exps/1206_feedback", help="path to pth file")
    args = parser.parse_args()

    main(args)
