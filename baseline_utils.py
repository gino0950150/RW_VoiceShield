import torch
import torch.nn as nn
from torch import Tensor
from tqdm import trange
import numpy as np
from FreeVC import utils
from FreeVC.speaker_encoder.params_data import *
import torchaudio.transforms as T

def emb_attack(
    smodel, cmodel, net_g, vc_src, vc_tgt, adv_tgt,eps,n_iters, hps
):  
    spectrogram = T.Spectrogram(
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
    ).cuda()
    melscale = T.MelScale(n_mels = mel_n_channels,sample_rate=sampling_rate, n_stft=int(sampling_rate * mel_window_length / 1000)//2 + 1).cuda()
    
    with torch.no_grad():
        org_emb = smodel.embed_utterance(vc_tgt)
        tgt_emb = smodel.embed_utterance(adv_tgt)

    wav_tgt = smodel.wav_preprocess(vc_tgt)
    wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
    org_emb = torch.from_numpy(org_emb).cuda()
    tgt_emb = torch.from_numpy(tgt_emb).cuda()
    
    tgt_spec = spectrogram(wav_tgt) + 1e-10
    
    """for plot"""
    ori_mel_spec = melscale(tgt_spec).detach().cpu().squeeze().numpy()
    ori_mel_spec = np.log10(ori_mel_spec)
    
    tgt_spec = torch.log10(tgt_spec)

    mean = torch.mean(tgt_spec, dim=2, keepdim=True)
    std = torch.std(tgt_spec, dim=2, keepdim=True)
    normalized_tgt_spec = (tgt_spec - mean) / std

    ptb = torch.zeros_like(tgt_spec).normal_(-1, 1).requires_grad_(True)
    opt = torch.optim.Adam([ptb])
    criterion = nn.MSELoss()
    pbar = trange(n_iters)

    for _ in pbar:
        adv_inp = normalized_tgt_spec + eps * ptb.tanh()
        adv_inp = adv_inp * std + mean
        adv_inp = torch.pow(10.0, adv_inp)
        adv_inp = melscale(adv_inp)
        adv_inp = torch.transpose(adv_inp, 1, 2)
        adv_emb = smodel.mel_embed_utterance(adv_inp)

        loss = criterion(adv_emb, tgt_emb) - 0.1 * criterion(adv_emb, org_emb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    adv_inp = normalized_tgt_spec + eps * ptb.tanh()
    adv_inp = adv_inp * std + mean
    adv_inp = torch.pow(10.0, adv_inp)
    adv_inp = melscale(adv_inp)
    
    """"for plot """
    adv_mel_spec = adv_inp.detach().cpu().squeeze().numpy()
    adv_mel_spec = np.log10(adv_mel_spec)
    
    adv_inp = torch.transpose(adv_inp, 1, 2)
    adv_emb = smodel.mel_embed_utterance(adv_inp)
    # print(adv_emb.shape)
    # input()
    adv_output =  _
    # adv_output = adv_output[0][0].data.cpu().float().numpy()
    
    griffinLim = T.GriffinLim(
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        length = wav_tgt.shape[1],
    )

    a = (normalized_tgt_spec + eps * ptb.tanh())*std + mean
    a = torch.pow(10.0, a).detach().cpu()
    adv_input = griffinLim(a)
    adv_input = adv_input[0].data.cpu().float().numpy()
    
    """for plot"""
    adv_input_ = torch.from_numpy(adv_input).unsqueeze(0).cuda()
    adv_input_mel_spec = spectrogram(adv_input_)
    adv_input_mel_spec = melscale(adv_input_mel_spec).detach().cpu().squeeze().numpy()
    adv_input_mel_spec = np.log10(adv_input_mel_spec)

    adv_feed_back_output = inference(vc_src, adv_input, smodel, cmodel, net_g)
    adv_feed_back_output = adv_feed_back_output[0][0].data.cpu().float().numpy()
    org_out = inference(vc_src, vc_tgt, smodel, cmodel, net_g)
    org_out = org_out[0][0].data.cpu().float().numpy()
    
    normalized_tgt_spec = normalized_tgt_spec.detach().cpu().squeeze().numpy()
    noise = (eps * ptb.tanh()).squeeze().detach().cpu().numpy()
    wav_tgt = wav_tgt.detach().cpu().squeeze().numpy()
    
    
    
    
    return adv_input, wav_tgt, adv_feed_back_output, org_out, normalized_tgt_spec , noise, ori_mel_spec, adv_mel_spec, adv_input_mel_spec

def inference(src, tgt, smodel, cmodel, net_g):
    g_tgt = smodel.embed_utterance(tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    wav_src = torch.from_numpy(src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)
    audio = net_g.infer(c, g=g_tgt)
    return audio 

def emb_inference(src, emb, cmodel, net_g):
    g_tgt = emb.unsqueeze(0)
    wav_src = torch.from_numpy(src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)
    audio = net_g.infer(c, g=g_tgt)
    return audio
