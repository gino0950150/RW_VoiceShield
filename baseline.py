import argparse

import soundfile as sf
import torch

import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import sys
sys.path.append("FreeVC")

import FreeVC.utils as vc_utils
from FreeVC.models import SynthesizerTrn
from FreeVC.wavlm import WavLM, WavLMConfig
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder
import logging
from baseline_utils import emb_attack 
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import utils
from speechbrain.pretrained import EncoderClassifier
from confidence_intervals import evaluate_with_conf_int

# from data_utils import denormalize, file2mel, load_model, mel2wav, normalize


def main(
    vc_tgt: str,
    adv_tgt: str,
    output: str,
    vc_src: str,
    eps: float,
    n_iters: int,
    attack_type: str,
    hpfile:str,
    ptfile:str,
    txtpath:str,
    use_timestamp:bool,
):

    hps = vc_utils.get_hparams_from_file(hpfile)

    with open(txtpath, "r") as f :
        lines = f.readlines()
        
    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = vc_utils.load_checkpoint(ptfile, net_g, None, True)

    print("Loading WavLM for content...")
    cmodel = vc_utils.get_cmodel(0)
    
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        smodel = SpeakerEncoder('FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt')
    emb_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    index = 0   
    L_attack_fail = []
    L_preserve_success = []
    L_spk = []
    spk_dic = {}
    spk_idx = 0
    for line in lines:
        vc_src, vc_tgt, adv_tgt= line.strip().split()
        spk = vc_tgt.split("/")[-2]
        if spk not in spk_dic.keys():
            spk_dic[spk] = int(spk_idx)
            spk_idx += 1
        out_path = os.path.join(output, f"{str(index)}_{str(spk_dic[spk])}")

        vc_tgt, _ = librosa.load(vc_tgt,sr=hps.data.sampling_rate)
        vc_tgt = librosa.util.normalize(vc_tgt)
        vc_tgt, _ = librosa.effects.trim(vc_tgt, top_db=20)
        
        
        adv_tgt, _ = librosa.load(adv_tgt,sr=hps.data.sampling_rate)
        adv_tgt = librosa.util.normalize(adv_tgt)
        adv_tgt, _ = librosa.effects.trim(adv_tgt, top_db=20)
        
        vc_src, _ = librosa.load(vc_src,sr=hps.data.sampling_rate)

        if attack_type == "e2e":
            adv_input, adv_output, ori_out= e2e_attack(smodel, cmodel, net_g, vc_src, vc_tgt, adv_tgt, eps, n_iters, hps)
        if attack_type == "emb":
            adv_input, ori_input, adv_feed_back_output, ori_out, spec_ori, noise, ori_mel_spec, adv_mel_spec, adv_input_mel_spec= emb_attack(smodel, cmodel, net_g, vc_src, vc_tgt, adv_tgt, eps, n_iters, hps)
        if attack_type == "self":
            adv_input = self_attack(smodel, cmodel, net_g, vc_src, vc_tgt, adv_tgt, eps, n_iters, hps)
        os.makedirs(os.path.join(output, f"{str(index)}_{str(spk_dic[spk])}"), exist_ok=True)
        write(os.path.join(out_path, "adv_input.wav"), hps.data.sampling_rate, adv_input)
        write(os.path.join(out_path, "ori_input.wav"), hps.data.sampling_rate, ori_input)
        write(os.path.join(out_path, "adv_tgt.wav"), hps.data.sampling_rate, adv_tgt)
        # write(os.path.join(out_path, "adv_output.wav"), hps.data.sampling_rate, adv_output)
        write(os.path.join(out_path, "before.wav"), hps.data.sampling_rate, ori_out)
        if attack_type == "emb":
            # write(os.path.join(out_path, "adv_output.wav"), hps.data.sampling_rate, adv_output)
            write(os.path.join(out_path, "vc_src.wav"), hps.data.sampling_rate, vc_src)
            write(os.path.join(out_path, "ori_out.wav"), hps.data.sampling_rate, ori_out)
            write(os.path.join(out_path, "adv_feed_back_output.wav"), hps.data.sampling_rate, adv_feed_back_output)
            
            attack_fail = utils.sv(os.path.join(out_path, "ori_input.wav"), os.path.join(out_path, "adv_feed_back_output.wav"), emb_model)
            preserve_success = utils.sv(os.path.join(out_path, "ori_input.wav"), os.path.join(out_path, "adv_input.wav"), emb_model)
            L_attack_fail.append(not attack_fail)
            L_preserve_success.append(preserve_success)
            L_spk.append(spk_dic[spk])
            
            # if attack_fail or not preserve_success:
            #     shutil.rmtree(out_path)
            # print("attak success rate : ", 1 - np.array(L_attack_fail).mean())
            # print("preserve success rate : ", np.array(L_preserve_success).mean())
        

            # spec_ori_fig = sns.heatmap(spec_ori).get_figure()
            # spec_ori_fig.savefig(os.path.join(out_path, "spec_ori.png"))
            # plt.clf()
            # noise_fig = sns.heatmap(noise).get_figure()
            # noise_fig.savefig(os.path.join(out_path, "noise.png"))
            # plt.clf()
            # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # axs[0].imshow(ori_mel_spec , aspect='auto', cmap='viridis', origin='lower')
            # axs[0].set_title('ori_mel_spec')
            # axs[1].imshow(adv_mel_spec, aspect='auto', cmap='viridis', origin='lower')
            # axs[1].set_title('adv_mel_spec')
            # axs[2].imshow(adv_input_mel_spec, aspect='auto', cmap='viridis', origin='lower')
            # axs[2].set_title('adv_input_mel_spec')
            # plt.tight_layout()
            # fig.savefig(os.path.join(out_path, "compare.png"))
            # plt.clf()
            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # axs[0].imshow(ori_mel_spec - adv_input_mel_spec, aspect='auto', cmap='viridis', origin='lower')
            # axs[0].set_title('ori_mel_spec - adv_input_mel_spec')
            # axs[1].imshow(ori_mel_spec - adv_mel_spec, aspect='auto', cmap='viridis', origin='lower')
            # axs[1].set_title('ori_mel_spec - adv_mel_spec')
            # plt.tight_layout()
            # fig.savefig(os.path.join(out_path, "compare_error.png"))
            # plt.clf()   
        index += 1
    # elif attack_type == "emb":
    #     adv_inp = emb_attack(model, vc_tgt, adv_tgt, eps, n_iters)
    # elif attack_type == "fb":
    #     adv_inp = fb_attack(model, vc_src, vc_tgt, adv_tgt, eps, n_iters)
    # else:
    #     raise NotImplementedError()
    num_bootstraps = 1000
    alpha = 5
    print("attak success rate : ", evaluate_with_conf_int(np.asarray(L_attack_fail), np.average, labels=None, conditions=L_spk, 
                       num_bootstraps=num_bootstraps, alpha=alpha))
    print("preserve success rate: ", evaluate_with_conf_int(np.asarray(L_preserve_success), np.average, labels=None, conditions=L_spk, 
                       num_bootstraps=num_bootstraps, alpha=alpha))
    # adv_inp = adv_inp.squeeze(0).T
    # adv_inp = denormalize(adv_inp.data.cpu().numpy(), attr)
    # adv_inp = mel2wav(adv_inp, **config["preprocess"])

    # sf.write(output, adv_inp, config["preprocess"]["sample_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vc_tgt",
        type=str,
        default = "/homes/jinyu/dataset/VCTK/wav/p225/p225_001_mic1.wav",
        help="The target utterance to be defended, providing vocal timbre in voice conversion.",
    )
    parser.add_argument(
        "--adv_tgt",type=str, default = "/homes/jinyu/dataset/VCTK/wav/p226/p226_001_mic1.wav", help="The target used in adversarial attack."
    )
    parser.add_argument("--output", type=str, default = "/mnt/sda/jinyu/attack_vc/baseline/", help="The output defended utterance.")
    parser.add_argument(
        "--vc_src",
        type=str,
        default="/homes/jinyu/dataset/VCTK/wav/p227/p227_001_mic1.wav",
        help="The source utterance providing linguistic content in voice conversion (required in end-to-end and feedback attack).",
    ) 
    parser.add_argument(
        "--eps",
        type=float,
        default=0.3,
        help="The maximum amplitude of the perturbation.",
    )
    parser.add_argument(
        "--n_iters",
        type=int,
        default=1500,
        help="The number of iterations for updating the perturbation.",
    )
    parser.add_argument(
        "--attack_type",
        type=str,
        choices=["e2e", "emb", "fb", "self"],
        default="emb",
        help="The type of adversarial attack to use (end-to-end, embedding, or feedback attack).",
    )
    parser.add_argument(
        "--hpfile", 
        type=str, 
        default="/homes/jinyu/attack_vc/FreeVC/configs/freevc.json", 
        help="path to json config file"
    )
    parser.add_argument(
        "--ptfile", 
        type=str, 
        default="/homes/jinyu/attack_vc/FreeVC/checkpoints/freevc.pth", 
        help="path to pth file"
    )
    parser.add_argument(
        "--txtpath", 
        type=str, 
        default="/homes/jinyu/attack_vc/data/test_pairs.txt", 
        help="path to txt file"
    )
    parser.add_argument(
        "--use_timestamp", 
        default=False, 
        action="store_true"
    )
    main(**vars(parser.parse_args()))
