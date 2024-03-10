import numpy as np
import librosa
import torch
from pathlib import Path
from resemblyzer import preprocess_wav
import torchaudio

def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def set_cyclic_lr(optimizer, it, epoch_it, cycles, min_lr, max_lr):
    cycle_length = epoch_it // cycles
    curr_cycle = min(it // cycle_length, cycles-1)
    curr_it = it - cycle_length * curr_cycle

    new_lr = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos((float(curr_it) / float(cycle_length)) * np.pi))
    set_lr(optimizer, new_lr)

def vc_infer(src_pth, tar_pth, smodel, cmodel, net_g, hps, utils):
    wav_tgt, _ = librosa.load(tar_pth, sr=hps.data.sampling_rate)
    g_tgt = smodel.embed_utterance(wav_tgt)
    g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).cuda()
    wav_src, _ = librosa.load(src_pth, sr=hps.data.sampling_rate)
    wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
    c = utils.get_content(cmodel, wav_src)
    audio = net_g.infer(c, g=g_tgt)
    audio = audio[0][0].data.cpu().float().numpy()
    return audio
# def sv(path1, path2, emb_model):
#     path1 = Path(path1)
#     wav1 = preprocess_wav(path1)
#     emb1 = emb_model.embed_utterance(wav1)
#     path2 = Path(path2)
#     wav2 = preprocess_wav(path2)
#     emb2 = emb_model.embed_utterance(wav2)
#     sim = np.dot(emb1,emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
#     if sim >= 0.683:
#         return True
#     else:
#         return False
def sv(wav1, wav2, classifier):
    signal1, fs =torchaudio.load(wav1)
    emb1 = classifier.encode_batch(signal1)
    signal2, fs =torchaudio.load(wav2)
    emb2 = classifier.encode_batch(signal2)
    sim = np.dot(emb1.squeeze().numpy(),emb2.squeeze().numpy())/(np.linalg.norm(emb1.squeeze().numpy())*np.linalg.norm(emb2.squeeze().numpy()))
    if sim >= 0.328:
        return True
    else:
        return False