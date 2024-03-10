import numpy as np
import os
import librosa 
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
from confidence_intervals import evaluate_with_conf_int
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


if __name__ == '__main__':
    path = "/mnt/sda/jinyu/attack_vc/baseline"
    dirs = os.listdir(path)
    sr = 16000
    
    snrs = []
    pesqs = []
    stois = []
    L_spk = []
    for dir in tqdm(dirs):
        p = os.path.join(path, dir)
        spk = int(dir.split("_")[-1])
        clean, _= librosa.load(os.path.join(p, "ori_input.wav"), sr = sr)
        noisy, _= librosa.load(os.path.join(p, "adv_input.wav"), sr = sr)
        snrs.append(SNR(clean, noisy))
        stois.append(stoi(clean, noisy, 16000, extended=True))
        pesqs.append(pesq(sr, clean, noisy, 'wb'))
        L_spk.append(spk)
        
    print(f"snr: {np.mean(np.array(snrs))}")
    print(f"pesq: {np.mean(np.array(pesqs))}")
    print(f"stoi: {np.mean(np.array(stois))}")
    num_bootstraps = 1000
    alpha = 5
    print(L_spk)
    with open(os.path.join(path, 'conf_c.txt'), "a") as f:
        f.writelines([str(evaluate_with_conf_int(np.asarray(snrs), np.average, labels=None, conditions=L_spk, 
                       num_bootstraps=num_bootstraps, alpha=alpha)), str(evaluate_with_conf_int(np.asarray(pesqs), np.average, labels=None, conditions=L_spk, 
                       num_bootstraps=num_bootstraps, alpha=alpha)), str(evaluate_with_conf_int(np.asarray(stois), np.average, labels=None, conditions=L_spk, 
                       num_bootstraps=num_bootstraps, alpha=alpha))])