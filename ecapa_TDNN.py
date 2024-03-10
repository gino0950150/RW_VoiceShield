import torchaudio
from speechbrain.pretrained import EncoderClassifier
import numpy as np 
import os 
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")


# signal1, fs =torchaudio.load('/mnt/sda/jinyu/attack_vc/exps/0107_from_snr20/1680000/adv_input.wav')
# emb1 = classifier.encode_batch(signal1)
# signal2, fs =torchaudio.load('/mnt/sda/jinyu/attack_vc/exps/0107_from_snr20/1680000/ori_input.wav')
# emb2 = classifier.encode_batch(signal2)


# sim = np.dot(emb1.squeeze().numpy(),emb2.squeeze().numpy())/(np.linalg.norm(emb1.squeeze().numpy())*np.linalg.norm(emb2.squeeze().numpy()))
# print(sim)
import numpy as np
import sklearn.metrics
import glob
import random 
from tqdm import tqdm
"""
Python compute equal error rate (eer)
ONLY tested on binary classification

:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""
def sim(wav1, wav2, classifier):
    signal1, fs =torchaudio.load(wav1)
    emb1 = classifier.encode_batch(signal1)
    signal2, fs =torchaudio.load(wav2)
    emb2 = classifier.encode_batch(signal2)


    sim = np.dot(emb1.squeeze().numpy(),emb2.squeeze().numpy())/(np.linalg.norm(emb1.squeeze().numpy())*np.linalg.norm(emb2.squeeze().numpy()))
    return sim 
def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label = positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer, eer_threshold

if __name__ == '__main__':
    vctk_path = "/homes/jinyu/dataset/VCTK/wav16"
    spks = os.listdir(vctk_path)
    label, pred = [], []
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    all_wavs = glob.glob("/homes/jinyu/dataset/VCTK/wav16/*/*.wav")
    
    for spk in tqdm(spks):
        wavs = glob.glob(os.path.join(os.path.join(vctk_path, spk), "*.wav"))
        N = 256
        if len(wavs) < 256:
            N = len(wavs)
        wavs_ = random.sample(wavs, N)
        random.shuffle(wavs_)
        P_wavs = wavs_[ :N//2]
        N_wavs = wavs_[N//2:]
        for wav in P_wavs:
            while True:
                wav_ = random.sample(wavs, 1)[0]
                if wav_ is not wav:
                    break
            s = sim(wav, wav_, classifier)
            label.append(1)
            pred.append(s)
        for wav in N_wavs:
            while True:
                wav_ = random.sample(all_wavs,1)[0]
                spk_ = wav_.split("/")[-2]
                if spk is not spk_:
                    break
            s = sim(wav, wav_, classifier)
            label.append(0)
            pred.append(s)
eer, thre = compute_eer(label, pred)                 
print(f"EER: {eer} thre: {thre}")            
                
        