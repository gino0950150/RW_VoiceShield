import numpy as np 
from sklearn.manifold import TSNE
from tqdm import tqdm 
from pathlib import Path
import pickle 
import matplotlib.pyplot as plt
from speechbrain.pretrained import EncoderClassifier
import os
import torchaudio
from tqdm import tqdm

def emb(path, emb_model):
    signal, fs =torchaudio.load(path)
    emb = emb_model.encode_batch(signal)
    return emb

root = "/mnt/sda/jinyu/attack_vc/exps/0217_blackbox512/result_tsne"
emb_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

fig, ax = plt.subplots(2,2, dpi=150)
dic = {"m_m": [0,0], "f_m" : [0,1], "m_f": [1,0], "f_f":[1,1]}
dic_2 = {"m_m": "Male to Male", "f_m" : "Female to Male", "m_f": "Male to Female", "f_f":"Female to Female"}
plt.rcParams["font.family"] = "serif"
for key in dic.keys():
    X = []
    Y = []
    for i in tqdm(range(50)):
        dir = os.path.join(f"{root}_{key}", str(i))
        X.append(emb(os.path.join(dir, "adv_input.wav"),emb_model).squeeze().numpy())
        Y.append(0)
        X.append(emb(os.path.join(dir, "after.wav"),emb_model).squeeze().numpy())
        Y.append(1)
        X.append(emb(os.path.join(dir, "ori_input.wav"),emb_model).squeeze().numpy())
        Y.append(2)
        X.append(emb(os.path.join(dir, "before.wav"),emb_model).squeeze().numpy())
        Y.append(3)
        X.append(emb(os.path.join(dir, "adv_spk.wav"),emb_model).squeeze().numpy())
        Y.append(4)

    X = np.array(X)
    Y = np.array(Y)
    D = {0:"adv. input", 1: "adv. output", 2: "pro. input", 3: "ori. output", 4: "target speaker"}

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    colors = plt.cm.rainbow(np.linspace(0, 1, 5))
    Y_color = [ "#ffd670", "#1982c4", "#ff595e", "#ffca3a","#ffd670"]
    label = [D[i] for i in Y]
    z0, z1 = dic[key]
    for i in range(1,4):
        idx = np.where(Y == i)
        ax[z0,z1].scatter(x = X_tsne[idx,0], y =X_tsne[idx,1], c = Y_color[i], label = label[i])
    ax[z0,z1].set_title(dic_2[key])
    ax[z0,z1].get_xaxis().set_visible(False)
    ax[z0,z1].get_yaxis().set_visible(False)
    # ax[z0,z1].legend(loc='upper center')
lines = [] 
labels = [] 
for ax_ in fig.axes: 
    Line, Label = ax_.get_legend_handles_labels() 
    # print(Label) 
    lines.extend(Line) 
    labels.extend(Label) 
# fig.legend(lines[:3], labels[:3], loc='lower center') 
fig.legend(lines[:3], labels[:3],loc='upper center', bbox_to_anchor=(0.5, 0.1),
          fancybox=True, shadow=True, ncol=3)
plt.savefig("tsne.jpg", bbox_inches="tight")



