import os
import math
import random 
import glob


with open("/homes/jinyu/dataset/VCTK/speaker-info.txt", "r") as f:
    lines = f.readlines()

root = "/homes/jinyu/dataset/VCTK/wav16"
target_pth = "/homes/jinyu/attack_vc/data" 
lines = [line.rstrip() for line in lines]
lines = lines[1:-1]

m_spks, f_spks = [], []

for line in lines:
    L = line.split()
    if L[2] == "M":
        m_spks.append(L[0])
    else:
        f_spks.append(L[0])
        
train_spks, val_spks, test_spks = [], [], []
random.shuffle(m_spks)
random.shuffle(f_spks)

train_spks += m_spks[:math.ceil(len(m_spks)*0.6)]
val_spks += m_spks[math.ceil(len(m_spks)*0.6):math.ceil(len(m_spks)*0.8)]
test_spks += m_spks[math.ceil(len(m_spks)*0.8):]
print(f"split: {len(train_spks)}, {len(val_spks)}, {len(test_spks)}")

train_spks += f_spks[:math.ceil(len(f_spks)*0.6)]
val_spks += f_spks[math.ceil(len(f_spks)*0.6):math.ceil(len(f_spks)*0.8)]
test_spks += f_spks[math.ceil(len(f_spks)*0.8):]
print(f"split: {len(train_spks)}, {len(val_spks)}, {len(test_spks)}")

lines = []
for spk in train_spks:
    lines += glob.glob(os.path.join(root, spk) + '/*.wav')
lines = [line + "\n" for line in lines]
with open(os.path.join(target_pth, "train.txt"), "w") as f:
    f.writelines(lines)

lines = []
for spk in val_spks:
    lines += glob.glob(os.path.join(root, spk) + '/*.wav')
lines = [line + "\n" for line in lines]
with open(os.path.join(target_pth, "val.txt"), "w") as f:
    f.writelines(lines)

lines = []
for spk in test_spks:
    lines += glob.glob(os.path.join(root, spk) + '/*.wav')
lines = [line + "\n" for line in lines]
with open(os.path.join(target_pth, "test.txt"), "w") as f:
    f.writelines(lines)

