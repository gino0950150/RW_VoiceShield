"""https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_NeMo.ipynb"""
import os 
from tqdm import tqdm
import json
import librosa
import re

manifest_path = '/homes/jinyu/RW_VoiceShield/nemo_asr/manifests/test_manifests.json'
with open("/homes/jinyu/attack_vc/data/test_pairs.txt", "r") as f:
    lines = f.readlines()
lines = [line.rstrip().split() for line in lines] 

with open(manifest_path, 'w') as fout:
    for src_path, tar_path, adv_path in tqdm(lines):
        # Lines look like this:
        # <s> transcript </s> (fileID)
        try:
            trans_path = tar_path.replace("wav16", "txt").replace("_mic1.wav", ".txt")
            with open(trans_path, "r") as f:
                line = f.readlines()[0]
            transcript = re.sub(r'[^\w\s]', '', line.strip().lower())
            

            duration = librosa.core.get_duration(filename=tar_path)
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": tar_path,
                "duration": duration,
                "text": transcript
            }
            json.dump(metadata, fout)
            fout.write('\n')
        except:
            pass