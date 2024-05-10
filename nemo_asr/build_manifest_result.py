"""https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/ASR_with_NeMo.ipynb"""
import os 
from tqdm import tqdm
import json
import librosa
import re
import os

result_path = "/mnt/sda/jinyu/attack_vc/exps/0217_blackbox512/result_conf"
manifest_path = '/homes/jinyu/RW_VoiceShield/nemo_asr/manifests/test_manifests_blackbox512.json'



with open(manifest_path, 'w') as fout:
    for root, dirs, files in os.walk(result_path, topdown=False):
       for name in dirs:
            try:
                tar_path =  os.path.join(os.path.join(root, name), "adv_input.wav")
                trans_path =  os.path.join(os.path.join(root, name), "tarpath.txt")
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