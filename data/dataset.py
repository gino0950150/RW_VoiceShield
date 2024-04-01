import os

import h5py
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
from tqdm import tqdm

from data.utils import load
import librosa
import random
import torch

class SeparationDataset(Dataset):
    def __init__(self, dataset, partition, instruments, sr, channels, shapes, random_hops, hdf_dir, audio_transform=None, in_memory=False):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(SeparationDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_dir = os.path.join(hdf_dir, partition + ".hdf5")

        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments

        # PREPARE HDF FILE

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_dir):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_dir, "w") as f:
                f.attrs["sr"] = sr
                f.attrs["channels"] = channels
                f.attrs["instruments"] = instruments

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load mix
                    mix_audio, _ = load(example["mix"], sr=self.sr, mono=(self.channels == 1))

                    source_audios = []
                    for source in instruments:
                        # In this case, read in audio and convert to target sampling rate
                        source_audio, _ = load(example[source], sr=self.sr, mono=(self.channels == 1))
                        source_audios.append(source_audio)
                    source_audios = np.concatenate(source_audios, axis=0)
                    assert(source_audios.shape[1] == mix_audio.shape[1])

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=mix_audio.shape, dtype=mix_audio.dtype, data=mix_audio)
                    grp.create_dataset("targets", shape=source_audios.shape, dtype=source_audios.dtype, data=source_audios)
                    grp.attrs["length"] = mix_audio.shape[1]
                    grp.attrs["target_length"] = source_audios.shape[1]

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_dir, "r") as f:
            if f.attrs["sr"] != sr or \
                    f.attrs["channels"] != channels or \
                    list(f.attrs["instruments"]) != instruments:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate and channel or instruments are not as expected. Did you load an out-dated HDF file?")

        # HDF FILE READY

        # SET SAMPLING POSITIONS

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_dir, "r") as f:
            lengths = [f[str(song_idx)].attrs["target_length"] for song_idx in range(len(f))]

            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [(l // self.shapes["output_frames"]) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):
        # Open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_dir, 'r', driver=driver)

        # Find out which slice of targets we want to read
        audio_idx = self.start_pos.bisect_right(index)
        if audio_idx > 0:
            index = index - self.start_pos[audio_idx - 1]

        # Check length of audio signal
        audio_length = self.hdf_dataset[str(audio_idx)].attrs["length"]
        target_length = self.hdf_dataset[str(audio_idx)].attrs["target_length"]

        # Determine position where to start targets
        if self.random_hops:
            start_target_pos = np.random.randint(0, max(target_length - self.shapes["output_frames"] + 1, 1))
        else:
            # Map item index to sample position within song
            start_target_pos = index * self.shapes["output_frames"]

        # READ INPUTS
        # Check front padding
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio_length:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio_length
            end_pos = audio_length
        else:
            pad_back = 0

        # Read and return
        audio = self.hdf_dataset[str(audio_idx)]["inputs"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = self.hdf_dataset[str(audio_idx)]["targets"][:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            targets = np.pad(targets, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

        targets = {inst : targets[idx*self.channels:(idx+1)*self.channels] for idx, inst in enumerate(self.instruments)}

        if hasattr(self, "audio_transform") and self.audio_transform is not None:
            audio, targets = self.audio_transform(audio, targets)

        return audio, targets

    def __len__(self):
        return self.length
    
    
    
class VCTKDataset(Dataset):
    def __init__(self, txt_dir, instruments, sr, channels, shapes, random_hops, smodel, audio_transform=None, in_memory=False):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(VCTKDataset, self).__init__()

        with open(txt_dir, "r") as f:
            lines = f.readlines()
        
        self.audiolist = [line.rstrip() for line in lines]
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments
        self.smodel = smodel


    def __getitem__(self, index):

        audio, _= librosa.load(self.audiolist[index], sr = self.sr)
        src_spk = self.audiolist[index].split("/")[-2]
        j = True
        while j:
            tar_pth = random.choice(self.audiolist)
            tar_spk = tar_pth.split("/")[-2]
            if tar_spk != src_spk:
                j = False
        tar_wav, _ = librosa.load(tar_pth, sr = self.sr)
        
        src_emb = self.smodel.embed_utterance(audio)
        src_emb = torch.from_numpy(src_emb).cuda()
        tar_emb = self.smodel.embed_utterance(tar_wav)
        tar_emb = torch.from_numpy(tar_emb).cuda()
        
        
        start_target_pos = np.random.randint(0, max(audio.shape[0] - self.shapes["output_frames"] + 1, 1))
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        m = np.abs(audio).max()
        audio =  audio / m * 0.9
        audio = np.expand_dims(audio, 0)

        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio.shape[1]:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio.shape[1]
            end_pos = audio.shape[1]
        else:
            pad_back = 0

        # Read and return
        audio = audio[:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

    
        
        
        return audio, src_emb, tar_emb, self.audiolist[index], tar_spk

    def __len__(self):
        return len(self.audiolist)

class VCTKFeedBackDataset(Dataset):
    def __init__(self, txt_dir, instruments, sr, channels, shapes, random_hops, smodel, audio_transform=None, in_memory=False):
        '''
        Initialises a source separation dataset
        :param data: HDF audio data object
        :param input_size: Number of input samples for each example
        :param context_front: Number of extra context samples to prepend to input
        :param context_back: NUmber of extra context samples to append to input
        :param hop_size: Skip hop_size - 1 sample positions in the audio for each example (subsampling the audio)
        :param random_hops: If False, sample examples evenly from whole audio signal according to hop_size parameter. If True, randomly sample a position from the audio
        '''

        super(VCTKFeedBackDataset, self).__init__()

        with open(txt_dir, "r") as f:
            lines = f.readlines()
        
        self.audiolist = [line.rstrip() for line in lines]
        self.random_hops = random_hops
        self.sr = sr
        self.channels = channels
        self.shapes = shapes
        self.audio_transform = audio_transform
        self.in_memory = in_memory
        self.instruments = instruments
        self.smodel = smodel
        # self.cmodel = cmodel
        # self.vc_utils = vc_untils


    def __getitem__(self, index):

        audio, _= librosa.load(self.audiolist[index], sr = self.sr)
        src_spk = self.audiolist[index].split("/")[-2]
        j = True
        while j:
            tar_pth, c_pth = random.sample(self.audiolist, 2)
            tar_spk = tar_pth.split("/")[-2]
            c_spk = c_pth.split("/")[-2]
            if tar_spk != src_spk and c_spk != src_spk and c_spk != tar_spk:
                j = False
        tar_wav, _ = librosa.load(tar_pth, sr = self.sr)
        c_wav, _ = librosa.load(c_pth, sr = self.sr)
        
        src_emb = self.smodel.embed_utterance(audio)
        src_emb = torch.from_numpy(src_emb).cuda()
        tar_emb = self.smodel.embed_utterance(tar_wav)
        tar_emb = torch.from_numpy(tar_emb).cuda()
        
        audio = self.crop_audio(audio)
        c_wav = self.crop_audio(c_wav)
        
        # start_target_pos = np.random.randint(0, max(audio.shape[0] - self.shapes["output_frames"] + 1, 1))
        # start_pos = start_target_pos - self.shapes["output_start_frame"]
        # m = np.abs(audio).max()
        # audio =  audio / m * 0.9
        # audio = np.expand_dims(audio, 0)

        # if start_pos < 0:
        #     # Pad manually since audio signal was too short
        #     pad_front = abs(start_pos)
        #     start_pos = 0
        # else:
        #     pad_front = 0

        # # Check back padding
        # end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        # if end_pos > audio.shape[1]:
        #     # Pad manually since audio signal was too short
        #     pad_back = end_pos - audio.shape[1]
        #     end_pos = audio.shape[1]
        # else:
        #     pad_back = 0

        # # Read and return
        # audio = audio[:, start_pos:end_pos].astype(np.float32)
        # if pad_front > 0 or pad_back > 0:
        #     audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

    
        
        
        return audio, src_emb, tar_emb, c_wav, self.audiolist[index], tar_spk

    def __len__(self):
        return len(self.audiolist)
    
    def crop_audio(self, audio):
        start_target_pos = np.random.randint(0, max(audio.shape[0] - self.shapes["output_frames"] + 1, 1))
        start_pos = start_target_pos - self.shapes["output_start_frame"]
        m = np.abs(audio).max()
        audio =  audio / m * 0.9
        audio = np.expand_dims(audio, 0)

        if start_pos < 0:
            # Pad manually since audio signal was too short
            pad_front = abs(start_pos)
            start_pos = 0
        else:
            pad_front = 0

        # Check back padding
        end_pos = start_target_pos - self.shapes["output_start_frame"] + self.shapes["input_frames"]
        if end_pos > audio.shape[1]:
            # Pad manually since audio signal was too short
            pad_back = end_pos - audio.shape[1]
            end_pos = audio.shape[1]
        else:
            pad_back = 0

        # Read and return
        audio = audio[:, start_pos:end_pos].astype(np.float32)
        if pad_front > 0 or pad_back > 0:
            audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)
        return audio