import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
from constants import *

class US8K(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #get audio sample path
        audio_sample_path = self.get_audio_sample_path(index)
        #get audio sample label
        label = self.annotations.iloc[index, 6]
        #load signal and sample rate
        signal, sr = torchaudio.load(audio_sample_path)
        #move signal to device
        signal = signal.to("cpu")
        
        #resample to target sample rate, unless
        signal = self.resample(signal, sr)
        #mix signal down to mono
        signal = self.mix_down(signal)
        #truncate length to fit num_samples
        signal = self.cut(signal)
        #pad signal length to fit num_samples
        signal = self.right_pad(signal)
        
        #convert to Mel Spectrogram
        signal = self.transformation(signal)
        
        return signal.to(self.device), label

    def cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[
            index, 0])
        return path