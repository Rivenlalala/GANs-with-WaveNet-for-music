import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio


def one_hot_encode(data, channels=256):
    data = data.numpy()
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    data = data.numpy()
    decoded = np.argmax(data, axis=axis)

    return decoded



class Piano(Dataset):

    def __init__(self, data_dir='./Dataset/', in_depth=256, sample_rate=4000):
        self.data_dir = data_dir
        self.in_depth = in_depth
        self.sample_rate = sample_rate

    def __len__(self):
        return len([file for file in os.listdir(self.data_dir)])

    def __getitem__(self, item):
        filename = os.path.join(self.data_dir, str(item) + '.mp3')
        waveform, sample_rate_old = torchaudio.load(filename)
        waveform_downsample = torchaudio.transforms.Resample(sample_rate_old, self.sample_rate)(waveform)
        waveform_mulaw = torchaudio.transforms.MuLawEncoding(self.in_depth)(waveform_downsample)
        one_hot = one_hot_encode(waveform_mulaw, channels=self.in_depth)
        data = one_hot.transpose()

        return data, waveform_mulaw, waveform_downsample



if __name__ == '__main__':
    piano = Piano()
    dataloader = DataLoader(piano, batch_size=1, shuffle=True, num_workers=4)
    for wave in dataloader:
        break


