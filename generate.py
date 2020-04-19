import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio
from model import WaveNet
from random import randint
from data import Piano
import numpy as np

from torch.utils.data import DataLoader

piano = Piano()

def one_hot_encode(data, channels=256):
    data = data.numpy()
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data] = 1

    return one_hot


#seedloader = DataLoader(piano, batch_size=1, shuffle=False)




recp_field = 5116
sample_len = 4000

model = WaveNet().cuda()
model.load_state_dict(torch.load('wavenet.pth'))


#for sample, _, audio in seedloader:
    #sample = Variable(sample[:, :, :recp_field + 1].type(torch.FloatTensor)).cuda()
    #sample = Variable(sample.type(torch.FloatTensor)).cuda()

filename = '132.mp3'
waveform, sample_rate_old = torchaudio.load(filename)
waveform_downsample = torchaudio.transforms.Resample(sample_rate_old, 4000)(waveform)
waveform_mulaw = torchaudio.transforms.MuLawEncoding(256)(waveform_downsample)
one_hot = one_hot_encode(waveform_mulaw, channels=256)
data = one_hot.transpose()
sample = torch.from_numpy(data)
sample = Variable(sample.type(torch.FloatTensor)).cuda().view(1, 256, -1)
sample = sample[:, :, :recp_field + 1]

for index in range(sample_len):
    #print(sample[:, :, -10:].argmax(1))
    output = model(sample)[:, :, -1]
    new = output.new_zeros(output.size())
    new[:, output.argmax().item()] = 1
    sample = torch.cat((sample, new.view(1, 256, 1)), dim=2)

    if index % 100 == 99:
        print(index + 1)

sample = sample.view(256, -1)
print(sample.size())
sample = sample.argmax(0)
print(sample.size())
print(sample)
sample = torchaudio.transforms.MuLawDecoding(256)(sample)




'''data = Variable(data.type(torch.FloatTensor)).cuda()
output = model(data.view(1, 256, -1))
output = output.view(256, -1)
output = output.argmax(0)
output = torchaudio.transforms.MuLawDecoding(256)(output)'''



torchaudio.save('1.wav', sample.cpu(), 4000, 16)

