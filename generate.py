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


seedloader = DataLoader(piano, batch_size=1, shuffle=False)




recp_field = 1276
sample_len = 4000 * 8


model = WaveNet().cuda()
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])

with torch.no_grad():
    for seed, _, audio in seedloader:
        seed = Variable(seed[:, :, 5000:recp_field + 5000].type(torch.FloatTensor)).cuda()
        #sample = Variable(sample.type(torch.FloatTensor)).cuda()
        output = seed
        for index in range(sample_len):
            #print(sample[:, :, -10:].argmax(1))
            new = model(seed)
            new_mag = new.argmax().item()
            new = new.new_zeros(new.size())
            new[:, new_mag] = 1
            output = torch.cat((output, new.view(1, 256, 1)), dim=2)
            seed = output[:, :, -recp_field:]
            if index % 100 == 99:
                print(index + 1)
        break

sample = output.squeeze()
sample = sample.argmax(0)
sample = torchaudio.transforms.MuLawDecoding(256)(sample)

torchaudio.save('1.wav', sample.cpu(), 4000, 16)



'''data = Variable(data.type(torch.FloatTensor)).cuda()
output = model(data.view(1, 256, -1))
output = output.view(256, -1)
output = output.argmax(0)
output = torchaudio.transforms.MuLawDecoding(256)(output)'''
'''filename = '132.mp3'
waveform, sample_rate_old = torchaudio.load(filename)
waveform_downsample = torchaudio.transforms.Resample(sample_rate_old, 4000)(waveform)
waveform_mulaw = torchaudio.transforms.MuLawEncoding(256)(waveform_downsample)
one_hot = one_hot_encode(waveform_mulaw, channels=256)
data = one_hot.transpose()
sample = torch.from_numpy(data)
sample = Variable(sample.type(torch.FloatTensor)).cuda().view(1, 256, -1)
sample = sample[:, :, :recp_field + 1]'''





