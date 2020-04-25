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


seedloader = DataLoader(piano, batch_size=1, shuffle=True)




recp_field = 1276
sample_len = 4000 * 8


model = WaveNet().cuda()
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])

song = 6
with torch.no_grad():
    for seed, _, audio in seedloader:
        seed = seed[:, :, 500:recp_field + 500].float().cuda()
        #sample = Variable(sample.type(torch.FloatTensor)).cuda()
        output = seed
        for index in range(sample_len):
            #print(sample[:, :, -10:].argmax(1))
            new = model(seed)
            p = torch.distributions.categorical.Categorical(logits=new.squeeze())
            new_mag = p.sample()
            new = new.zero_()
            new[:, new_mag] = 1
            output = torch.cat((output, new.view(1, 256, 1)), dim=2)
            seed = output[:, :, -recp_field:]
            if index % 100 == 99:
                print(index + 1)
        song += 1
        sample = output.squeeze()
        sample = sample.argmax(0)
        sample = torchaudio.transforms.MuLawDecoding(256)(sample)

        torchaudio.save('{}.wav'.format(song), sample.cpu(), 4000, 16)
        if song > 15:
            break