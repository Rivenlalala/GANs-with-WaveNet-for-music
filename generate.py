import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio
from model import WaveNet
from random import randint
from data import Piano

from torch.utils.data import DataLoader

piano = Piano()


model = WaveNet().cuda()
model.load_state_dict(torch.load('wavenet.pth'))
seedloader = DataLoader(piano, batch_size=1, shuffle=False)

recp_field = 5116
sample_len = 4000 * 3


with torch.no_grad():
    for sample, _, audio in seedloader:
        sample = Variable(sample[:, :, :recp_field].type(torch.FloatTensor)).cuda()
        for index in range(sample_len):
            print(sample[:, :, -1].argmax())
            output = model(sample)
            output = output[:, :, -1]
            print(output.argmax())
            new = output.new_zeros(output.size()).view(1, -1, 1)
            new[:, output.argmax().item(), :] = 1
            sample = torch.cat((sample, new), dim=2)

            if index % 100 == 99:
                print(index + 1)

        sample = sample.view(256, -1)
        print(sample.size())
        sample = sample.argmax(0)
        print(sample.size())
        print(sample)
        sample = torchaudio.transforms.MuLawDecoding(256)(sample)
        break
torchaudio.save('1.wav', sample.cpu(), 4000, 16)

