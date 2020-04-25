import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from model import WaveNet, GenLSTM, DisLSTM
from data import Piano
import torchaudio
import numpy as np

recp_field = 1276
sample_len = 4000 * 8

netW = WaveNet(dilation_depth=8).cuda()
for param in netW.parameters():
    param.requires_grad = False
netG = GenLSTM(stride=500).double().cuda()
netD = DisLSTM().double().cuda()
optimizerD = optim.Adam(netD.parameters(), lr=1e-3)
optimizerG = optim.Adam(netG.parameters(), lr=1e-3)

piano = Piano(data_dir='./Dataset_4s', length=4, sample_rate=4000)
dataloader = DataLoader(piano, batch_size=5, shuffle=True)
seedloader = DataLoader(piano, batch_size=1, shuffle=True)

checkpoint = torch.load('checkpoint.pth')
netW.load_state_dict(checkpoint['state_dict'])

criterion = nn.BCELoss()


def wavenetGen(batch_size=5, sample_len=4, recp_field=1276):
    for i, (seed, _, _) in enumerate(seedloader):
        if i >= batch_size:
            break
        seed = seed[:, :, 500:recp_field + 500].float().cuda()
        # sample = Variable(sample.type(torch.FloatTensor)).cuda()
        output = seed
        r_input = seed[:, :, -1]
        r_input = r_input.unsqueeze(0).double()
        for index in range(sample_len * 4000):
            new = netW(seed)
            p = torch.distributions.categorical.Categorical(logits=new.squeeze())
            new_mag = p.sample()
            new = new.zero_()
            new[:, new_mag] = 1
            if index % 500 == 499:
                r_new = netG(r_input)
                r_new = r_new[:, -1, :]
                r_new_mag = r_new.argmax(1)
                r_new = r_new.zero_()
                r_new[:, r_new_mag] = 1
                r_input = torch.cat((r_input.float(), new.permute(0, 2, 1).detach()), dim=1)
                output = torch.cat((output, r_new.view(1, 256, 1).float()), dim=2)
            else:
                output = torch.cat((output, new.view(1, 256, 1)), dim=2)
            seed = output[:, :, -recp_field:]
        if i == 0:
            sample = output
        else:
            sample = torch.cat((sample, output), dim=0)
        print(i)
    return sample


for epoch in range(100):
    for i, (data, _, _) in enumerate(dataloader):
        # update D
        # real batch
        print('updating D')
        print('real batch')
        netD.zero_grad()
        data = data[:, :, 0:-1:500].cuda()
        real = data.permute(0, 2, 1).double() # input: Tensor[batch, time_Step, in_depth]
        b_size = real.size(0)
        label = torch.full((b_size, 1), 1).double().cuda()
        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()

        # fake batch
        print('fake batch')
        print('generating')
        fake = wavenetGen()
        fake = fake.double()
        fake = netG(fake)
        label.fill_(0)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # update G
        print('updating G')
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/500][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
              % (epoch, i, len(dataloader), errD.item(), errG.item()))


