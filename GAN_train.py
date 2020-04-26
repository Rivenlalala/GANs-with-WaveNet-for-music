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
# netG = GenLSTM(stride=500).double().cuda()
netG = nn.LSTMCell(256, 256).cuda()
netD = DisLSTM().double().cuda()
optimizerD = optim.Adam(netD.parameters(), lr=1e-3)
optimizerG = optim.Adam(netG.parameters(), lr=1e-3)

piano = Piano(data_dir='./Dataset_4s', length=4, sample_rate=4000)
dataloader = DataLoader(piano, batch_size=2, shuffle=True)
seedloader = DataLoader(piano, batch_size=1, shuffle=True)

checkpoint = torch.load('checkpoint.pth')
netW.load_state_dict(checkpoint['state_dict'])

criterion = nn.BCELoss()
#frame = inspect.currentframe()
#gpu_tracker = MemTracker(frame)

def wavenetGen(batch_size=2, sample_len=4, recp_field=1276):
    for i, (seed, _, _) in enumerate(seedloader):
        if i >= batch_size:
            break
        print('Generating sample', i)
        seed = seed[:, :, 500:recp_field + 500].float().cuda()
        # sample = Variable(sample.type(torch.FloatTensor)).cuda()
        output = seed
        # r_input = seed[:, :, -1]
        # r_input = r_input.unsqueeze(0).double()
        for index in range(sample_len * 4000):
            new = netW(seed.detach())
            p = torch.distributions.categorical.Categorical(logits=new.squeeze())
            new_mag = p.sample()
            new = new.zero_()
            new[:, new_mag] = 1
            if index % 50 == 49:
                r_input = new.squeeze(2)
                try:
                    h, c = netG(r_input, (h, c))
                except:
                    h, c = netG(r_input)
                    r_new = h
                    r_new_mag = r_new.argmax(1)
                    r_new = r_new.zero_()
                    r_new[:, r_new_mag] = 1
                    r_output = r_new.unsqueeze(0)
                else:
                    r_new = h
                    r_new_mag = r_new.argmax(1)
                    r_new = r_new.zero_()
                    r_new[:, r_new_mag] = 1
                # r_input = torch.cat((r_input, new.permute(0, 2, 1).detach().double()), dim=1)
                    r_output = torch.cat((r_output, r_new.unsqueeze(0)), dim=1)
                output = torch.cat((output, r_new.view(1, 256, 1).float()), dim=2)
            else:

                output = torch.cat((output, new.view(1, 256, 1)), dim=2)
            seed = output[:, :, -recp_field:]
        h = 0
        c = 0
        if i == 0:
            sample = output
            r_sample = r_output
        else:
            sample = torch.cat((sample, output), dim=0)
            r_sample = torch.cat((r_sample, r_output), dim=0)

    return r_sample, sample


for epoch in range(100):
    for i, (data, _, _) in enumerate(dataloader):
        # update D
        # real batch
        print('updating D')
        netD.zero_grad()
        data = data[:, :, 0:-1:50].cuda()
        real = data.permute(0, 2, 1).double()  # input: Tensor[batch, time_Step, in_depth]
        b_size = real.size(0)
        label = torch.full((b_size, 1), 1).double().cuda()
        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # fake batch
        fake, _ = wavenetGen()
        fake = fake.double()
        output = netD(fake.detach())
        b_size = fake.size(0)
        label = label.new_zeros((b_size, 1))
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_fake + errD_real
        optimizerD.step()

        # update G
        print('updating G')
        netG.zero_grad()
        label.fill_(1)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        print('[%d/100][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
            % (epoch, i, len(dataloader),
               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    if epoch % 10 == 9:
        torch.save({"epoch": epoch + 1,
                    "state_dict": netD.state_dict(),
                    "optimizer": optimizerD.state_dict()}, 'netD.pth')
        torch.save({'epoch': epoch + 1,
                    'state_dict': netG.state_dict(),
                    'optimizer': optimizerG.state_dict()}, 'netG.pth')
