import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset,DataLoader
from model import WaveNet
from data import Piano
import numpy as np


piano = Piano()
training_data = DataLoader(piano, batch_size=1, shuffle=True)
model = WaveNet().cuda()
train_step = optim.Adam(model.parameters(), lr=2e-3, eps=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(train_step, milestones=[50,150,250], gamma=0.5)

for epoch in range(100):
    running_loss = 0.0
    for index, (data, target, _) in enumerate(training_data):
        data = Variable(data.type(torch.FloatTensor)).cuda()
        logits = model(data)
        y = target[:, :, 5116:].view(1, -1).cuda()

        loss = F.cross_entropy(logits, y).cuda()
        train_step.zero_grad()
        loss.backward()
        train_step.step()
        running_loss += loss.item()
        if index % 10 == 9:
            scheduler.step()
            print('[%d, %5d] loss: %.5f' % (epoch + 1, index + 1, running_loss / 10))
            running_loss = 0.0

print('finished')

PATH = './wavenet.pth'
torch.save(model.state_dict(), PATH)
