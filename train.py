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
optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=10e-8)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200], gamma=0.5)

for epoch in range(2000):
    running_loss = 0.0
    for index, (data, target, _) in enumerate(training_data):
        data = Variable(data.type(torch.FloatTensor)).cuda()
        logits = model(data)
        y = target[:, :, 5116:].view(1, -1).cuda()

        loss = F.cross_entropy(logits, y).cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if index % 10 == 9:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, index + 1, running_loss / 10))
            running_loss = 0.0

    scheduler.step()

    if epoch % 10 == 9:
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, 'checkpoint.pth')
        torch.save(model.state_dict(), 'wavenet.pth')
        print("checkpoint saved")


print('finished')

