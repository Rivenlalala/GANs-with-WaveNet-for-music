import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset,DataLoader
from model import WaveNet
from data import Piano

piano = Piano(data_dir='./Dataset_4s/', length=4)
training_data = DataLoader(piano, batch_size=2, shuffle=True)
model = WaveNet().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, 500, gamma=0.5)
recp_field = 1276


checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['state_dict'])
epoch_old = checkpoint['epoch']
optimizer.load_state_dict(checkpoint['optimizer'])
model.train()

for epoch in range(epoch_old, 10000):
    running_loss = 0.0
    for index, (data, target, _) in enumerate(training_data):
        data = Variable(data.type(torch.FloatTensor)).cuda()
        logits = model(data)
        logits = logits[:, :, :-1]
        y = target[:, :, recp_field:].squeeze(1).cuda()

        loss = F.cross_entropy(logits, y).cuda()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    # print("[%d %.3f]" % (epoch + 1, running_loss / (index+1)))

    if epoch % 100 == 99:
        print("[%d %.3f]" % (epoch + 1, running_loss / (index + 1)))
        torch.save({'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, 'checkpoint.pth')
print('finished')
