import torch
from torch.utils.data import dataloader
import data


device = (torch.device('cuda')
          if torch.cuda.is_available()
          else
          torch.device('cpu'))


model = torch.load('vgg19_bn.pth')
val_loader = dataloader.DataLoader(dataset=data.cifar100_val, batch_size=128, shuffle=False, pin_memory=True)


total = 0
correct = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        img = imgs.to(device=device)
        label = labels.to(device=device)

        out = model(img)

        _, predicted = torch.max(out, dim=1)

        total += labels.shape[0]

        correct += (predicted == label).sum().item()

print('accuracy: %05f' % (correct / total))