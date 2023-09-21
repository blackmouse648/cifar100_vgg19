from torch.utils.data import dataloader
import data
import model_vgg_hand
import model_vgg19_auto
import torch
import torch.nn as nn
import torch.optim as optim
import datetime


batches = 128

model_name = 'vgg19_bn.pth'


train_loader = dataloader.DataLoader(dataset=data.cifar100_train, batch_size=batches, shuffle=True, pin_memory=True)

val_loader = dataloader.DataLoader(dataset=data.cifar100_val, batch_size=batches, shuffle=False, pin_memory=True)


device = (torch.device('cuda')
          if torch.cuda.is_available()
          else
          torch.device('cpu'))

print(device)

learning_rate = 0.0001

model = model_vgg19_auto.vgg_19_bn().to(device=device)

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

loss_fn = nn.CrossEntropyLoss()

n_epochs = 50


if __name__ == '__main__':

    for epoch in range(1, n_epochs + 1):
        losses = 0.0
        total = 0
        correct = 0
        for imgs, labels in train_loader:

            img = imgs.to(device=device)
            label = labels.to(device=device)

            out = model(img)

            _, predicted = torch.max(out, dim=1)

            loss = loss_fn(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()
            correct += (predicted == label).sum().item()
            total += labels.shape[0]
        print("%10s Epoch: %d, Loss: %f, accuracy: %f" % (datetime.datetime.now(), epoch, losses/batches, correct/total))

    torch.save(model, model_name)