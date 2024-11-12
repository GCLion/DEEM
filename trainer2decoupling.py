import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from AE_decoupling_DEEM import Autoencoder_decoupling
from basenet import Model
from data_utils import genMydata, Dataset_ae, genSpoof_list, genMydata2, Dataset_ae_lfcc, genMydata3, \
    Dataset_ae_finetune, genMydata4For, Dataset_ae4For

matplotlib.use('AGG')
import matplotlib.pyplot as plt

root_dir = ''

np.random.seed(123)
torch.manual_seed(123)
BATCH_SIZE = 4
LR = 0.0001
EPOCHS = 150

latent_length = 64

input_size = 17280
listx, listxs, listxb = genMydata("label_all.txt")
listy = genMydata4For(2580, 7420)

full_dataset = Dataset_ae4For(listx, listxs, listxb, listy, root_dir)

train_size = int(0.7 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

hidden1 = 2048
hidden2 = 1024
hidden3 = 512
hidden4 = 128
train_loader = DataLoader(train_dataset, BATCH_SIZE, True, drop_last=False)

verify_loader = DataLoader(test_dataset, BATCH_SIZE * 2, False, drop_last=False)

print(train_dataset.__len__())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open("./conf.yaml", 'r') as f_yaml:
    parser1 = yaml.safe_load(f_yaml)
model = Model(parser1['model'])
model.load_state_dict(torch.load("./model/model.pth",map_location=device))
net = Autoencoder_decoupling(model,model.encoder.state_dict(),parser1['model'])

del model
if torch.cuda.is_available():
    net = net.cuda()

loss_f = nn.MSELoss()
# loss_f = MTLLoss(3)
if torch.cuda.is_available():
    loss_f = loss_f.cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
for name, parameter in net.Raw2feat.named_parameters():
    parameter.requires_grad = False

for name, parameter in net.encoder1[0].named_parameters():
    parameter.requires_grad = False
for name, parameter in net.encoder1[1].named_parameters():
    parameter.requires_grad = False

for name, parameter in net.encoder2[0].named_parameters():
    parameter.requires_grad = False
for name, parameter in net.encoder2[1].named_parameters():
    parameter.requires_grad = False


scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
optimizer.add_param_group({"params": loss_f.parameters(), 'lr': LR})
Loss_list_epoch = []
Loss_list_epoch_a = []
Loss_list_epoch_b = []
Loss_list_epoch_a1 = []
Loss_list_epoch_b1 = []
Loss_list_epoch_a2 = []
Loss_list_epoch_b2 = []
Loss_list_epoch_TRAIN = []
Loss_list_epoch_a_TRAIN = []
Loss_list_epoch_b_TRAIN = []
Loss_list_epoch_a_TRAIN1 = []
Loss_list_epoch_b_TRAIN1 = []
Loss_list_epoch_a_TRAIN2 = []
Loss_list_epoch_b_TRAIN2 = []
for epoch in range(EPOCHS):
    Loss_sum = 0
    Loss_suma = 0
    Loss_sumb = 0
    Loss_suma1 = 0
    Loss_sumb1 = 0
    Loss_suma2 = 0
    Loss_sumb2 = 0
    net.train()
    for step, (x1, x2, x12, x21, y1, y2) in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            x1 = x1.cuda()
            x2 = x2.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()
            x12 = x12.cuda()
            x21 = x21.cuda()
        x1, x2, y1, y2, recon_x1, recon_x2, recon_x1_, recon_x2_, recon_y1, recon_y2, recon_y1__, recon_y2__ = net(x1,
                                                                                                                   x2,
                                                                                                                   y1,
                                                                                                                   y2)
        x12 = net.Raw2feat(x12)
        x21 = net.Raw2feat(x21)
        loss_a1 = loss_f(recon_x1, x1) + loss_f(recon_x2, x2)
        loss_a2 = loss_f(recon_x1_, x12) + loss_f(recon_x2_, x21)
        loss_a = loss_a2 + loss_a1
        loss_b1 = loss_f(recon_y1, y1) + loss_f(recon_y2, y2)
        loss_b2 = loss_f(recon_y1__, y1) + loss_f(recon_y2__, y2)
        loss_b = loss_b2 + loss_b1
        loss = loss_a + loss_b
        # loss = loss_f([xsr,xbr,yr],[xs,xb,y])
        loss.backward()
        optimizer.step()
        Loss_sum += loss.item()
        Loss_suma += loss_a.item()
        Loss_sumb += loss_b.item()
        Loss_suma1 += loss_a1.item()
        Loss_sumb1 += loss_b1.item()
        Loss_suma2 += loss_a2.item()
        Loss_sumb2 += loss_b2.item()
        optimizer.zero_grad()
    scheduler.step(Loss_sum)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    print("----------------epoch:{}  loss:{}----------------".format(epoch, Loss_sum / train_dataset.__len__()))
    Loss_list_epoch.append(Loss_sum / train_dataset.__len__())
    Loss_list_epoch_a.append(Loss_suma / train_dataset.__len__())
    Loss_list_epoch_b.append(Loss_sumb / train_dataset.__len__())
    Loss_list_epoch_a1.append(Loss_suma1 / train_dataset.__len__())
    Loss_list_epoch_b1.append(Loss_sumb1 / train_dataset.__len__())
    Loss_list_epoch_a2.append(Loss_suma2 / train_dataset.__len__())
    Loss_list_epoch_b2.append(Loss_sumb2 / train_dataset.__len__())
    loss_sum_v = 0
    loss_sum_a = 0
    loss_sum_b = 0
    loss_sum_a1 = 0
    loss_sum_b1 = 0
    loss_sum_a2 = 0
    loss_sum_b2 = 0
    with torch.no_grad():
        for step, (x1, x2, x12, x21, y1, y2) in enumerate(verify_loader, 1):
            if torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
                y1 = y1.cuda()
                y2 = y2.cuda()
                x12 = x12.cuda()
                x21 = x21.cuda()
            x1, x2, y1, y2, recon_x1, recon_x2, recon_x1_, recon_x2_, recon_y1, recon_y2, recon_y1__, recon_y2__ = net(
                x1, x2, y1, y2)
            x12 = net.Raw2feat(x12)
            x21 = net.Raw2feat(x21)
            loss_a1 = loss_f(recon_x1, x1) + loss_f(recon_x2, x2)
            loss_a2 = loss_f(recon_x1_, x12) + loss_f(recon_x2_, x21)
            loss_b1 = loss_f(recon_y1, y1) + loss_f(recon_y2, y2)
            loss_b2 = loss_f(recon_y1__, y1) + loss_f(recon_y2__, y2)
            loss_a = loss_a1 + loss_a2
            loss_b = loss_b2 + loss_b1
            loss = loss_a + loss_b
            loss_sum_v += loss.item()
            loss_sum_a += loss_a.item()
            loss_sum_b += loss_b.item()
            loss_sum_a1 += loss_a1.item()
            loss_sum_b1 += loss_b1.item()
            loss_sum_a2 += loss_a2.item()
            loss_sum_b2 += loss_b2.item()
    Loss_list_epoch_TRAIN.append(loss_sum_v / test_dataset.__len__())
    Loss_list_epoch_a_TRAIN.append(loss_sum_a / test_dataset.__len__())
    Loss_list_epoch_b_TRAIN.append(loss_sum_b / test_dataset.__len__())
    Loss_list_epoch_a_TRAIN1.append(loss_sum_a1 / test_dataset.__len__())
    Loss_list_epoch_b_TRAIN1.append(loss_sum_b1 / test_dataset.__len__())
    Loss_list_epoch_a_TRAIN2.append(loss_sum_a2 / test_dataset.__len__())
    Loss_list_epoch_b_TRAIN2.append(loss_sum_b2 / test_dataset.__len__())

x = range(1, len(Loss_list_epoch) + 1)
y = Loss_list_epoch
y2 = Loss_list_epoch_TRAIN
y3 = Loss_list_epoch_a_TRAIN
y4 = Loss_list_epoch_b_TRAIN
y5 = Loss_list_epoch_a
y6 = Loss_list_epoch_b
plt.plot(x, y, '.-', color='r', label="train")
plt.plot(x, y5, '.-', color='k', label="train")
plt.plot(x, y6, '.-', color='c', label="train")
plt.plot(x, y2, '.-', color='g', label="verify")
plt.plot(x, y3, '.-', color='b', label="verify")
plt.plot(x, y4, '.-', color='y', label="verify")
plt_title = 'EPOCHS = {}; BATCH_SIZE = {}; LEARNING_RATE:{}'.format(EPOCHS, BATCH_SIZE, LR)
plt.title(plt_title)
plt.xlabel('per epoch')
plt.ylabel('loss')
plt.savefig("./log/loss_all_ae.jpg")
plt.close()

y = Loss_list_epoch_a1
y2 = Loss_list_epoch_a2
y3 = Loss_list_epoch_b1
y4 = Loss_list_epoch_b2
plt.plot(x, y, '.-', color='r')
plt.plot(x, y2, '.-', color='g')
plt.plot(x, y3, '.-', color='b')
plt.plot(x, y4, '.-', color='y')
plt_title = 'EPOCHS = {}; BATCH_SIZE = {}; LEARNING_RATE:{}'.format(EPOCHS, BATCH_SIZE, LR)
plt.title(plt_title)
plt.xlabel('per epoch')
plt.ylabel('loss')
plt.savefig("./log/loss_all_ae_01.jpg")
plt.close()

y = Loss_list_epoch_a_TRAIN1
y2 = Loss_list_epoch_a_TRAIN2
y3 = Loss_list_epoch_b_TRAIN1
y4 = Loss_list_epoch_b_TRAIN2
plt.plot(x, y, '.-', color='r')
plt.plot(x, y2, '.-', color='g')
plt.plot(x, y3, '.-', color='b')
plt.plot(x, y4, '.-', color='y')
plt_title = 'EPOCHS = {}; BATCH_SIZE = {}; LEARNING_RATE:{}'.format(EPOCHS, BATCH_SIZE, LR)
plt.title(plt_title)
plt.xlabel('per epoch')
plt.ylabel('loss')
plt.savefig("./log/loss_all_ae_02.jpg")
plt.close()

print("task over,saving model......")
torch.save(net, "./model/ae_FoR.pth".format(EPOCHS, BATCH_SIZE))
