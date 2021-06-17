import os
import numpy as np
import torch
import torchvision
import argparse
from utils import nt_xent_loss,loss_function
from tqdm import tqdm

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer,batch_size,epoch,epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for x_1, x_2, target in train_bar:
        x_1, x_2 = x_1.cuda(non_blocking=True), x_2.cuda(non_blocking=True)
        h_1, z_1 = net(x_1)
        h_2, z_2 = net(x_2)
        loss=nt_xent_loss(z_1,z_2,0.5)
        #loss=loss_function(z_1,z_2,0.5)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


