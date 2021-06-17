import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model2 import Model
from train_test import train
import torch.nn as nn
from utils import CIFAR10Pair,train_transform,test_transform

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature= args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs
    status="true"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data prepare
    train_data = CIFAR10Pair(root='data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    test_data = CIFAR10Pair(root='data', train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.cuda()
    #learning_rate=0.3*(batch_size/256)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    #c = len(memory_data.classes)
    
    # training loop
    batch_loss=[]
    results = {'train_loss': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, batch_size, epochs,status)
    if not os.path.exists('results'):
        os.mkdir('results')
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer,batch_size,epoch,epochs)
        results['train_loss'].append(train_loss)
        batch_loss.append(train_loss)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
    ##plot loss vs epoch
    #plot_losses(batch_loss, 'Training Losses', 'results/training_losses.png')
        
