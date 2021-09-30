import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib

from scheduler import CosineAnnealingWarmUpRestarts

device = torch.device("cuda")
dtype = torch.float
epochs = 32
learning_rate = 0.1

# lr_list = [1] * epochs
# e = math.exp(1)
# for i,v in enumerate(lr_list) :
#     lr_list[i] = 0.05 * (math.cos(i*math.pi/(epochs*2)))* math.exp(1.*i*-e/epochs)
# #     lr_list[i] = 0.05 * math.exp(1.*i*-e/NUM_EPOCH)
# print(lr_list)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# training function
def fit(model, dataloader, optimizer):
#     print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
#     for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
    for i, data in enumerate(dataloader) :
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target, 1)[1])
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    loss = running_loss/len(dataloader.dataset)
    accuracy = 100. * running_correct/len(dataloader.dataset)
    
#     print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")
    
    return loss, accuracy


#validation function
def validate(model, dataloader):
#     print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
#         for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
        for i, data in enumerate(dataloader) :
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, torch.max(target, 1)[1])
            
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        
        loss = running_loss/len(dataloader.dataset)
        accuracy = 100. * running_correct/len(dataloader.dataset)
#         print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')
        
        return loss, accuracy
    
    
def test(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, target = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()
    return correct, total    

def adjust_learning_rate(optimizer, epoch, lr_rate=learning_rate):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_rate * (0.1 ** (epoch // 8))
#     epoch += 1
#     lr = lr_rate * (math.cos(i*math.pi/(epoch*2)))* math.exp(1.*i*-e/epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_gated(model, trainloader, valloader, decay=1e-4) :
    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    start = time.time()
    optimizer = optim.SGD(model.parameters(), lr=0.0001)  
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=8, T_mult=1, eta_max=0.1,  T_up=3, gamma=0.5)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
#         adjust_learning_rate(optimizer, epoch)
#         learning_rate = lr_list[epoch]
#         optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        train_epoch_loss, train_epoch_accuracy = fit(model, trainloader, optimizer)
        val_epoch_loss, val_epoch_accuracy = validate(model, valloader)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        scheduler.step()
    end = time.time()
    print((end-start)/60, 'minutes')
    torch.save(model.state_dict(), f"../trained_models/resnet18_epochs{epochs}.pth")
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../outputs/plots/accuracy.png')
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/plots/loss.png')    
    
def train(model, trainloader, valloader, decay=1e-4, opt='adam') :
    train_loss , train_accuracy = [], []
    val_loss , val_accuracy = [], []
    start = time.time()
    if opt == 'sgd' :
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    elif opt == 'adam' :
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else :
        print('error, use adam')
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(optimizer)
    scheduler = None
#     if not opt == 'adam' :
#         scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=8, T_mult=1, eta_max=0.1,  T_up=2, gamma=0.5)
#     
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        adjust_learning_rate(optimizer, epoch)
        EPOCH_lr = optimizer.param_groups[0]["lr"]
        train_epoch_loss, train_epoch_accuracy = fit(model, trainloader, optimizer)
        val_epoch_loss, val_epoch_accuracy = validate(model, valloader)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        print('train loss {:.4f} val loss {:.4f} train acc {:.2f} val acc {:.2f} lr {:.4f}'.format(
            train_epoch_loss, val_epoch_loss, train_epoch_accuracy, val_epoch_accuracy, EPOCH_lr))
        if not scheduler == None :
            scheduler.step()
    end = time.time()
    print((end-start)/60, 'minutes')
    torch.save(model.state_dict(), f"../trained_models/resnet18_epochs{epochs}.pth")
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_accuracy, color='green', label='train accuracy')
    plt.plot(val_accuracy, color='blue', label='validataion accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('../outputs/plots/accuracy.png')
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(val_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/plots/loss.png')      