import matplotlib.pyplot as plt
import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import random
import os
# import pretrainedmodels
# from imutils import paths
from sklearn.model_selection import train_test_split
import torchvision.models as models

from tqdm import tqdm

from network import ResNet
from dataloader import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

batch_size = 64
epochs = 30

def fit(model, train_data, train_loader, criterion, optimizer):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(train_loader), total=int(len(train_data)/train_loader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, torch.max(target, 1)[1])
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        loss.backward()
        optimizer.step()
        
    loss = running_loss/len(train_loader.dataset)
    accuracy = 100. * running_correct/len(train_loader.dataset)
    
    print(f"Train Loss: {loss:.4f}, Train Acc: {accuracy:.2f}")
    
    return loss, accuracy

def validate(model, val_data, val_loader, criterion):
    print('Validating')
    model.eval()
    running_loss = 0.0
    running_correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(val_loader), total=int(len(val_data)/val_loader.batch_size)):
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)
            loss = criterion(outputs, torch.max(target, 1)[1])
            
            running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            running_correct += (preds == torch.max(target, 1)[1]).sum().item()
        
        loss = running_loss/len(val_loader.dataset)
        accuracy = 100. * running_correct/len(val_loader.dataset)
        print(f'Val Loss: {loss:.4f}, Val Acc: {accuracy:.2f}')
        
        return loss, accuracy

def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()
    return correct, total


def main():
    augments = ['crop', 'jitter', 'pca', 'flipflop', 'rot', 'edge']
    raw_x = np.load('../dataset/new_data/org.npy', allow_pickle=True)
    raw_y = np.load('../dataset/new_data/org_label.npy', allow_pickle=True)

    X, x_val , Y, y_val = train_test_split(raw_x, raw_y, 
                                                            test_size=0.2,
                                                            random_state=42)
    raw_x, x_test, raw_y, y_test = train_test_split(X, Y, 
                                                            test_size=0.2, 
                                                            random_state=42)
                                    
    np.save('./fixed_x_val.npy', x_val)
    np.save('./fixed_y_val.npy', y_val)     
    np.save('./fixed_x_test.npy', x_test)     
    np.save('./fixed_y_test.npy', y_test)
                        
    val_data = ImageDataset(x_val, y_val)
    test_data = ImageDataset(x_test, y_test)
    val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    for aug in augments:
        if aug != 'org':
            x_train = np.load('../dataset/new_data/{}.npy'.format(aug), allow_pickle=True)
            y_train = np.load('../dataset/new_data/{}_label.npy'.format(aug), allow_pickle=True)

            x_train = np.concatenate((x_train, raw_x), axis=0)
            y_train = np.concatenate((y_train, raw_y), axis=0)
        else:
            x_train = raw_x
            y_train = raw_y

        train_data = ImageDataset(x_train, y_train)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=0)

        model = models.resnet34(pretrained=True, progress=True)
        model.fc = nn.Linear(512, 101, bias=True)
        model.fc.bias.data.zero_()

        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        # loss function
        criterion = nn.CrossEntropyLoss()
        train_loss , train_accuracy = [], []
        val_loss , val_accuracy = [], []
        print(f"Training on {len(train_data)} examples, validating on {len(val_data)} examples...")
        start = time.time()
        val_acc = 0
        for epoch in tqdm(range(epochs)):
            print(f"Epoch {epoch+1} of {epochs}")
            train_epoch_loss, train_epoch_accuracy = fit(model, train_data, train_loader, criterion, optimizer)
            val_epoch_loss, val_epoch_accuracy = validate(model, val_data, val_loader, criterion)
            train_loss.append(train_epoch_loss)
            train_accuracy.append(train_epoch_accuracy)
            val_loss.append(val_epoch_loss)
            val_accuracy.append(val_epoch_accuracy)
            if val_epoch_accuracy > val_acc:
                val_acc = val_epoch_accuracy
                torch.save(model.state_dict(), f"./best_ckp_{aug}.pth")
        end = time.time()
        print((end-start)/60, 'minutes')
        torch.save(model.state_dict(), f"./last_ckp_{aug}.pth")
        # accuracy plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_accuracy, color='green', label='train accuracy')
        plt.plot(val_accuracy, color='blue', label='validataion accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('./accuracy_{}.png'.format(aug))
        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./loss_{}.png'.format(aug))

main()