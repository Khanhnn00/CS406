import torch
import numpy as np
import os
from models.network import PaperNet
from dataloader import ImageDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.set_num_threads(8)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# print(device)

augments = ['org', 'crop', 'jitter', 'pca', 'flipflop', 'rot', 'edge']

x_test = np.load('./fixed_x_test.npy', allow_pickle=True)
y_test = np.load('./fixed_y_test.npy', allow_pickle=True)
test_data = ImageDataset(x_test, y_test)
test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=True, num_workers=0)
model = PaperNet()
model = model.to(device)

def test(model, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data[0].to(device), data[1].to(device)
            # print(target.shape)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == torch.max(target, 1)[1]).sum().item()
    model.train()
    return correct, total, predicted, target

for augment in augments:
    print('-'*50)
    print('Testing {} augmentation'.format(augment))
    checkpoint = torch.load('./best_ckp_paper_{}.pth'.format(augment))
    model.load_state_dict(checkpoint)

    correct, total, predicted, target = test(model, test_loader)
    print('Accuracy : %0.3f %%' % (100 * correct / total))
    print('-'*50)