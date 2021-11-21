import os
import os.path as osp
import imageio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import glob
from tqdm import tqdm

import numpy as np


aug = ['org', 'flipflop', 'pca', 'crop', 'edge', 'rot', 'jitter']
# root = './subset_verflip'

# folders = folders.sort()
# print(folders)
# print(folders.sort())
# for root in aug:
# classes = ['elephant', 'dalmatian', 'butterfly', 'garfield', 'bonsai', 'Leopards', 'binocular', 'lotus', 'waterlilly', 'sunflower']						

# root = 'rot30'
for root in aug:
    data = []
    labels = []
    folders = os.listdir(root)
    for folder in tqdm(folders):
        # print(folder)
        tmp = osp.join(root, folder)
        for fimg in glob.glob(os.path.join(tmp, '*.jpg')):
            
            img = imageio.imread(fimg, pilmode="RGB")
            data.append(img)
            labels.append(folder)

    data = np.array(data)
    print(data[1].shape)
    labels = np.array(labels)
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    print(len(labels))

    np.save('./{}.npy'.format(root), data, allow_pickle=True)
    np.save('./{}_label.npy'.format(root), labels, allow_pickle=True)

# folders = os.listdir(root)
# for folder in (folders):
#     if folder != 'train':
#         data = []
#         labels = []
#         # print(folder)
#         tmp = osp.join(root, folder)
#         for fimg in glob.glob(os.path.join(tmp, '*.jpg')):
#             # print(fimg.split('/')[-1])
#             img = imageio.imread(fimg, pilmode="RGB")
#             data.append(img)
#             for i in range(len(classes)):
#                 if fimg.split('/')[-1].find(classes[i]) >= 0:
#                     labels.append(i)
#                 # else:
#                     # print(fimg)

#         data = np.array(data)
#         labels = np.array(labels)
#         lb = LabelBinarizer()
#         labels = lb.fit_transform(labels)
#         print(len(labels))

#         np.save('./{}/{}_same.npy'.format(root, folder), data, allow_pickle=True)
#         np.save('./{}/{}_same_label.npy'.format(root,folder), labels, allow_pickle=True)