import os
# import matplotlib.pyplot as plt
import imageio
from scipy import misc
import cv2
import numpy as np
import  random
import torch
import os.path as osp
import skimage.exposure as exposure
from PIL import Image

def verticalFlip(img):

    flip = cv2.flip(img, 0)
    return flip

def horizontalFlip(img):

    flip = cv2.flip(img, 1)
    return flip

def rotate(img, a=30):

    rot = misc.imrotate(img, a)
    rot2 = misc.imrotate(img, -a)
    return rot, rot2

def randomCrop(img):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    # perc = [0.1, 0.15, 0.2,0.3,0.4]
    # res = []
    # for ii in perc:
    #     h, w, _ = img.shape
    #     crop_width = int(w*ii)
    #     crop_height = int(h*ii)
    #     max_x = img.shape[1] - crop_width
    #     max_y = img.shape[0] - crop_height
    #     x = random.randint(0, int(crop_width))
    #     y = random.randint(0, int(crop_height))
    #     print(x, y)
    #     crop = img[y: y + max_y, x: x + max_x,:]
    #     res.append(crop)

    # return res

    perc = 0.2
    res = []
    h, w, _ = img.shape
    crop_width = int(w*perc)
    crop_height = int(h*perc)
    max_x = img.shape[1] - crop_width
    max_y = img.shape[0] - crop_height
    x = random.randint(0, int(crop_width))
    y = random.randint(0, int(crop_height))
    print(x, y)
    crop = img[y: y + max_y, x: x + max_x,:]

    return crop

def crop(img):
    topleft = img[32:, 32:, :]
    topright = img[32:, :-32, :]
    bottomleft = img[:-32, 32:, :]
    bottomright = img[:-32, :-32, :]
    center = img[16:-16, 16:-16, :]
    return topleft, topright, bottomleft, bottomright, center
    # return center

def color_jitter(im, brightness=0, contrast=0, saturation=0, hue=0):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    hsv[:,:,0] = hsv[:,:,0] + 14.32
    hsv[:,:,0] = np.where(hsv[:,:,0] > 179, 179, hsv[:,:,0])
    hsv[:,:,1] = hsv[:,:,1] + 35.7
    hsv[:,:,1] = np.where(hsv[:,:,1] > 255, 255, hsv[:,:,1])
    hsv[:,:,2] = hsv[:,:,2] + 20.4
    hsv[:,:,2] = np.where(hsv[:,:,2] > 255, 255, hsv[:,:,2])
    rst = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = np.ascontiguousarray(rst).astype(np.float32)
    
    return img 

def fancyPCA(img, alpha_std=0.1): 
    res = []

    orig_img = img.astype(float).copy()

    img = img / 255.0  # rescale to 0 to 1 range

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation (not once per channel)
    alpha = np.random.normal(0, alpha_std)

    # broad cast to speed things up
    m2[:, 0] = alpha * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    # about 100x faster after vectorizing the numpy, it will be even faster later
    # since currently it's working on full size images and not small, square
    # images that will be fed in later as part of the post processing before being
    # sent into the model
#     print("elapsed time: {:2.2f}".format(time.time() - start_time), "\n")

    return orig_img 

def edgeEnhancement(img):
     kernel = [[-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]]
     kernel = np.array(kernel).astype(np.float32)
     img = cv2.filter2D(img,-1,kernel,borderType=cv2.BORDER_CONSTANT)
     return img

def edge(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    
    sobel = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
    sobel = exposure.rescale_intensity(sobel, in_range='image', out_range=(-255,255)).clip(0,255).astype(np.uint8)
    #gray = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY)
    invert = cv2.bitwise_not(sobel)
    
    img_pil = Image.fromarray(img, 'RGBA')
    invert_pil = Image.fromarray(invert, 'RGBA')
    img_pil.putalpha(1)
    invert_pil.putalpha(1)
    #rst = Image.alpha_composite(img_pil, invert_pil)
    rst = Image.blend(img_pil, invert_pil, .4)
    
    rst = cv2.cvtColor(np.array(rst), cv2.COLOR_RGB2BGR)
    return rst

    

def peterGriffinIsAGayDogFuck(root, org):
    methods = ['rot', 'flip', 'pca', 'crop', 'crop2', 'edge', 'edge2', 'jitter']
    if not osp.exists(osp.join(root, 'rot30')):
        os.mkdir(osp.join(root, 'rot30'))
    if not osp.exists(osp.join(root, 'flipflop')):
        os.mkdir(osp.join(root, 'flipflop'))
    if not osp.exists(osp.join(root, 'fancyPCA')):
        os.mkdir(osp.join(root, 'fancyPCA'))
    if not osp.exists(osp.join(root, 'edgeEnhancement')):
        os.mkdir(osp.join(root, 'edgeEnhancement'))
    if not osp.exists(osp.join(root, 'edgeEnhancement_T')):
        os.mkdir(osp.join(root, 'edgeEnhancement_T'))
    if not osp.exists(osp.join(root, 'randomCrop')):
        os.mkdir(osp.join(root, 'randomCrop'))
    if not osp.exists(osp.join(root, 'randomCrop_T')):
        os.mkdir(osp.join(root, 'randomCrop_T'))
    if not osp.exists(osp.join(root, 'jitter')):
        os.mkdir(osp.join(root, 'jitter'))


    for folder in os.listdir(osp.join(root, org)):
        if not osp.exists(osp.join(osp.join(root, 'rot30'), folder)):
            os.mkdir(osp.join(osp.join(root, 'rot30'), folder))
        if not osp.exists(osp.join(osp.join(root, 'flipflop'), folder)):
            os.mkdir(osp.join(osp.join(root, 'flipflop'), folder))
        if not osp.exists(osp.join(osp.join(root, 'fancyPCA'), folder)):
            os.mkdir(osp.join(osp.join(root, 'fancyPCA'), folder))
        if not osp.exists(osp.join(osp.join(root, 'edgeEnhancement'), folder)):
            os.mkdir(osp.join(osp.join(root, 'edgeEnhancement'), folder))
        if not osp.exists(osp.join(osp.join(root, 'edgeEnhancement_T'), folder)):
            os.mkdir(osp.join(osp.join(root, 'edgeEnhancement_T'), folder))
        if not osp.exists(osp.join(osp.join(root, 'randomCrop'), folder)):
            os.mkdir(osp.join(osp.join(root, 'randomCrop'), folder))

        if not osp.exists(osp.join(osp.join(root, 'randomCrop_T'), folder)):
            os.mkdir(osp.join(osp.join(root, 'randomCrop_T'), folder))
        if not osp.exists(osp.join(osp.join(root, 'jitter'), folder)):
            os.mkdir(osp.join(osp.join(root, 'jitter'), folder))

        rot30_path = osp.join(osp.join(root, 'rot30'), folder)
        flipflop_path = osp.join(osp.join(root, 'flipflop'), folder)
        fancyPCA_path = osp.join(osp.join(root, 'fancyPCA'), folder)
        edgeEnhancement_path = osp.join(osp.join(root, 'edgeEnhancement'), folder)
        edgeEnhancement_T_path = osp.join(osp.join(root, 'edgeEnhancement_T'), folder)
        randomCrop_path = osp.join(osp.join(root, 'randomCrop'), folder)
        randomCrop_T_path = osp.join(osp.join(root, 'randomCrop_T'), folder)
        jitter_path = osp.join(osp.join(root, 'jitter'), folder)

        org_path = osp.join(osp.join(root, org), folder)

        for name in os.listdir(osp.join(osp.join(root, org), folder)):
            print('{}/{}/{}/{}'.format(root, org, folder, name))
            img = imageio.imread('{}/{}/{}/{}'.format(root, org, folder, name), pilmode="RGB")
            img = np.ascontiguousarray(img).astype(np.float32)
            rot30, rot_30 = rotate(img)
            img_horFlip = horizontalFlip(img)
            img_pca = fancyPCA(img)
            # img_crop1, img_crop2, img_crop3, img_crop4, img_crop5 = randomCrop(img)
            # img_crop1 = randomCrop(img)
            img_crop_T1, img_crop_T2, img_crop_T3,img_crop_T4,img_crop_T5 = crop(img)
            # img_crop_T1= crop(img)
            # img_edge = edgeEnhancement(img)
            img_edge_T = edge(img)
            img_jitter = color_jitter(img, 0.2, 0.2, 0.6, 0.6)

            neim, ext = name.split('.')[0], name.split('.')[-1]
            name_rot30 = neim + '_rot30' + '.' + ext
            name_rot_30 = neim + '_rot_30' + '.' + ext
            name_pca = neim + '_pca' + '.' + ext
            name_crop1 = neim + '_crop1' + '.' + ext
            name_crop2 = neim + '_crop2' + '.' + ext
            name_crop3 = neim + '_crop3' + '.' + ext
            name_crop4 = neim + '_crop4' + '.' + ext
            name_crop5 = neim + '_crop5' + '.' + ext
            name_edge = neim + '_edge' + '.' + ext
            name_horFlip = neim + '_horFlip' + '.' + ext
            name_jitter = neim + '_jitter' + '.' + ext

            misc.imsave(rot30_path + '/' + name_rot30, rot30)
            misc.imsave(rot30_path + '/' + name_rot30, rot_30)
            misc.imsave(fancyPCA_path + '/' + name_pca, img_pca)
            misc.imsave(randomCrop_T_path + '/' + name_crop1, img_crop_T1)
            misc.imsave(randomCrop_T_path + '/' + name_crop2, img_crop_T2)
            misc.imsave(randomCrop_T_path + '/' + name_crop3, img_crop_T3)
            misc.imsave(randomCrop_T_path + '/' + name_crop4, img_crop_T4)
            misc.imsave(randomCrop_T_path + '/' + name_crop5, img_crop_T5)
            misc.imsave(edgeEnhancement_T_path + '/' + name_edge, img_edge_T)
            misc.imsave(flipflop_path + '/' + name_horFlip, img_horFlip)
            misc.imsave(jitter_path + '/' + name_jitter, img_jitter)

            # misc.imsave(org_path + '/' + name_rot30, img_rot30)
            # misc.imsave(org_path + '/' + name_pca, img_pca)
            # misc.imsave(org_path + '/' + name_crop, img_crop)
            # misc.imsave(org_path + '/' + name_edge, img_edge)
            # misc.imsave(org_path + '/' + name_verFlip, img_verFlip)
            # misc.imsave(org_path + '/' + name_horFlip, img_horFlip)

root = '.'
org = 'org'
peterGriffinIsAGayDogFuck(root, org)