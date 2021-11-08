import matplotlib.pyplot as plt
import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random
import scipy.stats
import cv2
from sklearn.metrics import f1_score, precision_score, recall_score
import csv

def compute_evaluation(prediction, target):
    """
    :param prediction: 2d array, int,
            estimated targets as returned by a classifier
    :param target: 2d array, int,
            ground truth
    :return:
        f1: float
    """
    prediction.tolist(), target.tolist()
    img, target = np.array(prediction).flatten(), np.array(target).flatten()
    f1 = f1_score(y_true=target, y_pred=img, average='binary')
    precision = precision_score(y_true=target, y_pred=img, average='binary')
    recall = recall_score(y_true=target, y_pred=img, average='binary')
    return f1, precision, recall




image_names = os.listdir('Data/val/masks')
mask_names = os.listdir('Ensemble/output')
total_acc = []
total_f1 = []
total_precision = []
total_recall = []

a1 = []
a2 = []
for i in range(len(image_names)):
    im1 = Image.open('Data/val/masks/' + image_names[i])
    im2 = Image.open('Ensemble/output/' + mask_names[i])

    img1 = cv2.imread(image_names[i])
    img2 = cv2.imread(mask_names[i])
    arr1 = np.array(im1, dtype=np.uint8)
    arr1 -= 1

    arr2 = np.array(im2, dtype=np.uint8)
    arr2[np.where(arr2 == 0)] = 1
    arr2[np.where(arr2 == 255)] = 0
    a1.append(arr1)
    a2.append(arr2)

a1 = np.array(a1).ravel()
a2 = np.array(a2).ravel()
a = np.array(a1 == a2)
ac = len(a[np.where(a == True)]) / np.shape(a1)[0]
f1 = f1_score(y_true=a1, y_pred=a2, average='binary')
precision = precision_score(y_true=a1, y_pred=a2, average='binary')
recall = recall_score(y_true=a1, y_pred=a2, average='binary')
print(ac,precision,recall,f1)


for i in range(len(image_names)):
    im1 = Image.open('Data/val/masks/'+image_names[i])
    im2 = Image.open('Ensemble/output/'+mask_names[i])


    img1 = cv2.imread(image_names[i])
    img2 = cv2.imread(mask_names[i])
    arr1 = np.array(im1, dtype=np.uint8)
    arr1 -= 1

    arr2 = np.array(im2, dtype=np.uint8)
    arr2[np.where(arr2 == 0)] = 1
    arr2[np.where(arr2 == 255)] = 0

    map_fnfp = np.zeros((512, 512, 3))
    map_colormap = np.zeros((512, 512, 3))
    diff = arr1 - arr2
    index0 = np.where(diff == 0)
    index1 = np.where(diff == 1)
    index255 = np.where(diff == 255)

    index_f = np.where(arr2 == 0)
    index_t = np.where(arr2 == 1)

    map_fnfp[index0[0], index0[1], :] = [255, 255, 255]
    map_fnfp[index1[0], index1[1], :] = [0, 0, 128]
    map_fnfp[index255[0], index255[1], :] = [128, 0, 0]


    map_colormap[index_f[0], index_f[1], :] = [0, 128, 0]
    map_colormap[index_t[0], index_t[1], :] = [128, 0, 0]
    #map_colormap[index1[0], index1[1], :] = [255, 255, 0]
    #map_colormap[index255[0], index255[1], :] = [255, 255, 0]

    '''
    t = np.where(diff == 0)
    f = np.where(diff == 255)
    diff[t] = 255
    diff[f] = 0
    '''

    #diff = Image.fromarray(diff)
    map_fnfp = Image.fromarray(map_fnfp.astype('uint8')).convert('RGB')
    map_fnfp.save('result0/fp&fn_'+str(i)+'.jpg')
    map_colormap = Image.fromarray(map_colormap.astype('uint8')).convert('RGB')
    map_colormap.save('result0/false_' + str(i) + '.jpg')

    acc = np.array(arr1 == arr2)
    accuracy = len(acc[np.where(acc == True)]) / (np.shape(arr1)[0] * np.shape(arr1)[1])
    f1, precision, recall = compute_evaluation(arr2, arr1)

    total_acc.append(accuracy)
    total_f1.append(f1)
    total_precision.append(precision)
    total_recall.append(recall)

    print(image_names[i], 'accuracy =', accuracy, '. f1 =', f1, '. precision =', precision, '. recall =', recall)



    dst = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    title = 'accuracy = ' + str(accuracy)+'\n'+'f1 = ' + str(f1)+'\n'+'precision = ' + str(precision)+'\n'+'recall = ' + str(recall)
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 3, 1)
    plt.title('ground truth', bbox=dict(facecolor='g', edgecolor='blue', alpha=0.65))
    plt.imshow(arr1)
    plt.subplot(1, 3, 2)
    plt.title('segmentation', bbox=dict(facecolor='g', edgecolor='blue', alpha=0.65))
    plt.imshow(arr2)


    plt.subplot(1, 3, 3)
    #plt.imshow(arr1)
    #masked_imclass = np.ma.masked_where(arr2 == 0, arr2)
    plt.imshow(arr1, alpha=0.5 )
    #masked_imclass = np.ma.masked_where(arr1 == 0, arr1)
    plt.imshow(arr2, alpha=0.5)
    plt.title(title, bbox=dict(facecolor='g', edgecolor='blue', alpha=0.65))
    path = 'result0/'+str(i)+'.jpg'
    plt.savefig(path)


def writer(total_acc, total_f1, total_precision, total_recall, path):
    f = open(path, 'w', encoding='utf-8', newline='' )
    csv_writer = csv.writer(f)

    for i in range(len(total_acc)):
        csv_writer.writerow([total_acc[i],total_f1[i],total_precision[i],total_recall[i]])

path={}
method = 'Ensemble_m'
path['FCN'] = 'eval_FCN.csv'
path['ResNet'] = 'eval_Res.csv'
path['Dense'] = 'eval_Dense.csv'
path['UNet'] = 'eval_UNet.csv'
path['Ensemble'] = 'eval_Ensemble.csv'
path['Ensemble_w'] = 'eval_Ensemble_w.csv'
path['Ensemble_v'] = 'eval_Ensemble_v.csv'
path['Ensemble_m'] = 'eval_Ensemble_m.csv'

writer(total_acc,total_f1,total_precision,total_recall,path[method])
'''
total_acc = np.delete(total_acc, np.argmax(total_acc))
total_f1 = np.delete(total_f1, np.argmax(total_f1))
total_precision = np.delete(total_precision, np.argmax(total_precision))
total_recall = np.delete(total_recall, np.argmax(total_recall))

total_acc = np.delete(total_acc, np.argmin(total_acc))
total_f1 = np.delete(total_f1, np.argmin(total_f1))
total_precision = np.delete(total_precision, np.argmin(total_precision))
total_recall = np.delete(total_recall, np.argmin(total_recall))
'''
print(np.mean(total_acc))
print(np.mean(total_precision))
print(np.mean(total_recall))
print(np.mean(total_f1))


