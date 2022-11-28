import torch
import torch.nn as nn
import torchvision
import PIL
import math
import os
import random
import numpy as np
from matplotlib import pyplot as plt


def train_test_split(DATA_DIR, seed = 0, train_ratio = 0.8):
    random.seed(seed)
    all_files = os.listdir(DATA_DIR + 'suboriginals')
    for i, f in enumerate(all_files):
        if len(f) <=  7:

            _ = all_files.pop(i)
            print('Irrelevant file found')
            break
    print('Clean completed')

    for i in range(len(all_files)):
        all_files[i] = all_files[i][:-4]
        
    split_idx = int(train_ratio * len(all_files))
    
    return all_files[:split_idx], all_files[split_idx:]
class data_loader:
    def __init__(self, names, DATA_DIR, batch_size, transforms):

        self.batch_size = batch_size
        self.idx = 0
        
        self.DATA_DIR = DATA_DIR
        
        image_folder = DATA_DIR + 'suboriginals/'
        mask_folder = DATA_DIR + 'submasks/'
        bboxes_folder = DATA_DIR + 'subbboxes/'
        self.transforms = transforms
        

        self.names = names
   
        random.shuffle(self.names)
        
        if len(self.names) % batch_size == 0:
            self.length = len(self.names) // batch_size
        else:
            self.length = len(self.names) // batch_size + 1
            
        self.mask_folder = mask_folder
        self.bboxes_folder = bboxes_folder
        self.image_folder = image_folder
        
    def __iter__(self):
        random.shuffle(self.names)
        self.idx = 0
        return self
    def __next__(self):
        if self.idx < self.length:
        
            batch = self.names[(self.idx * self.batch_size):((self.idx+1) * self.batch_size)]
            
            sequences = []
            bboxes = []
            masks = []
            for b in batch:
                i,m,n,t = [int(n) for n in b.split('-')]
                

                sequences.append(self.transforms(PIL.Image.open(self.image_folder + b + '.tif')))


                bbox = np.load(self.bboxes_folder + b + '.npy')
                mask = np.load(self.mask_folder + b + '.npy')
                bboxes.append(bbox)
                masks.append(mask)
            sequences = torch.stack(sequences)
        
            self.idx += 1
        
            return sequences, masks, bboxes, batch
        
        else:
            raise StopIteration
    def __len__(self):
        return self.length

class recurrent_loader:
    def __init__(self, names, DATA_DIR, batch_size, lags, transforms, time_padding = torch.zeros(1,313,313)):

        self.batch_size = batch_size
        self.idx = 0
        
        self.DATA_DIR = DATA_DIR
        
        image_folder = DATA_DIR + 'suboriginals/'
        mask_folder = DATA_DIR + 'submasks/'
        bboxes_folder = DATA_DIR + 'subbboxes/'
        self.time_padding = time_padding
        self.lags = lags
        self.transforms = transforms
        

        self.names = names
   
        random.shuffle(self.names)
        
        if len(self.names) % batch_size == 0:
            self.length = len(self.names) // batch_size
        else:
            self.length = len(self.names) // batch_size + 1
            
        self.mask_folder = mask_folder
        self.bboxes_folder = bboxes_folder
        self.image_folder = image_folder
        
    def __iter__(self):
        random.shuffle(self.names)
        self.idx = 0
        return self
    def __next__(self):
        if self.idx < self.length:
        
            batch = self.names[(self.idx * self.batch_size):((self.idx+1) * self.batch_size)]
            
            sequences = []
            bboxes = []
            masks = []
            for b in batch:
                i,m,n,t = [int(n) for n in b.split('-')]

                forward_images = []
                backward_images = []

                if i == 0:
                    max_t = 49
                else:
                    max_t = 18

                for time_idx in range(t - self.lags, t + 1):
                    if time_idx < 0 or time_idx > max_t:
                        forward_images.append(self.time_padding)
                    else:
                        forward_images.append(self.transforms(PIL.Image.open(self.image_folder + f'{i}-{m}-{n}-{time_idx}' + '.tif')))
                forward_images = torch.stack(forward_images)  
                for time_idx in range(t, t + self.lags + 1):
                    if time_idx < 0 or time_idx > max_t:
                        backward_images.append(self.time_padding)
                    else:
                        backward_images.append(self.transforms(PIL.Image.open(self.image_folder + f'{i}-{m}-{n}-{time_idx}' + '.tif')))
                backward_images = torch.stack(backward_images)  
                sequence = torch.stack((forward_images, backward_images))

                sequences.append(sequence.clone())


                bbox = np.load(self.bboxes_folder + b + '.npy')
                mask = np.load(self.mask_folder + b + '.npy')
                bboxes.append(bbox)
                masks.append(mask)
            sequences = torch.stack(sequences)
        
            self.idx += 1
        
            return sequences, masks, bboxes, batch
        
        else:
            raise StopIteration
    def __len__(self):
        return self.length


if __name__ == "__main__":

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Grayscale(),
                                                torchvision.transforms.Normalize([0.5], [0.5])])
    DATA_DIR = '../processdata/'
    train_names, test_names = train_test_split(DATA_DIR)
    train_loader = recurrent_loader(train_names, DATA_DIR, batch_size = 4, lags = 2, transforms = transforms)
    test_loader = recurrent_loader(test_names, DATA_DIR, batch_size = 4, lags = 2, transforms = transforms)


    for sequences, masks, bboxes, batch in iter(train_loader):
        forward_images, backward_images = sequences[0]
        plt.figure(figsize = (20,3))
        i = 0
        for im in forward_images[:-1]:
            i+=1
            plt.subplot(1,5,i)

            plt.imshow(im.permute(1,2,0), cmap = 'gray')
            plt.axis('off')
        for im in backward_images:
            i+=1
            plt.subplot(1,5,i)

            plt.imshow(im.permute(1,2,0), cmap = 'gray')
            plt.axis('off')
        plt.show()
        plt.close()
        
        plt.figure(figsize = (12,3))
        plt.subplot(1,3,1)
        plt.imshow(forward_images[-1][0],cmap = 'gray')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.imshow(np.array(PIL.Image.open(train_loader.DATA_DIR + 'sublabeled/' + batch[0] + '.tif')),cmap = 'gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.imshow(1 - masks[0].sum(0),cmap = 'gray')
        plt.axis('off')
        plt.show()
        plt.close()
        
        break

