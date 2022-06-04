import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

class DataSets:
    def __init__(self):
        self.train_cats_path = "Datasets/Train/cats"
        self.train_dogs_path = "Datasets/Train/dogs"
        self.test_cats_path = "Datasets/Test/cats"
        self.test_dogs_path = "Datasets/Test/dogs"
        
        self.train_tempsave_cat_path = "Datasets/Train/processedcats"
        self.train_tempsave_dog_path = "Datasets/Train/processeddogs"
        self.test_tempsave_cat_path = "Datasets/Test/processedcats"
        self.test_tempsave_dog_path = "Datasets/Test/processeddogs"
        
        self.train_cats_datalst = os.listdir(self.train_cats_path)
        self.train_dogs_datalst = os.listdir(self.train_dogs_path)
        self.test_cats_datalst = os.listdir(self.test_cats_path)
        self.test_dogs_datalst = os.listdir(self.test_dogs_path)
        
        # Train Cats
        self.preprocessing_loadImage(self.train_cats_path, 
                                     self.train_cats_datalst, 
                                     self.train_tempsave_cat_path)
        # Train Dogs
        self.preprocessing_loadImage(self.train_dogs_path, 
                                     self.train_dogs_datalst, 
                                     self.train_tempsave_dog_path)
        
        # Test Cats
        self.preprocessing_loadImage(self.test_cats_path, 
                                     self.test_cats_datalst, 
                                     self.test_tempsave_cat_path)
        
        # Test Dogs
        self.preprocessing_loadImage(self.test_dogs_path, 
                                     self.test_dogs_datalst, 
                                     self.test_tempsave_dog_path)

    def preprocessing_loadImage(self, path, imgs, savpath):
        print("START PREPROCESSING")
        newarr = []
        for i in range(0, len(imgs)):
            try:
                temp_name = path + "/" + imgs[i]
                img_gray = cv2.imread(temp_name, cv2.IMREAD_GRAYSCALE)

                if img_gray.shape[0] < img_gray.shape[1]:
                    img_gray = img_gray[0:img_gray.shape[0], 0 :img_gray.shape[0]]
                    img_gray = cv2.resize(img_gray, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                else:
                    img_gray = img_gray[0:img_gray.shape[1], 0 :img_gray.shape[1]]
                    img_gray = cv2.resize(img_gray, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                newarr.append(img_gray)
                tempsave = savpath + "/" + imgs[i]
                cv2.imwrite(tempsave, img_gray)
            
            except:
                pass
        
        # plt.imshow(newarr[1], cmap = "Greys_r")
        
    
    def loadAnswer(self, imgs):
        pass

DataSets()