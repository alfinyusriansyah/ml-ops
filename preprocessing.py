import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder


def train_data(train_dir):
    train_data = []

    for r, d, f in os.walk(train_dir): 
        for file in f: 
            if ".jpg" in file: 
                imagePath = os.path.join(r, file) 
                image = cv2.imread(imagePath) 
                image = cv2.resize(image, (250,250)) 
                train_data.append(image) 
    train_data = np.array(train_data)            
    return train_data

def train_label(train_dir):
    train_label = []
    for r, d, f in os.walk(train_dir): 
        for file in f: 
            if ".jpg" in file: 
              imagePath = os.path.join(r, file) 
              label = imagePath.split(os.path.sep)[-2] 
              train_label.append(label)
    train_label = np.array(train_label)              
    return train_label


def test_data(testi_dir):
    test_data = []
    for r, d, f in os.walk(testi_dir):
        for file in f:
            if ".jpg" in file:
                imagePath = os.path.join(r, file)
                image = cv2.imread(imagePath)
                image = cv2.resize(image, (250,250))
                test_data.append(image)
    test_data = np.array(test_data)                
    return test_data

def test_label(testi_dir):  
    test_label = []    
    for r, d, f in os.walk(testi_dir):
      for file in f:
        if ".jpg" in file:
            imagePath = os.path.join(r, file)          
            label = imagePath.split(os.path.sep)[-2]
            test_label.append(label)
    test_label = np.array(test_label)            
    return test_label      

#normalisasi data
def normalisasi(dt_train, dt_test):
    x_train = dt_train.astype('float32') / 255.0
    x_test = dt_test.astype('float32') / 255.0  
    return x_train, x_test  
    
#label encoding
def label_encod(dt_train_label, dt_test_label):
    lb = LabelEncoder()
    y_train = lb.fit_transform(dt_train_label)
    y_test = lb.fit_transform(dt_test_label)
    return y_train, y_test
          











