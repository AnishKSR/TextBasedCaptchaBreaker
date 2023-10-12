import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

raw_image_directory = '/Users/anishravuri/Desktop/Junior Year/Semester 1/DATS 4001/captcha_dataset/samples'
processed_output_directory = '/Users/anishravuri/Desktop/Junior Year/Semester 1/DATS 4001/processed images'

data = []


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    r, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)  

    contours, n = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            cv2.drawContours(image, [contour], -1, (255, 255, 255), 1)
   
    inverted_image =  cv2.bitwise_not(image)
    #kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (3,3))
    
    
            
    return inverted_image





def new_preprocess(image):
    image = cv2.imread()


for filename in os.listdir(raw_image_directory):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(raw_image_directory, filename)

        img = cv2.imread(image_path)
        processed_image = preprocess(img)

        label =  os.path.splitext(filename)[0]
        data.append((processed_image, label))
        output_path = os.path.join(processed_output_directory, filename)
        cv2.imwrite(output_path, processed_image)


df = pd.DataFrame(data, columns=['Image', 'Label'])
for i in range(5):
    img = df.iloc[i]['Image']
    label = df.iloc[i]['Label']
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img) 
    plt.title(label)
    plt.axis('off')


plt.show()