# COMPUTER VISION
## IMAGE PREPROCESSING

<hr/>   

To make a demo version of Deep Neural Network we need to make proper datasets. First of all, we will make a DNN model which can classify dogs and cats. We can get datasets from Kaggle (https://www.kaggle.com/code/uysimty/keras-cnn-dog-or-cat-classification)
    
However each image of datasets are different. It will cause increasing of training time and increasing errors. Therefore we should adjust the size of each image in datasets.
    
    
Check the images in Datasets

    import cv2
    import os
    import matplotlib.pyplot as plt

    lst = os.listdir("Original/Train/cats")
    plt.figure(figsize = (15, 4))
    n = 10
    for i in range(n):
        ax = plt.subplot(2, 5, i+1)
        img_test = plt.imread("Original/Train/cats" + "/" + lst[i])
        # print(img_cv_gray.shape)
        plt.imshow(img_test)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


![Img](https://user-images.githubusercontent.com/87653966/171586749-bb9b6471-ac8c-485b-8829-81835dc9da88.png)

    
THe size of each image in datasets is different with others. Therefore, we need to set the size of images. We will change the all size of image like NxN. First of all, we check the size of image using ".shape" function in python. And when the width is shorter than the height, we set the width as N. The image will be changed N by N image. In addition, if the size of an image is big, we should decrease the size to improve the training time.

    def preprocessing_loadImage(self, path, imgs, savpath):
        print("START PREPROCESSING")
        newarr = []
        for i in range(0, len(imgs)):
            try:
                # Load Images
                temp_name = path + "/" + imgs[i]
                
                # Change to Gray Image
                img_gray = cv2.imread(temp_name, cv2.IMREAD_GRAYSCALE)

                # Compare Width and Height
                if img_gray.shape[0] < img_gray.shape[1]:
                    # Set NxN Square Image
                    img_gray = img_gray[0:img_gray.shape[0], 0 :img_gray.shape[0]]
                    
                    # Change size to 128 x 128
                    img_gray = cv2.resize(img_gray, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                else:
                    # Set NxN Square Image
                    img_gray = img_gray[0:img_gray.shape[1], 0 :img_gray.shape[1]]
                    
                    # Change size to 128 x 128
                    img_gray = cv2.resize(img_gray, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
                
                newarr.append(img_gray)
                
                # Save new images
                tempsave = savpath + "/" + imgs[i]
                cv2.imwrite(tempsave, img_gray)
            
            except:
                pass

### Let's check new images (128 x 128)

![img3](https://user-images.githubusercontent.com/87653966/171588927-2a2b65ec-b7c2-4ba4-ab85-1374ee71ed78.png)
    
Wow. We made new datasets!
    
If you change the size of each image in datasets. Please just change this parts of source.
    
    # dsize=(128, 128) means 128 x 128 image
    img_gray = cv2.resize(img_gray, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    

However, when you change the size of an image, the sharpness of an image will decrease like these:
### 64 x 64

    # dsize=(64, 64) means 64 x 64 image
    img_gray = cv2.resize(img_gray, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
    
![img64](https://user-images.githubusercontent.com/87653966/171590630-91aab707-8539-4254-9095-36efb2e82879.png)
    
### 32 x 32

    # dsize=(32, 32) means 32 x 32 image
    img_gray = cv2.resize(img_gray, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    
![img32](https://user-images.githubusercontent.com/87653966/171590799-8823d98f-bd29-46c6-9521-a9f1a35b5a0f.png)

### 16 x 16

    # dsize=(16, 16) means 16 x 16 image
    img_gray = cv2.resize(img_gray, dsize=(16, 16), interpolation=cv2.INTER_CUBIC)
    
![img16](https://user-images.githubusercontent.com/87653966/171590984-f0ce1b06-489c-450a-a840-27c56de14353.png)




