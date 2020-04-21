#Importing Necessary Libraries
import numpy as np
import cv2 
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from os import listdir
from sklearn.model_selection import train_test_split
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


default_image_size = tuple((64, 64))
image_dir = 'Leukemia Project/Leukemia_Dataset/fold_0'
directory_root = 'Leukemia Project/Leukemia_Dataset'

#Setting all the images in the directory to the set default size and converting them into Numpy array
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None
    
#Storing the images and the class labels in two seperate lists   
image_list,label_list = [],[]
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)

    for directory in root_dir :
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for leukemia_folder in root_dir :
        leukemia_folder_list = listdir(f"{directory_root}/{leukemia_folder}")
        print(f"[INFO] Processing {leukemia_folder} ...")
    
    for leukemia_disease_folder in leukemia_folder_list:
            for dir1 in leukemia_disease_folder :
              if dir1 == ".DS_Store" :
                leukemia_folder_list.remove(dir1)
                print("removed")
            print(f"[INFO] Processing {leukemia_disease_folder} ...")
            leukemia_disease_image_list = listdir(f"{directory_root}/{leukemia_folder}/{leukemia_disease_folder}/")
    
            for image in leukemia_disease_image_list:
                image_directory = f"{directory_root}/{leukemia_folder}/{leukemia_disease_folder}/{image}"
                if image_directory.endswith(".bmp") == True or image_directory.endswith(".BMP") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(leukemia_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")
    
#Data Preprocessing - Images are Preprocessed before passing them to the model

#Converting string class labels to integers
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
print(n_classes)

#Dividing the image pixels by 255 since the pixels range from 0-255 thus scaling them to range between 0 and 1, thus preventing biasing in the model
np_image_list = np.array(image_list, dtype=np.float16) / 255.0

#Splitting the dataset into training set and test set
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 42)

img = load_img('Leukemia_Dataset/Train/Fold_0_spyder/fold_0/hem/UID_H6_1_1_hem.bmp')
img1= cv2.imread("Leukemia_Dataset/Train/Fold_0_spyder/fold_0/hem/UID_H6_1_1_hem.bmp")
img_cvt = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB) 
plt.imshow(img_cvt) 
plt.show()

data = img_to_array(img)
samples = expand_dims(data, 0)

#Data Augmentation - To prevent overfitting of the model by introducing variations within the dataset and increase the size of the dataset
train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        horizontal_flip=True
       )

#Displaying sample of data augmentation
it = train_datagen.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()

#CNN Model using Keras
classifier = Sequential()
classifier.add(Convolution2D(32, 5, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 2, 2, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Convolution2D(32, 2, 2, activation = 'relu')) 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Saving the model in .h5 file
classifier.save("L5.h5")

classifier.summary()

#Training of the CNN model
BS = 8
history = classifier.fit_generator(
train_datagen.flow(x_train, y_train, batch_size=BS),
validation_data=(x_test, y_test),
steps_per_epoch=len(x_train) // BS,
epochs=180
)

#Graph to plot Epochs vs Accuracy for training and test set
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Graph to plot Epochs vs Loss for training and test set
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Predictions made by the model on Test data
y_pred =  classifier.predict(x_test)

#Model performance evaluation on Test data
print( classification_report(y_test, y_pred.round()))
print(confusion_matrix(y_test, y_pred.round()))