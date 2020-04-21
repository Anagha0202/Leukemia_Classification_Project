#Importing Necessary Libraries
import numpy as np
import cv2 
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from os import listdir
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression

default_image_size = tuple((299, 299))
image_dir = 'Leukemia_Dataset/Train/Fold_0_spyder/fold_0'
directory_root = 'Leukemia_Dataset/Train/Fold_0_spyder'

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

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)

n_classes = len(label_binarizer.classes_)
print(label_binarizer.classes_)
print(n_classes)

np_image_list = np.array(image_list, dtype=np.float16) / 255.0

x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 42)

train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        horizontal_flip=True
       )

classifier = Sequential()
classifier.add(Convolution2D(32, 5, 3, input_shape = (299, 299, 3), activation = 'relu'))

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

classifier.summary()

BS = 8
train_generator = train_datagen.flow_from_directory(
    image_dir,
    target_size=(default_image_size),
    batch_size=BS,
    class_mode='binary')

history = classifier.fit_generator(
train_datagen.flow(x_train, y_train, batch_size=BS),
validation_data=(x_test, y_test),
steps_per_epoch=len(x_train) // BS,
epochs=2
)

#Model Interpretation using Lime - Displays the features based on which the model has predicted for a particular image

Xi = skimage.io.imread("UID_H6_5_1_hem.bmp")
Xi = skimage.transform.resize(Xi, (299,299)) 
Xi = (Xi - 0.5)*2
skimage.io.imshow(Xi/2+0.5)

prediction = classifier.predict(np.array(x_test))

top_pred_classes = prediction[0].argsort()[-2:][::-1]

superpixels = skimage.segmentation.quickshift(Xi, kernel_size=4,max_dist=200, ratio=0.2)
num_superpixels = np.unique(superpixels).shape[0]

skimage.io.imshow(skimage.segmentation.mark_boundaries(Xi/2+0.5, superpixels))

num_perturb = 150
perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
perturbations[0] 

def perturb_image(img,perturbation,segments):
  active_pixels = np.where(perturbation == 1)[0]
  mask = np.zeros(segments.shape)
  for active in active_pixels:
      mask[segments == active] = 1 
  perturbed_image = copy.deepcopy(img)
  perturbed_image = perturbed_image*mask[:,:,np.newaxis]
  return perturbed_image

skimage.io.imshow(perturb_image(Xi/2+0.5,perturbations[0],superpixels))

predictions = []
for pert in perturbations:
  perturbed_img = perturb_image(Xi,pert,superpixels)
  pred = classifier.predict(perturbed_img[np.newaxis,:,:,:])
  predictions.append(pred)

predictions = np.array(predictions)

original_image = np.ones(num_superpixels)[np.newaxis,:]
distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()

kernel_width = 0.25
weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2))

class_to_explain = top_pred_classes[0]
simpler_model = LinearRegression()
simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
coeff = simpler_model.coef_[0]

num_top_features = 10
top_features = np.argsort(coeff)[-num_top_features:] 

mask = np.zeros(num_superpixels) 
mask[top_features]= True
skimage.io.imshow(perturb_image(Xi/2+0.5,mask,superpixels) )
