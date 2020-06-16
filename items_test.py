import os
from tensorflow import keras
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import cv2 as cv
from sklearn.neighbors import NearestNeighbors
import sys 

f=__file__
f=f.replace('items_test.py', '')
root_dir=f+'ROOM ITEMS'
model = keras.models.load_model(f+'model.h5')


labels=["BEDSIDE CLOCK","BEDSIDE TABLE","BLANKET BAGS",'CLOTH BRUSH & SHOE HORN',
        'CLOTH HANGER','COFFE & TEA PRESENTATION','DESK ORGANISERS',"Do Not Disturb Sign",
        "GUEST ROOM BLOTTERS","GUEST ROOM COASTERS"]
# predicting images 
classes=[]
query_path=sys.argv[1]
#query_path = input("Enter test path : ")
img = image.load_img(query_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

for x in range(10):
  if classes[0][x]>0.5:
    cls=labels[x]
    print("Classified as ",cls)

#traning
print("Training Started")
img_dir = os.path.join(root_dir+'/'+cls)
img=os.listdir(img_dir)
img_paths=[]
for i in img:
  s=img_dir+'/'+i
  img_paths.append(s)

model1 = VGG16(weights='imagenet', include_top=False)

img_vector_features = []
for img_path in img_paths:
  img = image.load_img(img_path, target_size=(150,150))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = preprocess_input(img_data)

  vgg16_feature = model1.predict(img_data)
  vgg16_feature = np.array(vgg16_feature)
  vgg16_feature = vgg16_feature.flatten()
  img_vector_features.append(vgg16_feature)

#testing
print("Testing Started")
img = image.load_img(query_path, target_size=(150,150))
img_data = image.img_to_array(img)
img_data = np.expand_dims(img_data, axis=0)
img_data = preprocess_input(img_data)

vgg16_feature = model1.predict(img_data)
vgg16_feature = np.array(vgg16_feature)
query_feature = vgg16_feature.flatten()

# Numbers of similar images that we want to show
if len(img_paths)<10:
  N_QUERY_RESULT = len(img_paths)
else: 
  N_QUERY_RESULT = 9
nbrs = NearestNeighbors(n_neighbors=N_QUERY_RESULT, metric="cosine").fit(img_vector_features)

distances, indices = nbrs.kneighbors([query_feature])
similar_image_indices = indices.reshape(-1)

print("Got {} similar images.".format(N_QUERY_RESULT))

img = cv.imread(query_path)
imgplot = cv.imshow("Input",img)
for i in range(N_QUERY_RESULT):
  img = cv.imread(img_paths[similar_image_indices[i]]) 
  imgplot = cv.imshow("Output {}".format(i+1),img)
cv.waitKey(0)
cv.destroyAllWindows()
