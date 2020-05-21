# DL_Project1
Classification of Dogs vs Cats Images using CNN

This repository contains ipython notebook.This notebook has the code to build a CNN model which classifies the cats-vs-dogs data. First, clone the data from the repository. Using image preprocessing, images are taken in (32*32) size. And it is rescaled to 1/255 for using it into the model.

model=Sequential()
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer='he_uniform',input_shape=(img_width,img_height,3)))
model.add(MaxPool2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())
model.add(Dense(units=128,activation='relu',kernel_initializer='he_uniform'))
model.add(Dense(units=1,activation='sigmoid'))

Sequential is imported from keras. 
1 Conv2D layer is added with 64 filters each of (3*3) size and relu as activation function.
1 MaxPool2D layer is added with pool size of (2*2) and strides 2.
Flatten is used to flatten the data.
Then, 2 dense layers(fully connected neural networks) are added. First with 128 units and relu as activation function.
And, Second with 1 unit and sigmoid as activation function because there is binary classification.

SGD is used as optimizer with a momentum of 0.9.

Model is created and fit with X_train and Y_train with 5 no. of iteration.
Now,learning curve is plotted.
Result:
loss: 0.2183 - accuracy: 0.9078 - val_loss: 0.7284 - val_accuracy: 0.7380

After that, 3 blocks of VGG is implemneted.
This is implemented same as above with 3 Conv2D layers each followed by MaxPool2D layer. First with 64 filters, second with 128 filters and 3rd with 256 filters.

Result:
loss: 0.1862 - accuracy: 0.9220 - val_loss: 0.6184 - val_accuracy: 0.7892

at last, Dropout and BatchNormalization is used in the above model and finally, result improves.

Result:
loss: 0.4296 - accuracy: 0.8056 - val_loss: 0.4044 - val_accuracy: 0.8126
