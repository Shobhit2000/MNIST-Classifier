import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
import cv2

np.random.seed(0)
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

Y_train = to_categorical(Y_train,10)
Y_test = to_categorical(Y_test,10)
X_train= X_train/255
X_test=X_test/255
no_of_pixels=784

X_train=X_train.reshape(X_train.shape[0],784)
X_test=X_test.reshape(X_test.shape[0],784)
def create_model():
    model = Sequential()
    model.add(Dense(no_of_pixels, input_dim = no_of_pixels, activation = 'relu'))
    model.add(Dense(784, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(Adam(lr = 0.01), loss = 'categorical_crossentropy', metrics =['accuracy'])
    return model
model=create_model()
print(model.summary())
operate= model.fit(X_train, Y_train , validation_split=0.1 ,epochs=10, batch_size=200 , verbose=1 , shuffle =1)

plt.plot(operate.history['loss'])
plt.plot(operate.history['acc'])
plt.plot(operate.history['val_loss'])
plt.plot(operate.history['val_acc'])
plt.legend(['loss','acc','validation loss' ,'validation acc'])
plt.title(['loss and accuracy'])
plt.xlabel(['no of epochs'])

score = model.evaluate(X_test,Y_test,verbose =1)
print(type(score))
print('Test Score',score[0])
print('Test Accuracy',score[1])

img= cv2.imread('sample.png')
img= cv2.resize(img ,(28,28))
img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')

image= cv2.bitwise_not(img)
plt.imshow(image, cmap='gray')

image = image/255
image = image.reshape(1, 784)

prediction = model.predict_classes(image)
print(str(prediction))
