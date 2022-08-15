'''
Importing necessary librariries
numpy and pandas - statiscal computing modules
tensorflow and keras - ML model libraries
sklearn - dataset split function
plotly, skimage, and matplotlib - data visualization libraries
cv2 - opencv module for generating colormaps and guassian blurring
'''
import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensordash.tensordash import Tensordash
import plotly.express as px
import json
import skimage.io as io
import cv2

# Reading the Training Data
dataset = pd.read_csv('<Enter path>')

# Display the dataset
dataset

# Checking if there are any null values in the dataset
dataset.isnull().any()

# Checking the column data type
dataset.dtypes

# Adding .jpg extension to every image_id
dataset['image_id'] = dataset['image_id']+'.jpg'

dataset

# Plotting histogram for healthy class
dataset.healthy.hist()
plt.title('Healthy Classes')

# Plotting histogram for multiple diseases class
dataset.multiple_diseases.hist()
plt.title('Multiple Diseases Classes')

# Plotting histogram for rust class
dataset.rust.hist()
plt.title('Rust Classes')

# Plotting histogram for scab class
dataset.scab.hist()
plt.title('Scab Classes')

# Visualizing the image classes
fig=plt.figure(figsize=(20, 14))
columns = 4
rows = 4
plt.title('Image Class')
plt.axis('off')
for i in range(1, columns*rows +1):
    img = plt.imread(f'<Enter path>')
    fig.add_subplot(rows, columns, i)
    
    if dataset.healthy[i] == 1:
        plt.title('Healthy')
    elif dataset.multiple_diseases[i] == 1:
        plt.title('Multiple Disease')
    elif dataset.rust[i] == 1:
        plt.title('Rust')
    else:
        plt.title('Scab')
    plt.imshow(img)
    plt.axis('off')
plt.show()

# Making histogram plots by changing the 2d array to flattened array(1d array)
fig=plt.figure(figsize=(20, 14))
columns = 4
rows = 4
plt.axis('off')
for i in range(1, columns*rows +1):
    img = plt.imread(f'<Enter path>')
    fig.add_subplot(rows, columns, i)
    plt.hist(img.ravel(), bins=32, range=[0, 256])
plt.show()

# Checking the image shape
img.shape

''' 
ImageDataGenerator class performs in-place/on-the-fly data augmentation, meaning that the class: 
Accepts a batch of images used for training. Takes this batch and applies a series of random transformations 
to each image in the batch.

rescale - rescale=1./255 will convert the pixels in range [0,255] to range [0,1]. This process is also called Normalizing the input. 
Scaling every images to the same range [0,1] will make images contributes more evenly to the total loss

other parameters are self explained.
'''

datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        zca_whitening=False,
        rotation_range=180,
        zoom_range = 0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        vertical_flip=True)

'''
Splitting the dataset using train_test_split. This function is from sklearn library.
shuffle = True implies that the images are shuffled and selected randomly for training and testing.
test size is 20%
train size is 80%
'''

X_train, X_valid = train_test_split(dataset, test_size=0.2, shuffle=True)

"""## Making a Tensorflow Dataset"""

'''
The batch size is a number of images processed before the model is updated. 
The number of epochs is the number of complete passes through the training dataset. 
The size of a batch must be more than or equal to one and less than or equal to the number of images in the training dataset.

train_generator is responsible for generating training data.
valid_generator is responsible for generating validation data.
'''
BATCH_SIZE = 64

train_generator = datagen.flow_from_dataframe(dataset, 
                    directory='<Enter path>',
                    x_col='image_id',
                    y_col=['healthy', 'multiple_diseases', 'rust', 'scab'], 
                    target_size=(512, 512), 
                    class_mode='raw',
                    batch_size=BATCH_SIZE, 
                    shuffle=True)

valid_generator = datagen.flow_from_dataframe(X_valid, 
                    directory='<Enter path>',
                    x_col='image_id',
                    y_col=['healthy', 'multiple_diseases', 'rust', 'scab'], 
                    target_size=(512, 512), 
                    class_mode='raw',
                    batch_size=BATCH_SIZE,
                    shuffle=True)

# Visualizaing generator images
fig=plt.figure(figsize=(20, 14))
columns = 2
rows = 4
plt.title('Image Class')
plt.axis('off')
for i in range(1, columns*rows):
    
    img_batch, label_batch = train_generator.next()
    fig.add_subplot(rows, columns, i)
    
    if label_batch[i][0] == 1:
        plt.title('Healthy')
    elif label_batch[i][1] == 1:
        plt.title('Multiple Disease')
    elif label_batch[i][2] == 1:
        plt.title('Rust')
    else:
        plt.title('Scab')
        
    plt.imshow(img_batch[i])
    plt.axis('off')
plt.show()

# Xception Model - Remove the dense layer in the sets of convolution layers
'''
include_top = False ensures that we eliminate the dense layers in the xception neural network.
Dense layers are added after convolution layers and few batch normalizers. 
This is simply because the fully connected layers at the end can only take fixed size inputs, 
which has been previously defined by the input shape and all processing in the convolutional layers. 
Any change to the input shape will change the shape of the input to the fully connected layers, 
making the weights incompatible (matrix sizes don't match and cannot be applied).

Global Average Pooling is a pooling operation designed to replace fully connected layers in classical CNNs. 
The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer.

The softmax function is used as the activation function in the output layer of neural network models that predict a 
multinomial probability distribution. That is, softmax is used as the activation function for multi-class classification 
problems where class membership is required on more than two class labels.

The results of the Adam optimizer are generally better than every other optimization algorithms, 
have faster computation time, and require fewer parameters for tuning.

Model training metrics is accuracy.
'''
xception_model = tf.keras.models.Sequential([
  tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
   tf.keras.layers.GlobalAveragePooling2D(),
   tf.keras.layers.Dense(4, activation='softmax')
])
xception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
xception_model.summary()

tf.keras.utils.plot_model(xception_model, to_file='xception_model.png')

densenet_model = tf.keras.models.Sequential([
   tf.keras.applications.DenseNet121(include_top = False, weights='imagenet',input_shape=(512, 512, 3)),
   tf.keras.layers.GlobalAveragePooling2D(),
   tf.keras.layers.Dense(4, activation='softmax')
])
densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
densenet_model.summary()

tf.keras.utils.plot_model(densenet_model, to_file='densenet_model.png')

'''
Feed same input to both models and average the outputs from both the models.
Gather the outputs from both the models. 
Average the outputs.
Averaged output corresponds to the ensemble model output.
Both models are trained simultaneously.
'''
inputs = tf.keras.Input(shape=(512, 512, 3))

xception_output = xception_model(inputs)
densenet_output = densenet_model(inputs)

outputs = tf.keras.layers.average([densenet_output, xception_output])


model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

tf.keras.utils.plot_model(model, to_file='model.png')

'''
LR = Leanring Rate

Learning rate decay (LR_EXP_DECAY) is a de facto technique for training modern neural networks. 
It starts with a large learning rate and then decays it multiple times. 
It is empirically observed to help both optimization and generalization.

Callbacks to the function are the update the necessary parameters (learning rate, decay).
'''

LR_START = 0.00001
LR_MAX = 0.0001 
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8
EPOCHS = 25

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

'''
model_checkpoint is used for storing the best values observed checkpoints.
After each epoch, the values are exmained and the best values get stored.
save_best_only stores the best observed and trained values.

Verbose is the choice that how you want to see the output of your Nural Network while it's training. 
If you set verbose = 0, It will show nothing.

Model weights are updated and rewritten into the model.h5 file.
'''
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', verbose=2, save_best_only=True)

"""# Training the Model"""

# Model training starts here
model_history = model.fit_generator(train_generator, epochs=EPOCHS, validation_data=valid_generator, callbacks=[model_checkpoint,lr_callback])

# Saving model history
pd.DataFrame(model_history.history).to_csv('ModelHistory.csv')

from google.colab import drive
drive.mount('/content/drive')

plt.plot(pd.DataFrame(model_history.history)['accuracy'])
plt.title("accuracy Plot")

plt.plot(pd.DataFrame(model_history.history)['loss'])
plt.title("Loss Plot")

plt.plot(pd.DataFrame(model_history.history)['val_accuracy'])
plt.title("Validation Accuracy Plot")

plt.plot(pd.DataFrame(model_history.history)['val_loss'])
plt.title("Validation Loss Plot")

"""# Predicting Classes"""

# Reading testing and submission data
test_dataset = pd.read_csv('<Enter path>')
submission = pd.read_csv('<Enter path>')
test_dataset

# Adding .jpg extension to image_id
test_dataset['image_id'] = test_dataset['image_id']+'.jpg'

test_gen = datagen.flow_from_dataframe(test_dataset, 
                    directory='<Enter path>',
                    x_col='image_id',
                    target_size=(512, 512), 
                    class_mode=None,
                    shuffle=False,
                    batch_size=8)

# Predicting class 
predictions = model.predict_generator(test_gen)

submission['healthy'] = predictions[:, 0]
submission['multiple_diseases'] = predictions[:, 1]
submission['rust'] = predictions[:, 2]
submission['scab'] = predictions[:, 3]

submission

submission.to_csv('submission.csv', index=False)
