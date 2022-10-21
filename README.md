# Classification-of-Foliar-Diseases-in-Apple-Leaves-using-Ensemble-CNN

## Project Description
#### Context
This is my final-year team project completed to fulfill the Bachelor's degree in Computer Science and Engineering award.
**Domain: Deep Learning(Classification task) and Image Processing**

#### Duration
9 months (almost one academic year)

#### Tools
- Tensorflow & Keras
- OpenCV
- Google Colab Pro(Just flexing)
- Statistical libraries such as Matplotlib, Pandas, NumPy

## Overview
Plant leaf diseases are a major threat to the growth of the respective species in agricultural production. As a result, reduced yield rates can lead to indeterminable economic downfall. Therefore, detection and classification of plant leaf diseases play a significant role in agricultural production. While many machine learning approaches such as Naïve Bayes and Support Vector Machines and deep learning approaches such as Convolutional Neural Network, that use single model base, exist to detect, and classify plant leaf diseases, hybrid modelling and ensemble approaches are least explored and examined. Therefore, there is a need to research and evaluate the ensemble approaches because it can produce improvements in accuracy enhancements. In this report, we present an ensemble CNN model for classifying foliar diseases in apple tree leaves. The two models that reported improved accuracy when combined, are Xception and DenseNet121 from the list of keras pre-trained models.

## Image Processing
We utilized the color spaces from the OpenCV module for image processing. Color spaces can amplify the extent to which the models decipher the differences in diseases in a leaf image and, therefore, classify the disease accurately.

We tried inferno, bone, hsv, jet, rainbow, and ocean color spaces from the OpenCV module.
!(/bone.jpeg)
!(/hsv.jpeg)
!(/rainbow.jpeg)
!(/ocean.jpeg)

## Ensemble Model Architecture
!(/model2.jpeg)

## Instructions to run the project
1. Clone/Download the repository.
2. Run the model.py and colormap.py files in google colaboratory. This project requires high GPU or TPU for processing images. You can download any plant pathology dataset from kaggle datasets. For this project, we have used the plant pathology-2021 dataset. After training the model, save and download the model weights.
3. Open the app.py file in any of the code editors.
4. In app.py file, enter the path to your model weights.
4. To run the application, you must install streamlit and install appropriate dependency versions listed in this repository.
5. Open the terminal in your system and type the command -"streamlit run app.py".
6. The application will open in one of the web browsers.

## Team members who contributed to the project
1. Mohammed Hussam Khatib(https://github.com/hussamkhatib)
2. Mohammed Saqlain
3. Nashra Tanseer

## Note
The model weights are not added to this repository because the weights file is more than 300MB. A suggestion is to run modify the code according to your requirements, train the model, and save the weights as weights_filename.h5
