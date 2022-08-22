# Imporiting Necessary Libraries
import base64
import streamlit as st
from PIL import Image
import io
import numpy as np
import pandas as pd
import cv2
import fpdf
import json
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache


def classification1_page():
    @st.cache(allow_output_mutation=True)
    def load_model(path):

        # Xception Model
        xception_model = tf.keras.models.Sequential([
            tf.keras.applications.xception.Xception(
                include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        # DenseNet Model
        densenet_model = tf.keras.models.Sequential([
            tf.keras.applications.densenet.DenseNet121(
                include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        # Ensembling the Models
        inputs = tf.keras.Input(shape=(512, 512, 3))

        xception_output = xception_model(inputs)
        densenet_output = densenet_model(inputs)

        outputs = tf.keras.layers.average([densenet_output, xception_output])

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Loading the Weights of Model
        model.load_weights(path)

        return model

    # Loading the Model
    model = load_model('model.h5')

    # Title and Description
    st.title('Apple Leaf Disease Classification')
    st.write("Upload the apple leaf image and get the classifications!")

    # Setting the files that can be uploaded
    uploaded_file = st.file_uploader(
        "Choose a Image file", type=["png", "jpg"])

    # If there is a uploaded file, start making prediction
    if uploaded_file != None:

        # Display progress and text
        progress = st.text("Processing Image")
        my_bar = st.progress(0)
        i = 0

        # Reading the uploaded image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        data = uploaded_file.getvalue()
        image_cv = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        im3 = cv2.applyColorMap(image_cv, cv2.COLORMAP_JET)
        im4 = cv2.applyColorMap(image_cv, cv2.COLORMAP_BONE)
        im5 = cv2.applyColorMap(image_cv, cv2.COLORMAP_INFERNO)
        im6 = cv2.applyColorMap(image_cv, cv2.COLORMAP_OCEAN)
        im7 = cv2.applyColorMap(image_cv, cv2.COLORMAP_RAINBOW)
        im8 = cv2.applyColorMap(image_cv, cv2.COLORMAP_HSV)

        st.subheader('Original Image')
        st.image(np.array(Image.fromarray(np.array(image)).resize(
            (700, 400), Image.ANTIALIAS)), width=None)
        st.subheader('Jet Colormap Image')
        st.image(im3)
        st.subheader('Bone Colormap Image')
        st.image(im4)
        st.subheader('Inferno Colormap Image')
        st.image(im5)
        st.subheader('Ocean Colormap Image')
        st.image(im6)
        st.subheader('Rainbow Colormap Image')
        st.image(im7)
        st.subheader('HSV Colormap Image')
        st.image(im8)
        my_bar.progress(i + 40)

        # Cleaning the image
        image = clean_image(image)

        # Making the predictions
        predictions, predictions_arr = get_prediction(model, image)
        my_bar.progress(i + 30)

        # Making the results
        result = make_results(predictions, predictions_arr)

        # Removing progress bar and text after prediction done
        my_bar.progress(i + 30)
        progress.empty()
        i = 0
        my_bar.empty()

        scab_file = open('scab.json')
        rust_file = open('rust.json')
        scab_content = json.load(scab_file)
        rust_content = json.load(rust_file)

        # Show the results
        st.title(
            f"The leaf {result['status']} with {result['prediction']} classification probability.")
        if result['status'] == ' has Scab ':
            st.subheader("Apple Trees Affected: ")
            st.write(scab_content['Affected'])

            st.subheader("Symptoms: ")
            st.write(scab_content['Symptoms'])

            st.subheader("Causes: ")
            st.write(scab_content['Causes'])

            st.subheader("Treatment: ")
            st.write(scab_content['Treatment'])

            st.subheader("Risk: ")
            st.write(scab_content['Risk'])

        elif result['status'] == ' has Rust ':
            st.subheader("Apple Trees Affected: ")
            st.write(rust_content['Affected'])

            st.subheader("Symptoms: ")
            st.write(rust_content['Symptoms'])

            st.subheader("Causes: ")
            st.write(rust_content['Causes'])

            st.subheader("Treatment: ")
            st.write(rust_content['Treatment'])

            st.subheader("Risk: ")
            st.write(rust_content['Risk'])

        elif result['status'] == ' has Multiple Diseases ':
            st.subheader(
                "The given leaf image has multiple diseases - fire blight, black rot and many more.")
        else:
            result = "Healthy"
            st.subheader("The given image is healthy!")


def classification2_page():
    @st.cache(allow_output_mutation=True)
    def load_model(path):

        inputs = tf.keras.Input(shape=(256, 256, 3))
        x = tf.keras.applications.MobileNetV2(include_top=False)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(6, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs, outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Loading the Weights of Model
        model.load_weights(path)

        return model

    # Loading the Model
    model = load_model('model_2021_version2.h5')

    # Title and Description
    st.title('Apple Leaf Disease Classification')
    st.write("Upload the apple leaf image and get the classifications!")

    # Setting the files that can be uploaded
    uploaded_file = st.file_uploader(
        "Choose a Image file", type=["png", "jpg"])

    # If there is a uploaded file, start making prediction
    if uploaded_file != None:

        # Display progress and text
        progress = st.text("Processing Image")
        my_bar = st.progress(0)
        i = 0

        # Reading the uploaded image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        data = uploaded_file.getvalue()
        image_cv = cv2.imdecode(np.frombuffer(data, np.uint8), -1)
        im3 = cv2.applyColorMap(image_cv, cv2.COLORMAP_JET)
        im4 = cv2.applyColorMap(image_cv, cv2.COLORMAP_BONE)
        im5 = cv2.applyColorMap(image_cv, cv2.COLORMAP_INFERNO)
        im6 = cv2.applyColorMap(image_cv, cv2.COLORMAP_OCEAN)
        im7 = cv2.applyColorMap(image_cv, cv2.COLORMAP_RAINBOW)
        im8 = cv2.applyColorMap(image_cv, cv2.COLORMAP_HSV)

        st.subheader('Original Image')
        st.image(np.array(Image.fromarray(np.array(image)).resize(
            (700, 400), Image.ANTIALIAS)), width=None)
        st.subheader('Jet Colormap Image')
        st.image(im3)
        st.subheader('Bone Colormap Image')
        st.image(im4)
        st.subheader('Inferno Colormap Image')
        st.image(im5)
        st.subheader('Ocean Colormap Image')
        st.image(im6)
        st.subheader('Rainbow Colormap Image')
        st.image(im7)
        st.subheader('HSV Colormap Image')
        st.image(im8)
        my_bar.progress(i + 40)

        # Cleaning the image
        image = clean_image(image)

        # Making the predictions
        predictions, predictions_arr = get_prediction(model, image)
        my_bar.progress(i + 30)

        df = pd.read_csv('Train.csv')
        result = df.columns[predictions_arr+1]

        # Removing progress bar and text after prediction done
        my_bar.progress(i + 30)
        progress.empty()
        i = 0
        my_bar.empty()

        st.title(
            f"The leaf has {result} classification with probability distribution of {round(predictions.max(), 2)}.")


def about_us():
    col1, col2 = st.columns(2)

    with col1:
        st.header("Nimesh Mohanakrishnan")
        st.image("./images/Nimesh.jpg")
        st.header("Mohammed Saqlain")
        st.image("./images/Saqlain.jpeg")

    with col2:
        st.header("Mohammed Hussam Khatib")
        st.image("./images/Hussam.jpeg")
        st.header("Nashra Tanseer")
        st.image("./images/Nashra1.jpeg")


def about_project():
    st.title("Classification of Foliar Diseases in Apple Leaves")
    st.write("Diseases in apple leaves are a major threat to the quality growth of apple fruit. While many machine learning and deep learning approaches that use single model base, exist to detect, and classify plant leaf diseases, hybrid modelling and ensemble approaches are least explored and examined. In this project, we focus on developing an ensemble CNN model to accuractely classify the diseases present in apple tree leaves.")
    st.write(
        "Appropriate apple leaf images are collected from a kaggle repository. The images present in the kaggle repository were mined from [khanlab](https://blogs.cornell.edu/applevarietydatabase/machine-learning-for-disease-detection/) - a research lab at cornell university.")

    st.header("Image Classes")
    st.write("**Classes:**  Healthy, Cedar Rust, Apple Scab, and Multiple Diseases")
    st.caption("Apple Scab")
    st.image("Train_0.jpg")
    st.caption("Multiple Disease")
    st.image("Train_1.jpg")
    st.caption("Healthy")
    st.image("Train_2.jpg")
    st.caption("Cedar Rust")
    st.image("Train_3.jpg")

    st.header("Tech Stack")
    dataframe = {"Technology": ['ML Library', 'Colormap Module', 'Statistical Module', 'Front-End'],
                 "Items": ['Tensorflow, Keras, and Sklearn', 'OpenCV', 'Numpy and Pandas', 'Streamlit']}
    df = pd.DataFrame(data=dataframe)
    st.table(df)

    st.header("Parameters for Model Learning")
    dataframe3 = {'Parameters': ['Starting Learning Rate', 'Maximum Learning Rate', 'Exponential Decay', 'Epochs', 'Activation Function', 'Optimizer', 'Loss Function', 'Metrics',
                                 'Input Shape', 'Monitoring', 'Colormap', 'No. of Outputs'], 'Values': ['0.00001', '0.0001', '0.8', '25', 'Softmax', 'Adam', 'Categorical_Crossentropy', 'Accuracy', '(512, 512, 3)', 'Validation Accuracy', 'Jet', '4']}
    df3 = pd.DataFrame(data=dataframe3)
    st.table(df3)

    st.header("Models")
    st.write("We have experimented with different sets of ensemble models. Of all, **DenseNet121 and Xception** ensemble models gave the best results.")
    dataframe2 = {'Model 1': ['Xception', 'InceptionResNetV2', 'DenseNet201'], 'Model 2': ['DenseNet121', 'ResNet152V2', 'DenseNet169'], 'Accuracy': [
        99.86, 99.81, 99.89], 'Validation Accuracy': [100, 99.91, 99.91], 'Loss': [0.9, 2.7, 1.1], 'Validation Loss': [0.24, 2.9, 2.3]}
    df2 = pd.DataFrame(data=dataframe2)
    st.table(df2)
    st.write("Best Ensemble Model Performance- Xception & DenseNet121")

    st.subheader("Xception Model")
    st.write("Layers:\n1. Input Layer\n2. Xception Model Layers\n3. GlobalPoolingAverage2d Layer\n4. Dense Layer(Output Layer)")
    xceptipn_code = '''xception_model = tf.keras.models.Sequential([
  tf.keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
   tf.keras.layers.GlobalAveragePooling2D(),
   tf.keras.layers.Dense(4,activation='softmax')
])
xception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
xception_model.summary()'''
    st.code(xceptipn_code, language='python')

    st.subheader("DenseNet121 Model")
    st.write("Layers:\n1. Input Layer\n2. DenseNet121 Model Layers\n3. GlobalPoolingAverage2d Layer\n4. Dense Layer(Output Layer)")
    densenet121_code = '''densenet_model = tf.keras.models.Sequential([
   tf.keras.applications.DenseNet201(include_top = False, weights='imagenet',input_shape=(512, 512, 3)),
   tf.keras.layers.GlobalAveragePooling2D(),
   tf.keras.layers.Dense(4,activation='softmax')
])
densenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
densenet_model.summary()'''
    st.code(densenet121_code, language='python')

    st.subheader("Ensemble Xception & DenseNet121")
    ensemble_code = '''inputs = tf.keras.Input(shape=(512, 512, 3))

xception_output = xception_model(inputs)
densenet_output = densenet_model(inputs)

outputs = tf.keras.layers.average([densenet_output, xception_output])


model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()'''
    st.code(ensemble_code, language='python')


page_names_to_funcs = {
    "Project Details": about_project,
    "Ensemble Model Classification": classification1_page,
    "MobileNetV2 Model Classification": classification2_page,
    "Team": about_us,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
