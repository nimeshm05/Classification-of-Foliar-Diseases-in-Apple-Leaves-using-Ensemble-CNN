# Imporiting Necessary Libraries
import base64
import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
import fpdf
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache


def main_page():
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

        #trees = "Apple Trees Affected:"
        #symptoms = "Symptoms:"
        #causes = "Causes:"
        #treatment = "Treatment:"
        #risk = "Risk:"
        #result = "Not Healthy"
        # Show the results
        st.title(
            f"The leaf {result['status']} with {result['prediction']} classification.")
        if result['status'] == ' has Scab ':
            st.subheader("Apple Trees Affected: ")
            treesInfo = "McIntosh, Cortland, and Macoun are susceptible to apple scab. There are many resistant cultivars."
            st.write(
                "McIntosh, Cortland, and Macoun are susceptible to apple scab. There are many resistant cultivars.")

            st.subheader("Symptoms: ")
            symptomsInfo = "Brown or olive green spots develop on apple tree leaves, which may then curl and fall off. On the apple, dark green spots appear on its surface, later to become darker, flaky, and even cracked. Infected fruit will usually drop, and infections may limit flower formation."
            st.write("Brown or olive green spots develop on apple tree leaves, which may then curl and fall off. On the apple, dark green spots appear on its surface, later to become darker, flaky, and even cracked. Infected fruit will usually drop, and infections may limit flower formation.")

            st.subheader("Causes: ")
            causesInfo = "Spores release from infected apple leaves that have remained on the ground through winter. These spores then infect nearby apple trees. Apple scab can also spread from nearby trees that are already infected. Frequent rains and prolonged leaf wetness enable severe scab infection conditions."
            st.write("Spores release from infected apple leaves that have remained on the ground through winter. These spores then infect nearby apple trees. Apple scab can also spread from nearby trees that are already infected. Frequent rains and prolonged leaf wetness enable severe scab infection conditions.")

            st.subheader("Treatment: ")
            treatmentInfo = "Rake up leaves and remove them from the orchard before May. Remove abandoned apple trees within 100 yards of your orchard. The University of Maine Cooperative Extension recommends applying preventive sprays such as captan, sulfur, or other fungicides."
            st.write("Rake up leaves and remove them from the orchard before May. Remove abandoned apple trees within 100 yards of your orchard. The University of Maine Cooperative Extension recommends applying preventive sprays such as captan, sulfur, or other fungicides.")

            st.subheader("Risk: ")
            riskInfo = "Apple scab rarely kills trees. Severe cases may cause complete defoliation by early summer. Repeated infections will weaken the tree, making it susceptible to other diseases."
            st.write("Apple scab rarely kills trees. Severe cases may cause complete defoliation by early summer. Repeated infections will weaken the tree, making it susceptible to other diseases.")

        elif result['status'] == ' has Rust ':
            st.subheader("Apple Trees Affected: ")
            treesInfo = "Golden Delicious is susceptible to cedar apple rust. Resistant varieties include Black Oxford, Enterprise, and William’s Pride."
            st.write("Golden Delicious is susceptible to cedar apple rust. Resistant varieties include Black Oxford, Enterprise, and William’s Pride.")

            st.subheader("Symptoms: ")
            symptomsInfo = "Yellow or orange spots approximately ¼-inch in diameter develop on the leaves. These colorful spots are vibrant on the upper surface of the leaf. As the spots age, block dots form in their center. Galls develop on branch tips in early spring. These galls begin to swell up to 2-inches in diameter and develop bright orange, jelly-like tubes."
            st.write("Yellow or orange spots approximately ¼-inch in diameter develop on the leaves. These colorful spots are vibrant on the upper surface of the leaf. As the spots age, block dots form in their center. Galls develop on branch tips in early spring. These galls begin to swell up to 2-inches in diameter and develop bright orange, jelly-like tubes.")

            st.subheader("Causes: ")
            causesInfo = "Cedar apple rust occurs when apple trees are grown in proximity to Eastern red cedar and other junipers. Apple trees, Eastern red cedar trees, and junipers spread the disease to each other. The fungus overwinters in infected branches and galls on red cedar and juniper trees. In spring, the galls produce the orange, gummy, fungal growth, which creates spores in wet conditions. These spores are then carried by the wind to apple trees up to one mile away. Over the summer, the infection begins to grow. As the apple tree leaves develop the bright red spots on their upper side, the undersides of the lesions develop small raised tubes that produce powdery, orange spores. These spores are then released in mid to late summer to infect juniper and red cedar trees."
            st.write("Cedar apple rust occurs when apple trees are grown in proximity to Eastern red cedar and other junipers. Apple trees, Eastern red cedar trees, and junipers spread the disease to each other. The fungus overwinters in infected branches and galls on red cedar and juniper trees. In spring, the galls produce the orange, gummy, fungal growth, which creates spores in wet conditions. These spores are then carried by the wind to apple trees up to one mile away. Over the summer, the infection begins to grow. As the apple tree leaves develop the bright red spots on their upper side, the undersides of the lesions develop small raised tubes that produce powdery, orange spores. These spores are then released in mid to late summer to infect juniper and red cedar trees.")

            st.subheader("Treatment: ")
            treatmentInfo = "To prevent the spread of disease, the UMaine Cooperative Extension recommends applying fungicides containing fenarimol or myclobutanil. Avoid planting the host trees near one another. Inspect nearby juniper and red cedar trees in late winter or early spring. Prune and remove galls before the orange, gummy structures form in the spring."
            st.write("To prevent the spread of disease, the UMaine Cooperative Extension recommends applying fungicides containing fenarimol or myclobutanil. Avoid planting the host trees near one another. Inspect nearby juniper and red cedar trees in late winter or early spring. Prune and remove galls before the orange, gummy structures form in the spring.")

            st.subheader("Risk: ")
            riskInfo = "Cedar apple rust should not cause severe damage to your tree’s health."
            st.write(
                "Cedar apple rust should not cause severe damage to your tree’s health.")

        elif result['status'] == ' has Multiple Diseases ':
            treesInfo = "Multiple trees can be affected"
            symptomsInfo = "Multiple symptoms"
            causesInfo = "Multiple Causes"
            treatmentInfo = "Multiple Treatments"
            riskInfo = "Multiple Risks"
            st.subheader(
                "The given leaf image has multiple diseases - fire blight, black rot and many more.")
        else:
            treesInfo = "No trees affected"
            symptomsInfo = "No symptoms"
            causesInfo = "No causes"
            treatmentInfo = "No treatments"
            riskInfo = "No risks"
            result = "Healthy"
            st.subheader("The given image is healthy!")


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


page_names_to_funcs = {
    "Classification": main_page,
    "Team": about_us,
}

selected_page = st.sidebar.selectbox(
    "Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
