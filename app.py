import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import os

# --- Configuration & Constants ---
MODEL_PATH = "iris_model.pkl"
MAP_PATH = "species_map.pkl"

IMAGE_LINKS = {
    "Setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg",
    "Versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
}

DESCRIPTIONS = {
    "Setosa": "Iris Setosa typically has smaller, distinctly separate petals and sepals, often found in colder regions. It is known for its bluish-purple flowers.",
    "Versicolor": "Iris Versicolor, or Blue Flag, features violet-blue flowers with petals showing prominent veining. It thrives in wetlands and marshes.",
    "Virginica": "Iris Virginica, or Southern Blue Flag, has large, showy flowers ranging from violet-blue to purple. It prefers moist environments like swamps and riverbanks.",
}

ABOUT_TEXT = """
This application uses a machine learning model (trained on the classic Iris dataset) to predict the species of an Iris flower based on its sepal and petal measurements.

**The Iris Dataset:**
*   Contains 150 samples from three species of Iris flowers: Setosa, Versicolor, and Virginica.
*   Features measured: Sepal Length, Sepal Width, Petal Length, Petal Width (in cm).
*   Widely used in machine learning for classification tasks.

**Model:**
*   The specific model used here is loaded from the `iris_model.pkl` file.
"""

# --- Helper Functions ---
def load_model_and_map(model_path, map_path):
    """Loads the pickled model and species map."""
    if not os.path.exists(model_path) or not os.path.exists(map_path):
        st.error(f"Error: Model file (\"{model_path}\") or species map file (\"{map_path}\") not found. Please ensure they are in the same directory as the script.")
        return None, None, None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(map_path, "rb") as f:
            species_map = pickle.load(f)
        reverse_map = {v: k for k, v in species_map.items()}
        return model, species_map, reverse_map
    except Exception as e:
        st.error(f"Error loading model or map files: {e}")
        return None, None, None

def predict_species(input_data, model, reverse_map):
    """Makes predictions using the loaded model."""
    try:
        pred_label = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        species_name = reverse_map[pred_label].capitalize()
        return species_name, pred_proba
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def create_confidence_chart(probabilities, labels):
    """Creates an interactive Plotly bar chart for confidence scores."""
    fig = go.Figure(
        go.Bar(
            x=probabilities,
            y=labels,
            orientation="h",
            marker_color=["#FF8C94", "#8EC6C5", "#D8BFD8"], # Example colors
            text=[f"{p:.2f}" for p in probabilities],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Model Confidence Scores",
        xaxis_title="Probability",
        yaxis_title="Species",
        yaxis=dict(autorange="reversed"), # Display Setosa at the top if it's first
        xaxis=dict(range=[0, 1]),
        height=250,
        margin=dict(l=10, r=10, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)", # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(textfont_size=12)
    return fig

# --- Streamlit App Layout ---
st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

# --- Load Model --- (Do this early to check for files)
model, species_map, reverse_map = load_model_and_map(MODEL_PATH, MAP_PATH)

# --- Sidebar for Inputs ---
st.sidebar.header("Input Flower Measurements")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.5, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

predict_button = st.sidebar.button("🔍 Predict Species", type="primary", disabled=(model is None)) # Disable if model failed to load

# --- Main Area ---
st.title("🌸 Iris Species Classifier")
st.markdown("Input the flower measurements using the sliders in the sidebar to predict the Iris species. The model will provide a prediction and confidence scores.")
st.divider()

# --- Prediction Output Area (Conditional) ---
if predict_button and model:
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    species_name, probabilities = predict_species(input_data, model, reverse_map)

    if species_name and probabilities is not None:
        results_container = st.container()
        with results_container:
            st.subheader("Prediction Result")
            st.metric(label="Predicted Species", value=species_name)
            st.divider()

        chart_container = st.container()
        with chart_container:
            st.subheader("Prediction Confidence")
            labels = [reverse_map[i].capitalize() for i in range(len(probabilities))]
            fig = create_confidence_chart(probabilities, labels)
            st.plotly_chart(fig, use_container_width=True)
            st.divider()

        info_container = st.container()
        with info_container:
            st.subheader(f"About Iris {species_name}")
            col1, col2 = st.columns([1, 2])
            with col1:
                if species_name in IMAGE_LINKS:
                    st.image(IMAGE_LINKS[species_name], caption=f"Iris {species_name}", use_container_width=True)
                else:
                    st.warning("Image not available for this species.")
            with col2:
                if species_name in DESCRIPTIONS:
                    st.info(DESCRIPTIONS[species_name])
                else:
                    st.warning("Description not available for this species.")

# --- About Section ---
st.divider()
with st.expander("ℹ️ About the Iris Dataset & Model"):
    st.markdown(ABOUT_TEXT)

# --- Footer (Optional) ---
st.markdown("<div style=\"text-align: center; color: grey; font-size: small; margin-top: 2em;\">By Shiva Prasad Naroju</div>", unsafe_allow_html=True)
