# ==========================================================
# 🌿 AI PLANT DISEASE DETECTION SYSTEM
# Developed by Md Salman Ali
# ==========================================================

# ==========================
# IMPORTS
# ==========================

import os
import json
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px

from PIL import Image

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from disease_info import disease_information


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD CSS
# ==========================================================

if os.path.exists("style.css"):

    with open("style.css") as css:

        st.markdown(

            f"<style>{css.read()}</style>",

            unsafe_allow_html=True

        )


# ==========================================================
# CACHE MODEL
# ==========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "plant_disease_model.keras"
    )

model = load_model()


# ==========================================================
# LOAD CLASS NAMES
# ==========================================================

with open("class_names.json","r") as f:

    class_names = json.load(f)

if isinstance(class_names,dict):

    class_names=list(class_names.values())


# ==========================================================
# CONSTANTS
# ==========================================================

IMAGE_SIZE = (224,224)


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(image):

    image=image.convert("RGB")

    image=image.resize(IMAGE_SIZE)

    image=np.array(image)

    image=image.astype("float32")/255.0

    image=np.expand_dims(image,axis=0)

    return image


# ==========================================================
# PREDICT FUNCTION
# ==========================================================

def predict_disease(image):

    processed=preprocess_image(image)

    prediction=model.predict(processed,verbose=0)[0]

    predicted_index=np.argmax(prediction)

    predicted_class=class_names[predicted_index]

    confidence=float(prediction[predicted_index])

    return predicted_class,confidence,prediction

# ==========================================================
# PDF REPORT GENERATOR
# ==========================================================

styles = getSampleStyleSheet()

def create_pdf(predicted_class, confidence, info):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    story = []

    title = Paragraph(
        "<b><font size=20 color='green'>AI Plant Disease Diagnostic Report</font></b>",
        styles["Title"]
    )

    story.append(title)
    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now().strftime('%d %B %Y %I:%M %p')}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1,15))

    disease = predicted_class.replace("___"," : ").replace("_"," ")

    table = Table([

        ["Disease", disease],

        ["Confidence", f"{confidence*100:.2f}%"],

        ["AI Model", "TensorFlow CNN"],

        ["Status",
         "Healthy" if "healthy" in predicted_class.lower()
         else "Disease Detected"]

    ])

    table.setStyle(TableStyle([

        ("GRID",(0,0),(-1,-1),1,colors.grey),

        ("BACKGROUND",(0,0),(0,-1),colors.darkgreen),

        ("TEXTCOLOR",(0,0),(0,-1),colors.white),

        ("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold"),

        ("BOTTOMPADDING",(0,0),(-1,-1),8)

    ]))

    story.append(table)

    story.append(Spacer(1,20))

    story.append(
        Paragraph("<b>Description</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(info["description"],styles["BodyText"])
    )

    story.append(Spacer(1,15))

    story.append(
        Paragraph("<b>Symptoms</b>",styles["Heading2"])
    )

    for item in info["symptoms"]:
        story.append(
            Paragraph("• "+item,styles["BodyText"])
        )

    story.append(Spacer(1,10))

    story.append(
        Paragraph("<b>Treatment</b>",styles["Heading2"])
    )

    for item in info["treatment"]:
        story.append(
            Paragraph("• "+item,styles["BodyText"])
        )

    story.append(Spacer(1,10))

    story.append(
        Paragraph("<b>Prevention</b>",styles["Heading2"])
    )

    for item in info["prevention"]:
        story.append(
            Paragraph("• "+item,styles["BodyText"])
        )

    story.append(Spacer(1,25))

    story.append(
        Paragraph(
            "<b>Generated using AI Plant Disease Detection System</b>",
            styles["Italic"]
        )
    )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf

# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<h1 style="
text-align:center;
font-size:52px;
font-weight:900;
letter-spacing:-1px;
">
🌿 AI Plant Disease Detection System
</h1>

<h3 style="text-align:center;color:#94A3B8;">
Deep Learning Powered by TensorFlow CNN
</h3>

<p style="
text-align:center;
max-width:850px;
margin:auto;
font-size:18px;
line-height:1.8;
color:#D1D5DB;
">

Upload a plant leaf image and receive AI-powered disease detection,
confidence score, symptoms, treatment recommendations,
and prevention methods.

</p>
""", unsafe_allow_html=True)

# ==========================================================
# DASHBOARD
# ==========================================================

c1,c2,c3,c4=st.columns(4)

with c1:

    st.metric(

        "Model Accuracy",

        "97.85%"

    )

with c2:

    st.metric(

        "Disease Classes",

        len(class_names)

    )

with c3:

    st.metric(

        "Dataset",

        "54,305"

    )

with c4:

    st.metric(

        "Framework",

        "TensorFlow"

    )


st.markdown("<br>",unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown(
    """
    <div class="sidebar-logo">
    """,
    unsafe_allow_html=True
    )

    st.image(
        "assets/logo.png",
        width=280
    )

    st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True
    )

    st.markdown("## 🌿 AI Plant Doctor")

    st.caption("Deep Learning Powered by TensorFlow")

    st.success("✅ Model Loaded Successfully")

    st.divider()

    st.subheader("📊 Model Information")

    st.metric("Architecture", "CNN")
    st.metric("Accuracy", "97.85%")
    st.metric("Disease Classes", len(class_names))
    st.metric("Image Size", "224 × 224")

    st.divider()

    st.subheader("🌱 Supported Crops")

    st.markdown("""
🍎 Apple

🍒 Cherry

🌽 Corn

🍇 Grape

🍊 Orange

🍑 Peach

🫑 Pepper

🥔 Potato

🍓 Strawberry

🍅 Tomato
""")

    st.divider()

    st.subheader("👨‍💻 Developer")

    st.write("**Md Salman Ali**")

    st.caption("B.Tech CSE (Data Science)")

    st.caption("Jamia Millia Islamia")

# ==========================================================
# MAIN SECTION
# ==========================================================

left, right = st.columns([1.4, 1])

# ==========================================================
# LEFT PANEL
# ==========================================================

with left:

    st.markdown("## 📤 Upload Plant Leaf")

    uploaded_file = st.file_uploader(

        "Choose a plant leaf image",

        type=["jpg", "jpeg", "png"]

    )

    image = None

    predict_button = False

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(

            image,

            caption="Uploaded Leaf",

            use_container_width=True

        )

        predict_button = st.button(

            "🔍 Detect Disease",

            type="primary",

            use_container_width=True

        )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    st.image(

        "assets/doctor.png",

        use_container_width=True

    )

    st.markdown("## 🤖 AI Plant Expert")

    st.write("""

The AI model can identify **38 plant diseases**
using a Convolutional Neural Network trained on
the PlantVillage dataset.

""")

    st.success("✔ AI Powered Diagnosis")

    st.success("✔ Confidence Score")

    st.success("✔ Disease Information")

    st.success("✔ Symptoms")

    st.success("✔ Treatment")

    st.success("✔ Prevention")

    st.success("✔ PDF Report")

st.markdown("---")

# ==========================================================
# INFORMATION CARDS
# ==========================================================

# ======================================================
# GLASS INFORMATION CARDS
# ======================================================


c1, c2, c3 = st.columns(3)


with c1:

    st.markdown("""

    <div class="glass-card green-card">

    <h2>🌿 Disease Detection</h2>

    <p>
    Upload a clear leaf image and let AI
    identify plant diseases within seconds.
    </p>

    <span>✔ CNN Based Prediction</span><br>
    <span>✔ 38 Disease Classes</span><br>
    <span>✔ Confidence Score</span>

    </div>

    """, unsafe_allow_html=True)




with c2:

    st.markdown("""

    <div class="glass-card yellow-card">

    <h2>💊 Treatment</h2>

    <p>
    Receive AI-generated treatment,
    symptoms and prevention methods.
    </p>

    <span>✔ Disease Information</span><br>
    <span>✔ Treatment Guidance</span><br>
    <span>✔ Prevention Tips</span>

    </div>

    """, unsafe_allow_html=True)




with c3:

    st.markdown("""

    <div class="glass-card blue-card">

    <h2>📄 AI Report</h2>

    <p>
    Download a professional PDF diagnosis
    report after prediction.
    </p>

    <span>✔ Disease Result</span><br>
    <span>✔ Confidence Score</span><br>
    <span>✔ Complete Analysis</span>

    </div>

    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# AI PREDICTION
# ==========================================================

if uploaded_file is not None and predict_button:

    with st.spinner("🧠 AI is analyzing the leaf image..."):

        predicted_class, confidence, predictions = predict_disease(image)

        disease_name = (
            predicted_class
            .replace("___", " : ")
            .replace("_", " ")
        )

        info = disease_information.get(
            predicted_class,
            {
                "description": "Information not available.",
                "symptoms": [],
                "treatment": [],
                "prevention": []
            }
        )

    st.success("✅ Analysis Completed Successfully")

    st.markdown("---")

    # ======================================================
    # RESULT HEADER
    # ======================================================

    left_result, right_result = st.columns([2, 1])

    with left_result:

        st.markdown(f"""
        <div class="result-card">

        <h2>🩺 Disease Detected</h2>

        <h1>{disease_name}</h1>

        <h3>Confidence Score</h3>

        <h2>{confidence*100:.2f}%</h2>

        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)

    with right_result:

        st.metric(
            "Prediction",
            disease_name
        )

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.metric(
            "Model",
            "TensorFlow CNN"
        )

        st.metric(
            "Status",
            "Healthy" if "healthy" in predicted_class.lower() else "Disease Detected"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # TOP 5 PREDICTIONS
    # ======================================================

    st.subheader("📊 Top 5 Prediction Confidence")

    top5 = np.argsort(predictions)[::-1][:5]

    chart_df = pd.DataFrame({

        "Disease":[

            class_names[i].replace("___"," : ").replace("_"," ")

            for i in top5

        ],

        "Confidence":[

            float(predictions[i])*100

            for i in top5

        ]

    })

    fig = px.bar(

        chart_df,

        x="Confidence",

        y="Disease",

        orientation="h",

        text="Confidence",

        color="Confidence",

        color_continuous_scale="Greens"

    )

    fig.update_layout(

        template="plotly_dark",

        height=500,

        xaxis_title="Confidence (%)",

        yaxis_title="",

        title="AI Prediction Confidence",

        coloraxis_showscale=False

    )

    fig.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.markdown("---")

    # ======================================================
    # DISEASE DESCRIPTION
    # ======================================================

    st.subheader("📖 Disease Description")

    st.info(

        info.get(

            "description",

            "Description not available."

        )

    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # SYMPTOMS • TREATMENT • PREVENTION
    # ======================================================

    col1, col2, col3 = st.columns(3)

    # ---------------- Symptoms ----------------

    with col1:

        st.markdown("## 🍂 Symptoms")

        symptoms = info.get("symptoms", [])

        if symptoms:

            for symptom in symptoms:

                st.success(f"✔ {symptom}")

        else:

            st.write("No symptoms available.")

    # ---------------- Treatment ----------------

    with col2:

        st.markdown("## 💊 Treatment")

        treatment = info.get("treatment", [])

        if treatment:

            for item in treatment:

                st.info(f"✔ {item}")

        else:

            st.write("No treatment information available.")

    # ---------------- Prevention ----------------

    with col3:

        st.markdown("## 🌱 Prevention")

        prevention = info.get("prevention", [])

        if prevention:

            for item in prevention:

                st.warning(f"✔ {item}")

        else:

            st.write("No prevention information available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ======================================================
    # MODEL INFORMATION
    # ======================================================

    st.subheader("🤖 AI Model Information")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Architecture", "CNN")

    with m2:
        st.metric("Input Size", "224 × 224")

    with m3:
        st.metric("Disease Classes", len(class_names))

    with m4:
        st.metric("Framework", "TensorFlow")

    st.markdown("---")


    # ======================================================
    # DOWNLOAD PDF REPORT
    # ======================================================

    pdf = create_pdf(

        predicted_class,

        confidence,

        info

    )

    st.download_button(

        "📄 Download Diagnostic Report",

        data=pdf,

        file_name="Plant_Disease_Report.pdf",

        mime="application/pdf",

        use_container_width=True

    )
st.markdown("---")

st.markdown("""

<div class="footer">

<h2>🌿 AI Plant Disease Detection System</h2>

<p>

Deep Learning Powered by TensorFlow · Keras · Streamlit

</p>

<br>

<b>Developed by</b>

<h3>Md Salman Ali</h3>

<p>

B.Tech Computer Science & Engineering (Data Science)

<br>

Jamia Millia Islamia

</p>

</div>

""",unsafe_allow_html=True)