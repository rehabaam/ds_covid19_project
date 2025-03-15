# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Introduction", page_icon="🦠")
st.title("Introduction")
st.subheader("Problem statement")
st.markdown(
    """
The rapid detection of COVID-19 in patients is critical for effective treatment and containment.
Chest X-ray scans are widely available and can serve as a useful tool for identifying the presence of COVID-19-related lung abnormalities.
However, manually analyzing large volumes of X-ray images is time-consuming and requires expert knowledge.
There is a need for an efficient and automated solution to accurately detect COVID-19 in chest X-ray images to assist healthcare professionals in making timely diagnosis and treatment decisions.
"""
)
st.subheader("Objective")
st.markdown(
    """
1.	Data Exploration and Preprocessing:
    1.	Use the dataset provided for this project and preprocess a labeled dataset of chest X-ray images to train and evaluate the machine learning model.
    2.	Perform data augmentation and normalization to improve the model’s generalization ability.
2.	Model Development:
    1.	Design and implement a supervised machine learning model capable of accurately detecting COVID-19 from chest X-ray images.
    2.	Experiment with various deep learning techniques, such as convolutional neural networks (CNNs), to improve model performance.
3.	Model Evaluation and Optimization:
    1.	Evaluate the model’s performance using appropriate metrics, such as accuracy, precision, recall, and F1-score.
    2.	Optimize the model by fine-tuning hyperparameters and employing techniques like cross-validation and regularization to minimize overfitting.
"""
)
