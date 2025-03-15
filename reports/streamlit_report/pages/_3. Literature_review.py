# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠")

st.title("Literature Review")
st.subheader("Summary")
st.markdown(
    """
The detection of COVID-19 in X-ray scans has been a critical area of research during the pandemic due to the rapid need for scalable diagnostic tools. Chest X-rays (CXR) are widely available and offer a non-invasive approach to detect pulmonary abnormalities caused by COVID-19, such as ground-glass opacities (GGOs) and consolidations. Early studies leveraged traditional machine learning models on handcrafted features like texture analysis using Gray Level Co-occurrence Matrix (GLCM) to differentiate between healthy and infected lungs.

However, this approach has its strengths and limitations that significantly impact its clinical utility and broader adoption.
"""
)
st.subheader("Strengths")
st.markdown(
    """
One of the primary strengths of using CXR scans for COVID-19 detection is their widespread availability, especially in low-resource settings where access to advanced imaging modalities like computed tomography (CT) is limited. Chest X-rays are inexpensive and portable, allowing their use even in rural or remote areas [(Wang et al., 2020)](https://www.sciencedirect.com/science/article/pii/S0889159120305110). Additionally, X-rays provide rapid imaging results, enabling timely diagnosis and intervention during critical phases of the disease.

Machine learning and deep learning models have significantly enhanced the potential of CXR-based COVID-19 detection. These methods can extract and analyze subtle patterns in lung textures, such as ground-glass opacities (GGOs) and consolidations, which may not be easily noticeable to the human eye [(Apostolopoulos & Mpesiana, 2020)](https://pubmed.ncbi.nlm.nih.gov/32524445/). For instance, convolutional neural networks (CNNs) such as ResNet-50, VGG-16, and COVID-Net have demonstrated high sensitivity and specificity in detecting COVID-19, often exceeding 90% accuracy in some studies ([Wang et al., 2020](https://www.sciencedirect.com/science/article/pii/S0889159120305110); [Ozturk et al., 2020](https://pubmed.ncbi.nlm.nih.gov/32568675/)). Moreover, these models, when combined with interpretability techniques like Grad-CAM, provide heatmaps to highlight infected regions, helping radiologists validate and trust AI predictions [(Selvan et al., 2021)](https://arxiv.org/abs/2109.07138).

Data augmentation and transfer learning have addressed some challenges associated with limited COVID-19 datasets. Transfer learning enables the adaptation of pre-trained models, such as ImageNet-based architectures, to small COVID-19 datasets, reducing the dependency on large labeled data [(Apostolopoulos & Mpesiana, 2020)](https://pubmed.ncbi.nlm.nih.gov/32524445/). Synthetic data generation using techniques like generative adversarial networks (GANs) has further improved model performance by addressing class imbalance.
"""
)
st.subheader("Limitations")
st.markdown(
    """
Despite its promise, CXR-based COVID-19 detection has several limitations. Firstly, chest X-rays are inherently less sensitive compared to CT scans in identifying COVID-19-specific abnormalities, as X-rays may not detect mild cases or early-stage infections [(Kanne et al., 2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7233379/). This limitation can result in false negatives, particularly in asymptomatic or mildly symptomatic patients.

Another significant challenge lies in distinguishing COVID-19 pneumonia from other types of viral or bacterial pneumonia, as their radiological features often overlap. This diagnostic ambiguity can lead to false positives, undermining the specificity of AI models [(Li et al., 2020)](https://www.ajronline.org/doi/10.2214/AJR.20.22954). Additionally, many studies rely on publicly available datasets, which are often imbalanced, small in size, and lack demographic diversity. This limits the generalizability of models to diverse populations and imaging protocols.

Explainability and interpretability also remain critical challenges. While methods like Grad-CAM provide insights into model decision-making, there is still a lack of consensus on how to ensure that AI models focus exclusively on clinically relevant features without overfitting to spurious correlations [(Selvan et al., 2021)](https://arxiv.org/abs/2109.07138). Moreover, real-world deployment of these models requires rigorous validation on clinically curated datasets, which is often lacking in academic studies.

Lastly, ethical concerns such as data privacy, informed consent, and biases in training datasets must be addressed to ensure equitable and fair deployment. Models trained on datasets from specific regions or populations may fail to generalize to other contexts, exacerbating health disparities.
"""
)
st.subheader("Conclusion")
st.markdown(
    """
While CXR-based COVID-19 detection offers an accessible and cost-effective solution, its limitations in sensitivity, specificity, and generalizability highlight the need for cautious implementation. Collaborative efforts, including larger datasets, robust validation, and integration with clinical workflows, are essential to harness the full potential of this technology.
"""
)
