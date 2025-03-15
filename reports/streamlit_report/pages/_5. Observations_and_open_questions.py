# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Observations and Open Questions", page_icon="🦠")

st.title("Observations and Open Questions")

st.markdown(
    """
Based on our analysis of the dataset, we have compiled a list of key observations and open questions that could significantly inform the modeling phase. These insights will play a crucial role in selecting the appropriate model and setting realistic expectations for its accuracy and precision:
"""
)
st.subheader("Observations")
st.markdown(
    """
1.	The mask images consistently depict the same orientation (patient facing forward with their right hand on the left side of the image).
2.	In contrast, the chest X-ray images exhibit variability in orientation and quality, including rotated or flipped views, patients facing the opposite direction, and low-resolution images.
3.	Oversampling or undersampling techniques are necessary to address the class imbalance in the dataset effectively.
4.	The GLCM analysis reveals systematic differences between the classes, such as higher homogeneity in Lung Opacity and higher entropy in Normal images. These features could be crucial for model interpretation and class separation.
5.	Homogeneity and energy suggest that certain texture differences are confined to the lung regions. The analysis of inversely masked images shows that these features remain consistent and are likely not influenced by artifacts or noise.
"""
)
st.subheader("Open questions")
st.markdown(
    """
1.	How does normalization or standardization help in the image preprocessing step before training the model?
2.	Is it necessary to align all images with their corresponding masks?
3.	Should new masks be generated based on the chest X-ray images?
4.	What to do with image features like medical devices etc. shadowing the lung area? (to cut them out->black, do fill with mean value of one of the calculated metrics like contrast etc.)
5.	Does/will the orientation of an image (tilt, flips, rotations) affect the model prediction quality?
6.	What methods could be applied for oversampling?
7.	How to handle low resolution images?
8.	How to handle low contrast images?
9.	Should a model be trained directly on the masked images to utilize the complete image information, or is it more effective to develop a model based on the significant extracted GLCM features?
10.	How stable are the extracted GLCM features with variations in image quality? What strategies could be applied to minimize such influences?
11.	Would it be beneficial to use the GLCM features as additional inputs for a model trained on the masked images? Could a hybrid solution improve the separability of the classes?
"""
)
