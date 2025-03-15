# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Covid-19 Detection", page_icon="🦠")

st.title("Analysis of Covid-19 🤕 chest x-rays")

st.info("This is a purely informational message", icon="ℹ️")
st.code("print('Hello, Streamlit!')")
st.subheader("Contributors")
st.write("This project was developed by the following contributors who attended Aug24 CDS class:")
st.markdown(
    """
* Maja
* Hanna
* Valerian
* Ahmad
"""
)

st.latex(
    r"""
a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
\sum_{k=0}^{n-1} ar^k =
a \left(\frac{1-r^{n}}{1-r}\right)
"""
)
