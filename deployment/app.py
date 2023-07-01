import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Page: ', ('EDA','Predict'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()