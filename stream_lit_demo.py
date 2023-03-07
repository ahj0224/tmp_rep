import pandas as pd
import streamlit as st
from abb.tapas import update

st.title('Table QA(beta)')
query = st.text_input(label='질문을 입력하세요')
# st.write('The current movie title is', title)
uploaded_file = st.file_uploader("Choose a xlsx file")

if uploaded_file:
    converter = 0
    dataframe = pd.read_excel(uploaded_file)

    if query:
        query_results = update(query, dataframe)
        st.subheader('정답 : '+query_results[0])
        converter = 1

    if converter == 0:
        st.dataframe(dataframe)

    elif converter == 1:
        st.dataframe(query_results[1])
    


    