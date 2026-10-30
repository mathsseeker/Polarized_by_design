import streamlit as st

st.title("Demo")

button_press = st.button("Say hi")

if button_press:
    st.text("Hello!")
