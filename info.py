import streamlit as st

def info_page():
    st.title("How to Use the App")
    st.write("""
        This app helps you upload chest x-rays and classify between `NORMAL` and `PNEUMONIA` using a pre-trained CNN model.
        
        **Steps**:
        1. Navigate to the **Predict** page via the sidebar.
        2. Upload an image using the **Browse files** button.
        3. The model will predict if the x-ray shows pneumonia and display the result.
    """)

if __name__ == '__page__':
    info_page()