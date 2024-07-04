import streamlit as st
import main  # Import the entire main module

# Streamlit app
st.title("RetailX AI Assistant")
st.write("Ask a question about RetailX customers, products, and sales:")

question = st.text_input("Question")
if st.button("Submit"):
    if question:
        inputs = {"question": question}
        result = main.app.invoke(inputs)  # Use main.app instead of just app
        st.write("Answer:", result['answer'])
    else:
        st.write("Please enter a question.")