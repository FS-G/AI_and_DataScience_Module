import streamlit as st
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit app title
st.title("Part-of-Speech Tagging with SpaCy")

# Input text box
text_input = st.text_area("Enter your text here:", height=200)

# Process the input text
if st.button("Analyze"):
    if text_input:
        doc = nlp(text_input)

        # Display the input text
        st.subheader("Input Text:")
        st.write(text_input)

        # Display tokens and their POS tags
        st.subheader("Tokens and Part-of-Speech Tags:")
        pos_data = [(token.text, token.pos_) for token in doc]
        for token, pos in pos_data:
            st.write(f"{token} : {pos}")

        # # Display dependency parsing
        # st.subheader("Dependency Parsing:")
        # for token in doc:
        #     st.write(f"{token.text} (Head: {token.head.text})")

    else:
        st.warning("Please enter some text to analyze.")
