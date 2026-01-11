import streamlit as st
import spacy
from spacy import displacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit app title
st.title("Entity Recognition with SpaCy")

# Input text box
text_input = st.text_area("Enter your text here:", height=200)

# Process the input text
if st.button("Analyze"):
    if text_input:
        doc = nlp(text_input)

        # Display the input text
        st.subheader("Input Text:")
        st.write(text_input)

        # Display recognized entities
        st.subheader("Recognized Entities:")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        for entity, label in entities:
            st.write(f"{entity} : {label}")

        # Visualize entities
        st.subheader("Entity Visualization:")
        html = displacy.render(doc, style="ent", jupyter=False)
        st.components.v1.html(html, height=500)

    else:
        st.warning("Please enter some text to analyze.")
