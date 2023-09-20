# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 12:04:22 2023

@author: tadiw
"""


from tensorflow.keras.models import load_model
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

# Initialize the tokenizer
tokenizer = Tokenizer()

# Load the model
model = load_model("best_model1.h5")

# Create tokenizer function
def predict_next_words(model, tokenizer, text, num_words=1):
    """
    Predict the next set of words using the trained model.
    
    Args:
    - model (keras.Model): The trained model.
    - tokenizer (Tokenizer): The tokenizer object used for preprocessing.
    - text (str): The input text.
    - num_words (int): The number of words to predict.

    Returns:
    - str: The predicted words.
    """
    st.write("Inside function...")  # Debug print
    for _ in range(num_words):
        st.write("In the loop...")  # Debug print

        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])[0]
        st.write(f"Sequence: {sequence}")  # Debug print

        sequence = pad_sequences([sequence], maxlen=5, padding='pre')
        st.write(f"Padded sequence: {sequence}")  # Debug print
        
        # Predict the next word
        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)
        st.write(f"Predicted index: {predicted}")  # Debug print
        
        # Convert the predicted word index to a word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        
        st.write(f"Output word: {output_word}")  # Debug print

        # Append the predicted word to the text
        text += " " + output_word

    return ' '.join(text.split(' ')[-num_words:])

# Streamlit UI
st.title("Shona Language Model")
user_input = st.text_area("Type 5 words in Shona:")
if st.button("Predict"):
    predicted_words = predict_next_words(model, tokenizer, user_input, num_words=3)
    st.write(f"The next words might be: {predicted_words}")
