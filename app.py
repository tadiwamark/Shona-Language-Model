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
import pickle


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the model
model = load_model("best_model2.h5")

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
    for _ in range(num_words):
        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=5, padding='pre')
        
        # Predict the next word
        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)
        
        # Convert the predicted word index to a word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        # Append the predicted word to the text
        text += " " + output_word

    return ' '.join(text.split(' ')[-num_words:])

# Streamlit UI
st.title("Shona Language Model")
user_input = st.text_area("Type 5 words in Shona:")
if st.button("Predict"):
    predicted_words = predict_next_words(model, tokenizer, user_input, num_words=3)
    st.write(f"The next words might be: {predicted_words}")
