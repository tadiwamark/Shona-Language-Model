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
import streamlit as st
import time

# Flashing warning
for _ in range(1):
    st.markdown("<span style='color:red'>**DISCLAIMER:** This model is trained on Jehovah's Witness reading material and does not represent the entire Shona language.</span>", unsafe_allow_html=True)
    time.sleep(0.5)
    st.markdown("<span style='color:blue'>**DISCLAIMER:** This model is trained on Jehovah's Witness reading material and does not represent the entire Shona language.</span>", unsafe_allow_html=True)
    time.sleep(0.5)
    st.write("You can check out some of the reading material to prompt on [link](https://www.jw.org/sn/Raibhurari/magazini/)")

user_input = st.text_area("Type 5 words in Shona that might be found in Jehovha's witness reading and I'll try and predict the next word/s:")
if st.button("Predict"):
    predicted_words = predict_next_words(model, tokenizer, user_input, num_words=3)
    st.write(f"The next words might be: {predicted_words}")
