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
def predict_next_words(model, text, num_to_predict):
    for _ in range(num_to_predict):
        token_list = tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=5, padding='pre')
        probabilities = model.predict(token_list, verbose=0)
        predicted = np.argmax(probabilities, axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        text += " " + output_word
    return text

# Streamlit UI
st.title("Shona Language Model")
user_input = st.text_area("Type 5 words in Shona:")
if st.button("Predict"):
    predicted_words = predict_next_words(model, tokenizer, user_input, num_words=3)
    st.write(f"The next words might be: {predicted_words}")
