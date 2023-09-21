# Shona Language Model Project

## Overview
This project encompasses the development of a language model for the Shona language, focusing predominantly on Jehovah's Witness reading material. The goal is to develop a model capable of understanding and predicting subsequent words in a given sequence of Shona words, utilizing advanced NLP and machine learning techniques.

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Text Extraction](#text-extraction)
  - [Text Preprocessing](#text-preprocessing)
  - [Word Embeddings](#word-embeddings)
  - [Model Development](#model-development)
  - [Training & Testing](#training--testing)
- [Limitations](#limitations)
- [Solutions and Workarounds](#solutions-and-workarounds)
- [Disclaimer](#disclaimer)
- [Conclusion](#conclusion)
- [Further Development](#further-development)

## Objective
The primary objective is to construct a proficient model for comprehending and generating text in the Shona language, harnessing modern natural language processing and machine learning methodologies.

## Dataset
The dataset used is primarily composed of Shona language reading materials from Jehovah's Witness literature.

## Methodology
### Text Extraction
The PyPDF2 library is used for extracting text from PDF files containing the Shona language literary material. The `extract_text_from_pdf` function reads and extracts text from each page of the PDF.

```python
import PyPDF2
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
    return text
```

### Text Preprocessing
The text data is tokenized using the Tokenizer class from the keras.preprocessing.text, converting the text into a sequence of tokens (words).
```python
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([shona_text])
```

### Word Embeddings
Word embeddings are generated using Word2Vec from gensim.models, allowing the conversion of words into numerical vectors, essential for the model to understand the relationships and similarities between different words.
```python
from gensim.models import Word2Vec
model_gensim = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
```

### Model Development
Two Recurrent Neural Network (RNN) models are developed using LSTM layers; one with its own embedding layer, and the other with pre-trained embeddings from the Word2Vec model. These models are implemented using Keras.
```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
```

### Training & Testing
The models are trained with a training dataset, and their performance is validated using a validation dataset. The model with the lower validation loss is saved as the best model.
```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Limitations
Memory Constraints: The model training phase experienced crashes due to memory constraints even with substantial memory (51 GB).
Dataset Context Limitation: The model is trained exclusively on Jehovah's Witness reading material, limiting its representation of the Shona language.

## Solutions and Workarounds
Data Chunking: The dataset is split and loaded in smaller batches during training to manage memory usage effectively.
Model Optimization: Intermediate results and model weights were saved to avoid recomputation.
Disclaimer
This model is contextually limited and does not represent the entirety of the Shona language. Users should acknowledge the model’s limitations when considering its predictions and capabilities.

## Conclusion
Despite its limitations and challenges, this project is a foundational step towards modeling underrepresented languages, demonstrating potential in understanding and developing NLP models for the Shona language.

## Further Development
Expanding the Dataset: To include a diverse range of Shona texts to improve the model’s comprehension.
Hyperparameter Tuning: To optimize the model's predictive accuracy.
Deployment: Implementing this model into real-world applications for practical assessment and utilization.
