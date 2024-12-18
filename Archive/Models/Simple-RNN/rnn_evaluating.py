import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('./datasets/IMDB Dataset.csv', encoding='latin1')
X = data['review']
y = data['sentiment']

y = y.map({'negative': 0, 'positive': 1})

# Preprocess the data (tokenizing and padding)
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)

# Adding a progress bar for tokenizing text
print("Tokenizing the text data...")
X_tokenized = [tokenizer.texts_to_sequences([x])[0] for x in tqdm(X)]

# Padding the sequences
X_padded = pad_sequences(X_tokenized, maxlen=200)

# Load the saved Simple-RNN model
print("Loading the Simple-RNN model...")
rnn_model = load_model('./saved_models/rnn_sentiment_model.h5')

# Adding a progress bar to the prediction process
print("Making predictions with the Simple-RNN model...")
y_pred_prob = np.array([rnn_model.predict(np.expand_dims(X_padded[i], axis=0)) for i in tqdm(range(X_padded.shape[0]))])
y_pred = np.argmax(y_pred_prob, axis=1)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
