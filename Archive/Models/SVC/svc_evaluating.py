import joblib
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm  # Progress bar
import numpy as np

# Load the dataset
data = pd.read_csv('./datasets/IMDB Dataset.csv', encoding='latin1')

# Assume the dataset has 'review' as features and 'sentiment' as labels
X = data['review']
y = data['sentiment']

y = y.map({'negative': 0, 'positive': 1})

# Preprocess the text data (example using TF-IDF)
tfidf = TfidfVectorizer(max_features=1000)

# Adding a progress bar to the fit_transform process
print("Transforming text data into TF-IDF features...")
X_tfidf = tfidf.fit_transform(tqdm(X))

# Load the saved SVC model
svc_model = joblib.load('./saved_models/svc_sentiment_model.pkl')

# Adding a progress bar to the prediction process
print("Making predictions on the dataset...")
y_pred = np.array([svc_model.predict(X_tfidf[i]) for i in tqdm(range(X_tfidf.shape[0]))])

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")