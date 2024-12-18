from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from scipy.special import softmax

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Define the sentiment_analysis function
def sentiment_analysis(text, tokenizer, model):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)
    labels = ['Negative', 'Positive']
    scores = {l: float(s) for (l, s) in zip(labels, scores_)}
    return scores

# Define the map_sentiment_score_to_rating function
def map_sentiment_score_to_rating(score):
    min_score = 0.0
    max_score = 1.0
    min_rating = 1
    max_rating = 10
    rating = ((score - min_score) / (max_score - min_score)) * (max_rating - min_rating) + min_rating
    return rating
    