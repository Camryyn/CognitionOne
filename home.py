# import streamlit as st
# from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
# from functions import preprocess, sentiment_analysis, map_sentiment_score_to_rating


# def render_home(model, tokenizer):
#     st.title("Movie Sentiment Analysis")
#     st.write("Movie Sentiment Analysis Project for CS 124 Honors Fall 2024 at the University of Illinois- Urbana Champaign")
#     st.image("Assets/movie_review.png", caption="", use_column_width=True)
    
#     comments = []
    
    
#     input_text = st.text_area("Write your movie review here...")

#     # Displays sentiment
#     if st.button("Analyze Review"):
#         if input_text:
#             # Perform sentiment analysis using the model
#             scores = sentiment_analysis(input_text, tokenizer, model)

#             # Display sentiment scores
#             st.text("Sentiment Scores:")
#             for label, score in scores.items():
#                 st.text(f"{label}: {score:.2f}")

#             sentiment_label = max(scores, key=scores.get)

#             # Sentiment mapping
#             sentiment_mapping = {
#                 "Negative": "Negative",
#                 "Positive": "Positive"
#             }
#             sentiment_readable = sentiment_mapping.get(sentiment_label)

#             # Display the sentiment label
#             st.text(f"Sentiment: {sentiment_readable}")

            
#             rating = map_sentiment_score_to_rating(scores[sentiment_label])
#             rating = int(rating)

#             st.text(f"Rating: {rating}")            

#     # st.button to clear the text that was inputted
#     if st.button("Clear Input"):
#         input_text = ""

import streamlit as st
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from CognitionOne.functions import preprocess, sentiment_analysis, map_sentiment_score_to_rating

# Navigation Menu Options
def main():
    st.sidebar.title("🎥 Movie Themed Website")
    menu = ["Home", "Analyze Movie Review", "Top Reviews"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load model and tokenizer (to avoid reloading each time)
    model_name = "bert-base-uncased"  # Replace with your model's name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if choice == "Home":
        render_home()
    elif choice == "Analyze Movie Review":
        render_analyze_review(model, tokenizer)
    elif choice == "Top Reviews":
        render_top_reviews()

# Home Page
def render_home():
    st.title("🎬 Movie Sentiment Analysis")
    st.write("This project was developed as part of CS 124 Honors Fall 2024 at the University of Illinois - Urbana Champaign.")
    st.image("Assets/movie_review.png", use_column_width=True)
    st.write("""
    Welcome to our Movie Sentiment Analysis website! 🎥🍿
    - Navigate to the **Analyze Movie Review** section to find out what sentiment your review conveys.
    - Check out some of the best movie reviews on the **Top Reviews** page.
    """)

# Analyze Movie Review Page
def render_analyze_review(model, tokenizer):
    st.title("🔍 Analyze Your Movie Review")
    st.write("Type or paste your movie review below to analyze its sentiment.")

    input_text = st.text_area("Write your movie review here...")

    # Perform Sentiment Analysis
    if st.button("Analyze Review"):
        if input_text:
            # Perform sentiment analysis using the model
            scores = sentiment_analysis(input_text, tokenizer, model)

            # Display sentiment scores
            st.subheader("Sentiment Scores")
            st.write("These are the individual scores predicted by the model for each sentiment:")
            for label, score in scores.items():
                st.write(f"**{label}**: {score:.2f}")

            sentiment_label = max(scores, key=scores.get)

            # Sentiment mapping
            sentiment_mapping = {
                "Negative": "Negative",
                "Positive": "Positive"
            }
            sentiment_readable = sentiment_mapping.get(sentiment_label)

            # Display the sentiment label
            st.subheader(f"Sentiment: **{sentiment_readable}**")

            # Display a star rating
            rating = map_sentiment_score_to_rating(scores[sentiment_label])
            rating = int(rating)
            st.subheader(f"⭐ Rating: {rating}/5")

    # Clear Input Button
    if st.button("Clear Input"):
        st.experimental_rerun()

# Top Reviews Page
def render_top_reviews():
    st.title("🌟 Top Movie Reviews")
    st.write("Here are some highly rated movie reviews:")

    # Sample movie reviews (these can be dynamically fetched from a database or API)
    top_reviews = [
        {
            "title": "Inception",
            "review": "A mind-bending masterpiece by Christopher Nolan.",
            "rating": 5
        },
        {
            "title": "The Dark Knight",
            "review": "Heath Ledger's Joker is unforgettable in this gripping thriller.",
            "rating": 5
        },
        {
            "title": "Interstellar",
            "review": "An epic journey through space and time with emotional depth.",
            "rating": 4
        }
    ]

    for review in top_reviews:
        st.write(f"**{review['title']}**")
        st.write(f"Review: {review['review']}")
        st.write(f"⭐ Rating: {review['rating']}/5")
        st.write("---")

# Run the main function
if __name__ == '__main__':
    main()