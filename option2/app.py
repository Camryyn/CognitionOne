import streamlit as st 
import altair as alt
import plotly.express as px 
import pandas as pd 
import numpy as np 
from datetime import datetime
from transformers import pipeline

# Loading pre-trained emotion classifier pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table

def predict_emotions(docx):
    results = emotion_classifier(docx)
    results_sorted = sorted(results[0], key=lambda x: x['score'], reverse=True)
    return results_sorted[0]['label']

def get_prediction_proba(docx):
    results = emotion_classifier(docx)
    return {result['label']: result['score'] for result in results[0]}

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happiness":"🤗", "joy":"😂", "neutral":"😐", "sadness":"😔", "surprise":"😮"}

def main():
    st.set_page_config(page_title="Emotion Classifier App: Veer", layout="wide")
    st.markdown(  #camryn, riya and sahana, you can add the custom css over here
       """ 
        <style>
        /* General body styling */
        body {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
            /* Background with emoji image */
            background-image: url('https://m.media-amazon.com/images/I/718Hb42+PhL.jpg'); /* Replace with the actual image URL or path */
            background-repeat: no-repeat; /* Ensures no repetition of the image */
            background-size: cover; /* Scales the image to cover the entire background */
            background-position: center; /* Centers the image */
            margin: 0; /* Removes default margins */
            font-family: Arial, sans-serif; /* Sets a default font */

        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
            color: white;
        }
        [data-testid="stSidebar"] .css-1d391kg {
            color: #FFFFFF;
        }

        /* Title and headers */
        h1, h2, h3 {
            color: #333333;
        }

        /* Custom button style */
        .stButton button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            border: none;
            font-size: 16px;
            padding: 10px 20px;
            cursor: pointer;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }

        /* DataFrame styling */
        .css-17z80pu {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }

        /* Custom chart area */
        .stAltairChart {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Menu")
    menu = ["🏠 Home", "📊 Monitor", "ℹ️ About"]
    choice = st.sidebar.radio("Select Option", menu)
	

    create_page_visited_table()
    create_emotionclf_table()
    
    if choice == "🏠 Home":
        add_page_visited_details("Home", datetime.now())
        st.title("Emotion Classifier App")
        st.subheader("Enter text to analyze its emotion")
        
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            
            add_prediction_details(raw_text, prediction, max(probability.values()), datetime.now())

            col1, col2 = st.columns(2)

            with col1:
                st.success("Input Text")
                st.write(raw_text)

                st.success("Sentiment Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {max(probability.values()):.2f}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(list(probability.items()), columns=["emotions", "probability"])

                fig = alt.Chart(proba_df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif choice == "📊 Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.title("App Monitoring")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Pagename','Time_of_Visit'])
            st.dataframe(page_visited_details)    

            pg_count = page_visited_details['Pagename'].value_counts().rename_axis('Pagename').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Pagename', y='Counts', color='Pagename')
            st.altair_chart(c, use_container_width=True)    

            p = px.pie(pg_count, values='Counts', names='Pagename')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):  #initially showed Unicode decode error as Luke and I discussed in last week's meeting: "utf-8 codec cant decode byte"; fix given below:
            try:
                prediction_details = view_all_prediction_details()
                df_emotions = pd.DataFrame(prediction_details, columns=['Rawtext','Prediction','Probability','Time_of_Visit'])

                # fix for unicodedecodeerror: Ensuring all columns are converted to strings to avoid decoding errors:
                df_emotions = df_emotions.applymap(lambda x: x.decode('utf-8', 'ignore') if isinstance(x, bytes) else str(x))
                st.dataframe(df_emotions)

                prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
                pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
                st.altair_chart(pc, use_container_width=True)
            except UnicodeDecodeError as e:
                st.error(f"Error decoding data: {e}")
            
    else:
        st.title("About")
        add_page_visited_details("About", datetime.now())
        st.subheader("Emotion Classifier App")
        st.text("A simple application to classify emotions from text.")

if __name__ == '__main__':
    main()
