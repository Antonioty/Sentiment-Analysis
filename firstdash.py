import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
import string
import nltk
import seaborn as sns
from collections import Counter

# Set Streamlit page configuration to wide layout
st.set_page_config(layout="wide")

# Download stopwords if you haven't already
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('indonesian'))

# Load the dataset
df = pd.read_csv('data.csv')

# Preprocess text (remove stopwords and punctuation, lowercase)
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return " ".join(words)

df['Cleaned_Text'] = df['Ulasan'].apply(preprocess_text)

# Sentiment analysis function
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df[['Polarity', 'Subjectivity']] = df['Translated_Ulasan'].apply(lambda x: pd.Series(get_sentiment(x)))

# Classify sentiment
def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Polarity'].apply(classify_sentiment)

# Streamlit dashboard
st.title('🎨 Customer Reviews Sentiment Analysis Dashboard')
st.markdown("---")

# Display sentiment counts in columns
st.subheader("📊 Sentiment Count Overview")
col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div style='background-color:#D3D3D3;padding:20px;border-radius:10px;text-align:center'>"
              f"<h2>Total Reviews</h2><p style='font-size:24px;'><strong>{df.shape[0]}</strong></p></div>", unsafe_allow_html=True)
col2.markdown(f"<div style='background-color:#A6FFCB;padding:20px;border-radius:10px;text-align:center'>"
              f"<h2>Positive</h2><p style='font-size:24px;color:green;'><strong>{df[df['Sentiment'] == 'Positive'].shape[0]}</strong></p></div>", unsafe_allow_html=True)
col3.markdown(f"<div style='background-color:#FFAAAA;padding:20px;border-radius:10px;text-align:center'>"
              f"<h2>Negative</h2><p style='font-size:24px;color:red;'><strong>{df[df['Sentiment'] == 'Negative'].shape[0]}</strong></p></div>", unsafe_allow_html=True)
col4.markdown(f"<div style='background-color:#FFD580;padding:20px;border-radius:10px;text-align:center'>"
              f"<h2>Neutral</h2><p style='font-size:24px;color:orange;'><strong>{df[df['Sentiment'] == 'Neutral'].shape[0]}</strong></p></div>", unsafe_allow_html=True)

# Side-by-side visualizations for WordCloud and Word Frequency Bar Chart
st.subheader("🔎 Visual Analysis")
col5, col6 = st.columns(2)

# WordCloud of Indonesian reviews
with col5:
    st.markdown("🌥️ **WordCloud of Indonesian Reviews**")
    def generate_wordcloud(text):
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='coolwarm').generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

    all_text = " ".join(df['Cleaned_Text'])
    generate_wordcloud(all_text)

# Bar Chart of Top 10 Most Frequent Words
with col6:
    st.markdown("📈 **Top 10 Most Frequent Words**")
    word_counts = Counter(" ".join(df['Cleaned_Text']).split())
    common_words = word_counts.most_common(10)

    # Plot bar chart for most frequent words
    words, counts = zip(*common_words)
    plt.figure(figsize=(8, 4))
    plt.bar(words, counts, color=['#4CAF50', '#2196F3', '#FFC107', '#F44336', '#FF5722', '#009688', '#9C27B0', '#795548', '#E91E63', '#00BCD4'])
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 10 Most Frequent Words in Reviews', fontsize=15)
    st.pyplot(plt)

# Create two columns for the visualizations
col7, col8 = st.columns(2)

with col7:
    st.markdown("📊 **Sentiment Distribution**")
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(2, 2))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140,
            colors=['#A6FFCB', '#FFD580', '#FFAAAA'], explode=(0.05, 0.05, 0.05))
    plt.axis('equal')
    st.pyplot(plt)

with col8:
    st.markdown("🔍 **Polarity vs Sentiment**")
    
    # Mapping sentiment categories to numbers for better scatter plot
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment_Numeric'] = df['Sentiment'].map(sentiment_map)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='Polarity', y='Sentiment_Numeric', data=df, hue='Sentiment', palette=['green', 'orange', 'red'], alpha=0.6)
    plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Polarity', fontsize=12)
    plt.ylabel('Sentiment', fontsize=12)
    plt.title('Polarity vs. Sentiment', fontsize=15)
    st.pyplot(plt)
