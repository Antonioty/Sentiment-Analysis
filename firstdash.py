import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from nltk.corpus import stopwords
import nltk
import string
from collections import Counter
import seaborn as sns

# Set Streamlit page configuration
st.set_page_config(layout="wide")

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Indonesian stopwords + Custom words to remove
stop_words = set(stopwords.words('indonesian')) | {
    'nya', 'yang', 'barang', 'produk', 'toko', 'dan', 'untuk', 'ok', 
    'yg', 'msi', 'dgn', 'ga', 'bgt', 'gak', 'sih', 'semoga', 'pengiriman', 
    'packing', 'seller', 'banget', 'deskripsi', 'bubble'
}

# Load dataset
df = pd.read_csv('data.csv', encoding='utf-8')

# Preprocess text: Remove stopwords, punctuation, and lowercase the text
def preprocess_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [
        word for word in words 
        if word not in stop_words and word not in string.punctuation
    ]
    return " ".join(words)

df['Cleaned_Text'] = df['Ulasan'].apply(preprocess_text)

# Sentiment analysis function
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

df[['Polarity', 'Subjectivity']] = df['Translated_Ulasan'].fillna('').apply(
    lambda x: pd.Series(get_sentiment(x))
)

# Classify sentiment based on polarity
def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment'] = df['Polarity'].apply(classify_sentiment)

# Dashboard title
st.title('🎨 Customer Reviews Sentiment Analysis Dashboard')
st.markdown("---")

# Sentiment count overview
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

# Visual analysis: WordCloud and word frequency bar chart
st.subheader("🔎 Visual Analysis")
col5, col6 = st.columns(2)

# Generate WordCloud
with col5:
    st.markdown("🌥️ **WordCloud of Indonesian Reviews**")
    
    def generate_wordcloud(text):
        wordcloud = WordCloud(
            width=800, height=400, background_color='white', 
            colormap='coolwarm', stopwords=stop_words
        ).generate(text)
        
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.clf()  # Clear the plot to prevent overlap

    all_text = " ".join(df['Cleaned_Text'])
    generate_wordcloud(all_text)

# Bar chart of top 10 most frequent words
with col6:
    st.markdown("📈 **Top 10 Most Frequent Words**")
    word_counts = Counter(" ".join(df['Cleaned_Text']).split())
    common_words = word_counts.most_common(10)

    # Plot bar chart
    words, counts = zip(*common_words)
    plt.figure(figsize=(8, 4))
    plt.bar(words, counts, color=['#4CAF50', '#2196F3', '#FFC107', '#F44336', 
                                  '#FF5722', '#009688', '#9C27B0', '#795548', 
                                  '#E91E63', '#00BCD4'])
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 10 Most Frequent Words in Reviews', fontsize=15)
    st.pyplot(plt)
    plt.clf()  # Clear the plot

# New: Top 5 Most Frequent Words in Positive and Negative Sentiments
st.subheader("📊 Most Frequent Words by Sentiment")

col7, col8 = st.columns(2)

with col7:
    st.markdown("🟢 **Top 5 Most Frequent Words in Positive Reviews**")
    positive_text = " ".join(df[df['Sentiment'] == 'Positive']['Cleaned_Text'])
    positive_counts = Counter(positive_text.split()).most_common(5)

    # Plot positive sentiment chart
    words, counts = zip(*positive_counts)
    plt.figure(figsize=(6, 4))
    plt.bar(words, counts, color='#4CAF50')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 5 Words in Positive Sentiment', fontsize=15)
    st.pyplot(plt)
    plt.clf()

with col8:
    st.markdown("🔴 **Top 5 Most Frequent Words in Negative Reviews**")
    negative_text = " ".join(df[df['Sentiment'] == 'Negative']['Cleaned_Text'])
    negative_counts = Counter(negative_text.split()).most_common(5)

    # Plot negative sentiment chart
    words, counts = zip(*negative_counts)
    plt.figure(figsize=(6, 4))
    plt.bar(words, counts, color='#F44336')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 5 Words in Negative Sentiment', fontsize=15)
    st.pyplot(plt)
    plt.clf()


# Sentiment distribution and polarity vs sentiment scatterplot
col9, col10 = st.columns(2)

with col9:
    st.markdown("📊 **Sentiment Distribution**")
    sentiment_counts = df['Sentiment'].value_counts()
    plt.figure(figsize=(4, 4))
    plt.pie(
        sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
        startangle=140, colors=['#A6FFCB', '#FFD580', '#FFAAAA'], explode=(0.05, 0.05, 0.05)
    )
    plt.axis('equal')
    st.pyplot(plt)
    plt.clf()

with col10:
    st.markdown("🔍 **Polarity vs Sentiment**")

    # Map sentiment categories to numbers
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    df['Sentiment_Numeric'] = df['Sentiment'].map(sentiment_map)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x='Polarity', y='Sentiment_Numeric', data=df, hue='Sentiment',
        palette=['green', 'orange', 'red'], alpha=0.6
    )
    plt.yticks([-1, 0, 1], ['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Polarity', fontsize=12)
    plt.ylabel('Sentiment', fontsize=12)
    plt.title('Polarity vs. Sentiment', fontsize=15)
    st.pyplot(plt)
    plt.clf()
