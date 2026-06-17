import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load Model
@st.cache_resource
def load_rnn_model():
    return load_model('simple_rnn_imdb.keras')

# Load Word Index
@st.cache_data
def load_word_index():
    return imdb.get_word_index()

model      = load_rnn_model()
word_index = load_word_index()

# Helper Functions
def preprocess(text):
    words   = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    return pad_sequences([encoded], maxlen=500)

def predict(review):
    score     = model.predict(preprocess(review), verbose=0)[0][0]
    sentiment = 'Positive' if score > 0.5 else 'Negative'
    return sentiment, float(score)

# UI
st.title('🎬 IMDB Sentiment Analysis')
st.write('Enter a movie review to classify it as **Positive** or **Negative**.')

user_input = st.text_area('Movie Review', height=150,
                          placeholder='e.g. The film was absolutely brilliant...')

if st.button('Classify', use_container_width=True):
    if user_input.strip():
        with st.spinner('Analysing...'):
            sentiment, score = predict(user_input)

        if sentiment == 'Positive':
            st.success(f'✅ Sentiment: **{sentiment}**')
        else:
            st.error(f'❌ Sentiment: **{sentiment}**')

        st.write(f'Prediction Score: `{score:.4f}`')
        st.progress(score if sentiment == 'Positive' else 1 - score,
                    text=f'Confidence: {max(score, 1-score)*100:.1f}%')
    else:
        st.warning('Please enter a review before classifying.')