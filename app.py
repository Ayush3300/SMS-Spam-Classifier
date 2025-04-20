# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords
# import nltk
# from nltk.stem.porter import PorterStemmer

# ps = PorterStemmer()


# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words('english'):
#             y.append(i)

#     text = y[:]
#     y.clear()

#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Function to clean the input text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Set custom CSS for background and text
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f8;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #3f51b5;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        color: gray;
        margin-top: 30px;
    }
    .msg-box {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="title">SMS Spam Classifier</div>', unsafe_allow_html=True)
st.markdown("This tool helps you detect whether a message is **Spam** or **Not Spam** using Random Forest Classifier and Machine Learning.")

# Input area
input_sms = st.text_area("📝 Enter your message:", placeholder="Type your SMS or email content here...")

# Predict Button
if st.button('🚀 Predict'):

    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    probability = model.predict_proba(vector_input)[0]

    # Show result with style
    if result == 1:
        st.markdown("<div class='msg-box'><h2 style='color:red;'>🚫 Spam</h2></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='msg-box'><h2 style='color:green;'>✅ Not Spam</h2></div>", unsafe_allow_html=True)

    # Optional: show confidence
    st.markdown(f"🔍 **Confidence:** `{max(probability) * 100:.2f}%`")
# Create three columns to center the button



# Sidebar info
with st.sidebar:
    st.header("📊 Model Details")
    st.write("- Model: Random Forest Classifier")
    st.write("- Vectorizer: TF-IDF")
    st.write("- Preprocessing: Tokenizing, Stopword removal, Stemming")
    st.markdown("---")
    # st.write("**Developed by:**   ")
    st.write("**Developed by:**")
    st.write("- Ayush Mutha")
    st.write("- Aavishkar Amrutwar")
    st.write("- Uday Londhe")
    st.write("- Parth Chavan")

# Footer
st.markdown('<div class="footer">Developed by You · April 2025</div>', unsafe_allow_html=True)
