import streamlit as st
import pickle
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer1.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()


def remove_urls(text):
    url_pattern = r"http[s]?://\S+|www\.\S+"
    return re.sub(url_pattern, '', text)

def remove_html_tags(text):
    html_pattern = re.compile(r'<.*?>')
    return re.sub(html_pattern, '', text)

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

lemm = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    
    text = text.lower()
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_special_characters(text)
    tokens = nltk.word_tokenize(text)
    clean_tokens = []


    for token in tokens:
        if token.isalnum() and token not in stop_words and token not in string.punctuation:
            clean_tokens.append(lemm.lemmatize(token))

    return " ".join(clean_tokens)


st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

st.title("Spam Classifier App")
st.write("Enter a message below and the model will classify it as **Spam** or **Not Spam**.")

user_input = st.text_area("✍️ Enter your message:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        processed_text = transform_text(user_input)  

        if processed_text.strip() == "":
            st.warning("⚠️ The message became empty after preprocessing. Try another input.")
        else:
            try:
                vectorized = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized)[0]

                if prediction == 1:  
                    st.error("🚨 This message is classified as **SPAM**.")
                else:
                    st.success("✅ This message is classified as **NOT SPAM**.")

                proba = model.predict_proba(vectorized)[0]
                st.write("### Prediction Probabilities")
                st.write(f"Not Spam: {proba[0]*100:.2f}%")
                st.write(f"Spam: {proba[1]*100:.2f}%")
            except Exception as e:
                st.error(f"❌ Vectorizer error: {e}")
