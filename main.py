import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl','rb'))
        # Fix scikit-learn version compatibility issues where _idf_diag is present but idf_ is missing
        if hasattr(tfidf, '_tfidf') and not hasattr(tfidf._tfidf, 'idf_') and hasattr(tfidf._tfidf, '_idf_diag'):
            tfidf._tfidf.idf_ = tfidf._tfidf._idf_diag.diagonal()
        model = pickle.load(open('model.pkl','rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f" Error: Pickle files not found. Make sure vectorizer.pkl and model.pkl exist in the same directory.")
        st.stop()
    except (pickle.UnpicklingError, EOFError) as e:
        st.error(f" Error: Pickle files are corrupted or invalid. Please regenerate your model files.")
        st.stop()
    except Exception as e:
        st.error(f" Error loading models: {str(e)}")
        st.stop()

tfidf, model = load_models()
st.title('Email/SMS spam classifier')
input_sms=st.text_input('Enter your message')
ps=PorterStemmer()
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
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
if st.button('predict'):

    #1.preprocess
    transform_sms=transform_text(input_sms)
    #2.vectorize
    vector_input=tfidf.transform([transform_sms])
    prediction = model.predict(vector_input)[0]
    #3.predict
    #4. Display
    if prediction == 1:
        st.header('Email/SMS is spam')
    else:
        st.header('Email/SMS is not spam')




