# Spam Detection System

A machine learning-based email and SMS spam classifier built with Streamlit. This application uses natural language processing (NLP) and machine learning to accurately classify messages as spam or legitimate.

## Features

- **Real-time Classification**: Instantly classify email or SMS messages as spam or not spam
- **Text Preprocessing**: Automatically processes input text with tokenization, stopword removal, and stemming
- **Machine Learning Model**: Uses trained scikit-learn models for accurate predictions
- **User-Friendly Interface**: Built with Streamlit for an intuitive web-based UI

## Requirements

- Python 3.7+
- streamlit
- scikit-learn
- nltk
- pickle (built-in)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Spam-ditection-system
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure you have the trained model files in the project root:
   - `vectorizer.pkl` - TF-IDF vectorizer for text transformation
   - `model.pkl` - Trained classification model

## Usage

Run the Streamlit application:
```bash
streamlit run main.py
```

The application will open in your default browser. Enter an email or SMS message and click "predict" to classify it as spam or legitimate.

## How It Works

1. **Text Preprocessing**: 
   - Converts text to lowercase
   - Tokenizes the text into individual words
   - Removes non-alphanumeric characters
   - Removes English stopwords and punctuation
   - Applies Porter Stemming for word normalization

2. **Vectorization**: 
   - Transforms the processed text using TF-IDF vectorizer

3. **Prediction**: 
   - Passes the vectorized text to the trained ML model
   - Returns classification result (spam or not spam)

## Project Structure

```
Spam-ditection-system/
├── main.py              # Main Streamlit application
├── vectorizer.pkl       # Pre-trained TF-IDF vectorizer
├── model.pkl            # Pre-trained classification model
├── README.md            # This file
└── .gitignore           # Git ignore file
```

## Notes

- The application automatically downloads required NLTK data (punkt tokenizer and stopwords) on first run
- Ensure pickle model files are compatible with your scikit-learn version

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature requests.
