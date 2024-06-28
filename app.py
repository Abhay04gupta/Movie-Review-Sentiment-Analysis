from flask import Flask, render_template, request
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already done
nltk.download('stopwords')

dict=stopwords.words("english")

app = Flask(__name__)

# Load the sentiment analysis model and TF-IDF vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('embedding.pkl', 'rb') as f:
    embedding = pickle.load(f)

exclude = string.punctuation

short_forms = {
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "I've": "I have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "I'll": "I will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "let's": "let us",
    "that's": "that is",
    "who's": "who is",
    "what's": "what is",
    "here's": "here is",
    "there's": "there is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "y'all": "you all",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "ne'er": "never",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "oughta": "ought to",
    "dunno": "do not know",
    "whatcha": "what are you",
    "could've": "could have",
    "would've": "would have",
    "should've": "should have",
    "might've": "might have",
    "must've": "must have",
    "ain't": "is not"
}

##Replace Short forms
def replace_shortfrm(text):
    for word in text.split():
        if word in short_forms:
           text=text.replace(word,short_forms[word])
    return text
##Remove Tags
def remove_tags(text):
    pattern=re.compile("<.*?>")
    return pattern.sub(r"",text)
##Remove Urls
def remove_urls(text):
    pattern=re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub(r"",text)
##Remove Numeric Chracs
def remove_numchar(text):
    pattern=re.compile(r"\d+")
    return pattern.sub(r"",text)
##Remove Punctuation
def remove_punctuation(text):
    for char in exclude:
        text=text.replace(char,"")
    return text
##Remove Stopwords
def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in dict:
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)


def text_preprocessing(text):
    text=replace_shortfrm(text)
    text=remove_tags(text)
    text=remove_urls(text)
    text=remove_numchar(text)
    text=remove_punctuation(text)
    text=remove_stopwords(text)
    return text

@app.route('/', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'POST':
        comment = request.form.get('comment')

        # Check if comment is a string
        if isinstance(comment, str):
            # Preprocess the comment
            preprocessed_comment = text_preprocessing(comment)

            # Transform the preprocessed comment into a feature vector
            comment_vector = embedding.transform([preprocessed_comment])

            # Predict the sentiment
            sentiment = model.predict(comment_vector)[0]

            return render_template('index.html', sentiment=sentiment)
        else:
            return "Invalid input type", 400

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
