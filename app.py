from flask import Flask, request, render_template, jsonify
from MODEL import SimpleSpellingGrammarChecker
import numpy as np
from gensim.models import KeyedVectors
from keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords

app = Flask(__name__)
simple_spelling_grammar_checker = SimpleSpellingGrammarChecker()

# Load the pre-trained model and Word2Vec embeddings
lstm_model = load_model('final_lstm.h5')
word2vec_model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub("[^A-Za-z]", " ", text)
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# Convert text to vector
def text_to_vector(text):
    words = preprocess_text(text)
    vec = np.zeros((300,), dtype="float32")
    no_of_words = 0
    for word in words:
        if word in word2vec_model:
            vec = np.add(vec, word2vec_model[word])
            no_of_words += 1
    if no_of_words > 0:
        vec /= no_of_words
    return vec

# Get score
def get_score(text):
    text_vec = text_to_vector(text)
    text_vec = np.reshape(text_vec, (1, 1, text_vec.shape[0]))
    score = lstm_model.predict(text_vec)[0][0]
    return round(score, 2)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spell', methods=['POST'])
def spell():
    if request.method == 'POST':
        text = request.form['text']
        corrected_text = simple_spelling_grammar_checker.check_spelling(text)
        corrected_grammar = simple_spelling_grammar_checker.check_grammar(text)
        return render_template('index.html', corrected_text=corrected_text, corrected_grammar=corrected_grammar)

@app.route('/grammar', methods=['POST'])
def grammar():
    if request.method == 'POST':
        file_content = request.form['file']
        corrected_grammar = simple_spelling_grammar_checker.check_grammar(file_content)
        return render_template('index.html', corrected_grammar=corrected_grammar)

# @app.route('/score', methods=['POST'])
# def score():
#     if request.method == 'POST':
#         essay_text = request.form['essay']
#         if essay_text.strip() == "":
#             return jsonify({"error": "Please enter an essay!"})
#         else:
#             score = get_score(essay_text)
#             # Convert score to regular float
#             score = float(score)
#             return jsonify({"score": score})

@app.route('/score', methods=['POST'])
def score():
    if request.method == 'POST':
        essay_text = request.form['essay']
        if essay_text.strip() == "":
            return jsonify({"error": "Please enter an essay!"})
        else:
            score = get_score(essay_text)
            # Convert score to regular float
            score = float(score)
            # Return JSON response with the score
            return jsonify({"score": score})

@app.route('/submit_essay', methods=['GET'])
def submit_essay():
    return render_template('submit_essay.html')


if __name__ == "__main__":
    app.run(debug=True)
