from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('faraz.h5')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(open('ahmad_faraz.txt', encoding='utf-8').read().lower().split("\n"))


# Define function for text generation
def generate_text(seed_text, next_words=10):
    max_sequence_len = max([len(x) for x in tokenizer.texts_to_sequences([seed_text])])
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for text generation
@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.form['seed_text']
    next_words = int(request.form['next_words'])
    generated_text = generate_text(seed_text, next_words)
    return render_template('index.html', seed_text=seed_text, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
