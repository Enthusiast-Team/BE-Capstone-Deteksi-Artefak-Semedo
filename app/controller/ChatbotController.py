import random
import json
from keras.models import load_model
import pickle
import numpy as np
from flask import make_response, jsonify
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')

class Chatbot:
    def __init__(self, model_path):
        # Pindahkan model loading ke dalam __init__
        self.model = load_model(model_path)
        self.intents = json.loads(open("./modelai/intents.json").read())
        self.words = pickle.load(open('./modelai/words.pkl', 'rb'))
        self.classes = pickle.load(open('./modelai/classes.pkl', 'rb'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_up_sentence(self, sentence):
        # Gunakan self.lemmatizer
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    if show_details:
                        print("found in bag: %s" % w)
        return np.array(bag)

    def predict_class(self, sentence):
        p = self.bow(sentence, self.words, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        error = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > error]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints):
        tag = ints[0]['intent']
        list_of_intents = self.intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

    def chatbot_response(self, text):
        ints = self.predict_class(text)
        res = self.getResponse(ints)
        return res

# Buat instance Chatbot
chatbot_instance = Chatbot('./modelai/chatbot.h5')