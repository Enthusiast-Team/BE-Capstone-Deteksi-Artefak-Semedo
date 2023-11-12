import numpy as np
from util import base64_to_pil
from flask import Flask, request, make_response, jsonify
from keras.models import load_model
from keras.utils import img_to_array
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
import json
import pickle
import nltk
import random
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')



model = load_model('./modelai/modelMobileNet.h5')
# chatbot_model = load_model('./model/chatbot.h5')

def model_predict(img, model):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 255.0
    preds = model.predict(x)
    return preds

target_names = ['gigi_hiu', 
                'gigi_gajah', 
                'gigi_buaya',
                'fragmen_tengkorak_parential',
                'badak_bercula_1']
display_names = ['gigi_hiu', 
                'gigi_gajah', 
                'gigi_buaya', 
                'fragmen_tengkorak_parential', 
                'badak_bercula_1']
label_mapping = dict(zip(target_names, display_names))

class Predict(Resource):
    def post(self):
        try:
            data = request.json
            img = base64_to_pil(data)
            pred = model_predict(img, model)
            hasil_label = label_mapping[target_names[np.argmax(pred)]]
            hasil_prob = "{:.2f}".format(100 * np.max(pred))
            return make_response(jsonify({
                'status': '200', 'error': 'false', 'message': 'Berhasil melakukan prediksi', 'nama': hasil_label, 'probability': hasil_prob}))

        except Exception as e:
            return make_response(jsonify({'status': '400', 'error': 'true', 'message': str(e), 'nama': '', 'probability': ''}))

# Implementasi Model Chatbot
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

# Pengenalan Artefak
{
    "" : "(paste code base64 hasil convert)"
}

# Pengenalan Artefak
{
    "message" : "(pertanyaan)"
}