import numpy as np
from util import base64_to_pil
from flask import Flask, request, make_response, jsonify
from keras.models import load_model
from keras.utils import img_to_array
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_ngrok import run_with_ngrok

app = Flask(__name__)

api = Api(app)
CORS(app)

model = load_model('./model/ArtefakOne.h5')

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
api.add_resource(Predict, '/predict', methods=['POST'])

if __name__ == '__main__':
    app.run()


{
    "" : "(paste code base64 hasil convert)"
}