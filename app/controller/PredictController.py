import numpy as np
from util import base64_to_pil
from flask import request
from keras.models import load_model
from keras.utils import img_to_array
from flask_restful import Resource
from flask_ngrok import run_with_ngrok
from tensorflow import keras
from tensorflow.keras.utils import load_img, img_to_array



model = load_model('./modelai/modelMobileNetNew.h5')

def model_predict(img, model):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 255.0
    preds = model.predict(x)
    return preds

target_names = ['australopithecus_aferensis', 
                'badak_bercula_satu', 
                'batu_lempung_krisikan',
                'fauna_darat_semedo',
                'fragmen_tengkorak',
                'fragmen_tengkorak_parential',
                'fragmen_tengkorak_sambung_macan',
                'fragmen_tulang_fermur_bawah',
                'gigi_buaya',
                'gigi_gajah',
                'gigi_hiu',
                'homo_erectus_ngawi',
                'homo_habilis',
                'homo_sapiens_wajak',
                'kapak_penetak',
                'kapak_primbas']

display_names = ['Australopithecus Aferensis', 
                'Badak Bercula Satu', 
                'Batu Lempung Krisikan',
                'Fauna Darat Semedo',
                'Fragmen Tengkorak',
                'Fragmen Tengkorak Parential',
                'Fragmen Tengkorak Sambung_macan',
                'Fragmen Tulang Fermur Bawah',
                'Gigi Buaya',
                'Gigi Gajah',
                'Gigi Hiu',
                'Homo Erectus Ngawi',
                'Homo Habilis',
                'Homo Sapiens Wajak',
                'Kapak Penetak',
                'Kapak Primbas']

label_mapping = dict(zip(target_names, display_names))

class Predict(Resource):
    def post(self):
        try:
            jsonObj = request.json
            data = jsonObj['message']
            img = base64_to_pil(data)
            pred = model_predict(img, model)
            hasil_label = label_mapping[target_names[np.argmax(pred)]]
            hasil_prob = "{:.2f}".format(100 * np.max(pred))
            return {'message': 'Berhasil melakukan prediksi', 'nama': hasil_label, 'probability': hasil_prob}, 200

        except Exception as e:
            print("EXCEPTION in Predict POST")
            return {'message': str(e), 'nama': '', 'probability': ''}, 400