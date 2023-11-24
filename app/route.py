from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from app import app
from app.controller.ArtikelController import ArtikelController
from Ai import chatbot_instance, Predict
from flask_cors import CORS


api = Api(app)
CORS(app)


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data.get('message')
    response = chatbot_instance.chatbot_response(message)
    return jsonify({'response': response})


api.add_resource(Predict, '/predict', methods=['POST'])


#TAMBAH DATA ARTIKEL 
@app.route('/artikel', methods=['POST'])
def tambah_artikel():
    data = request.json
    gambar = data.get('gambar')
    judul = data.get('judul')
    isi = data.get('isi')
    category_id = data.get('category_id')
    artikel_baru = ArtikelController.tambah_artikel(gambar, judul, isi, category_id)
    return jsonify({'message': 'Kategori berhasil ditambahkan', 
                    'kategori': 
                    {'nama': artikel_baru.gambar,
                     'judu': artikel_baru.judul ,
                     'isi': artikel_baru.isi,
                     'catagory': artikel_baru.category_id}}), 200

#GET SEMUA DATA ARTIKEL
@app.route('/artikel', methods=['GET'])
def dapatkan_semua_artikel():
    semua_artikel = ArtikelController.dapatkan_semua_artikel()
    artikel_list = []
    for artikel in semua_artikel:
        artikel_dict = {
            'id': artikel.id,
            'gambar': artikel.gambar,
            'judul': artikel.judul,
            'isi': artikel.isi,
            'created_at': artikel.created_at,
            'update_at': artikel.update_at,
            'category_id': artikel.category_id
        }
        artikel_list.append(artikel_dict)

    return jsonify(artikel_list), 200

##get data artikel by ID
@app.route('/artikel/<int:artikel_id>', methods=['GET'])
def dapatkan_artikel_by_id(artikel_id):
    artikel = ArtikelController.dapatkan_artikel_by_id(artikel_id)
    if artikel is not None:
        artikel_dict = {
            'id': artikel.id,
            'gambar': artikel.gambar,
            'judul': artikel.judul,
            'isi': artikel.isi,
            'created_at': artikel.created_at,
            'update_at': artikel.update_at,
            'category_id': artikel.category_id
        }
        return jsonify(artikel_dict), 200
    else:
        return jsonify({'message': 'Artikel tidak ditemukan'}), 404

#UPDATE ARTIKEL 
@app.route('/artikel/<int:artikel_id>', methods=['PUT'])
def perbarui_artikel(artikel_id):
    data = request.json
    gambar = data.get('gambar')
    judul = data.get('judul')
    isi = data.get('isi')
    category_id = data.get('category_id')

    artikel = ArtikelController.dapatkan_artikel_by_id(artikel_id)
    if artikel is not None:
        ArtikelController.perbarui_artikel(artikel, gambar, judul, isi, category_id)
        return jsonify({'message': 'Artikel berhasil diperbarui', 'artikel_id': artikel.id}), 200
    else:
        return jsonify({'message': 'Artikel tidak ditemukan'}), 404


@app.route('/artikel/<int:artikel_id>', methods=['DELETE'])
def hapus_artikel(artikel_id):
    artikel = ArtikelController.dapatkan_artikel_by_id(artikel_id)
    if artikel is not None:
        ArtikelController.hapus_artikel(artikel)
        return jsonify({'message': 'Artikel berhasil dihapus'}), 200
    else:
        return jsonify({'message': 'Artikel tidak ditemukan'}), 404
    
@app.route('/artikel/<judul>', methods=['GET'])
def dapatkan_artikel_by_judul(judul):
    artikel = ArtikelController.dapatkan_artikel_by_judul(judul)

    if artikel:
        # Jika artikel ditemukan, kembalikan data artikel dalam format JSON
        return jsonify({
            'status': 'success',
            'data': {
                'id': artikel.id,
                'judul': artikel.judul,
                'gambar': artikel.gambar,
                'isi': artikel.isi,
                'created_at': artikel.created_at,
                'update_at': artikel.update_at,
                'category_id': artikel.category_id,
                # 'nama': artikel.nama  # Sertakan nama jika diperlukan
            }
        }), 200
    else:
        # Jika artikel tidak ditemukan, kembalikan pesan error
        return jsonify({
            'status': 'error',
            'message': 'Artikel tidak ditemukan.'
        }), 404


from app.controller.CatagoryController import CategoryController

@app.route('/category', methods=['POST'])
def tambah_category():
    data = request.json
    nama = data.get('nama')
    deskripsi = data.get('deskripsi')
    
    category_baru = CategoryController.tambah_category(nama, deskripsi)
    return jsonify({'message': 'Kategori berhasil ditambahkan', 'kategori': {'nama': category_baru.nama, 'deskripsi': category_baru.deskripsi}}), 201


@app.route('/category', methods=['GET'])
def dapatkan_semua_category():
    semua_kategori = CategoryController.dapatkan_semua_category()
    return jsonify([{'id': kategori.id, 'nama': kategori.nama, 'deskripsi': kategori.deskripsi} for kategori in semua_kategori]), 200


@app.route('/category/<int:category_id>', methods=['GET'])
def dapatkan_category_by_id(category_id):
    kategori = CategoryController.dapatkan_category_by_id(category_id)
    if kategori is not None:
        return jsonify(kategori.__dict__), 200
    else:
        return jsonify({'message': 'Kategori tidak ditemukan'}), 404

@app.route('/category/<int:category_id>', methods=['PUT'])
def perbarui_category(category_id):
    data = request.json
    nama = data.get('nama')
    deskripsi = data.get('deskripsi')

    kategori = CategoryController.dapatkan_category_by_id(category_id)
    if kategori is not None:
        kategori = CategoryController.perbarui_category(kategori, nama, deskripsi)
        return jsonify({'message': 'Kategori berhasil diperbarui', 'kategori': kategori.__dict__}), 200
    else:
        return jsonify({'message': 'Kategori tidak ditemukan'}), 404

@app.route('/category/<int:category_id>', methods=['DELETE'])
def hapus_category(category_id):
    kategori = CategoryController.dapatkan_category_by_id(category_id)
    if kategori is not None:
        CategoryController.hapus_category(kategori)
        return jsonify({'message': 'Kategori berhasil dihapus'}), 200
    else:
        return jsonify({'message': 'Kategori tidak ditemukan'}), 404
