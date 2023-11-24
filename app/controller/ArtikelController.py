from app import db
from app.model.artikel import Artikel
from datetime import datetime

class ArtikelController:
    @staticmethod
    def tambah_artikel(gambar, judul, isi, category_id):
        artikel_baru = Artikel(gambar=gambar, judul=judul, isi=isi, category_id=category_id)
        db.session.add(artikel_baru)
        db.session.commit()
        return artikel_baru

    @staticmethod
    def dapatkan_semua_artikel():
        return Artikel.query.all()

    @staticmethod
    def dapatkan_artikel_by_id(artikel_id):
        return Artikel.query.get(artikel_id)

    @staticmethod
    def perbarui_artikel(artikel, gambar, judul, isi, category_id):
        artikel.gambar = gambar
        artikel.judul = judul
        artikel.isi = isi
        artikel.update_at = datetime.utcnow()
        artikel.category_id = category_id
        db.session.commit()
        return artikel

    @staticmethod
    def hapus_artikel(artikel):
        db.session.delete(artikel)
        db.session.commit()

    @staticmethod
    def dapatkan_artikel_by_judul(judul):
        return Artikel.query.filter_by(judul=judul).first()
