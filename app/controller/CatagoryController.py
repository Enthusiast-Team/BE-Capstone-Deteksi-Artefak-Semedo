from app import db
from app.model.Catagory import Category

class CategoryController:
    @staticmethod
    def tambah_category(nama, deskripsi):
        category_baru = Category(nama=nama, deskripsi=deskripsi)
        db.session.add(category_baru)
        db.session.commit()
        return category_baru

    @staticmethod
    def dapatkan_semua_category():
        return Category.query.all()

    @staticmethod
    def dapatkan_category_by_id(category_id):
        return Category.query.get(category_id)

    @staticmethod
    def perbarui_category(category, nama, deskripsi):
        category.nama = nama
        category.deskripsi = deskripsi
        db.session.commit()
        return category

    @staticmethod
    def hapus_category(category):
        db.session.delete(category)
        db.session.commit()
