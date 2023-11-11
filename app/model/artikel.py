from app import db
from datetime import datetime

class Artikel(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    gambar = db.Column(db.String(250), nullable=True)
    judul = db.Column(db.String(250), nullable=False)
    isi = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    update_at = db.Column(db.DateTime, default=datetime.utcnow)
    category_id = db.Column(db.BigInteger, db.ForeignKey('category.id'), nullable=False)
    
    category = db.relationship('Category', backref=db.backref('artikel', lazy=True))

    def __repr__(self):
        return '<Artikel {}>'.format(self.judul)
