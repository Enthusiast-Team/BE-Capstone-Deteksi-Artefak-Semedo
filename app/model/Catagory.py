from app import db

class Category(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    nama = db.Column(db.String(250), nullable=False)
    deskripsi = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return '<Category {}>'.format(self.nama)
