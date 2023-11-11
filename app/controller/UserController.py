from app.model.user import User
from app import reponse, app, db
from flask import request
from flask_jwt_extended import *
import datetime

def AddAdmin():
    try:
        name = request.form.get('name')
        password = request.form.get('password')
        email = request.form.get('email')
        level = 1

        # Validasi input
        if not name or not password or not email:
            return reponse.badRequest(None, "Harap isi semua kolom")

        user = User(name=name,  level=level, email=email)
        user.setPassword(password)
        db.session.add(user)
        db.session.commit()

        return reponse.success('', 'Berhasil menambahkan data admin')
    except Exception as e:
        # Mengembalikan respons error
        return reponse.badRequest(None, str(e))

def SingelObject(data):
    data = {
        'id' : data.id,
        'name' : data.name,
        'email': data.email,
        'level' : data.level
    }
    return data

def login():
    try:
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if not user:
            return reponse.info([],'email tidak terdaftar')
        
        if not user.checkPassword(password):
            return reponse.info([],'password salah')
        
        data = SingelObject(user)
        session = datetime.timedelta(days=7)
        session_refresh =  datetime.timedelta(days=7)

        access_token = create_access_token(data, fresh=True, expires_delta= session)
        refresh_token = create_refresh_token(data, expires_delta =session_refresh)
        
        return reponse.success({
            "data": data,
            "access_token": access_token,
            "refresh_token": refresh_token
        },"Login Berhasil!!!")
    
    except Exception as e:
        print(e)
