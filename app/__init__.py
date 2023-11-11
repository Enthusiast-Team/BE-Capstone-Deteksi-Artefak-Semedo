from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi aplikasi dari objek Config
app.config.from_object(Config)

# Inisialisasi modul SQLAlchemy
db = SQLAlchemy(app)

# Inisialisasi modul Flask-Migrate
migrate = Migrate(app, db)

# Inisialisasi modul JWTManager untuk manajemen JSON Web Tokens
# jwt = JWTManager(app)

# Impor rute dan model dari aplikasi Anda
from app import route
from app.model import user, artikel, Catagory
