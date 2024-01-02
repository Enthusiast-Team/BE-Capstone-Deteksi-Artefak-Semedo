import os

# Mengambil direktori basis dari file ini
basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    HOST = str(os.environ.get("DB_HOST"))
    DATABASE = str(os.environ.get("DB_DATABASE"))
    USERNAME = str(os.environ.get("DB_USERNAME"))
    PASSWORD = str(os.environ.get("DB_PASSWORD"))

    # Kunci rahasia untuk JWT
    JWT_SECRET_KEY = str(os.environ.get("JWT_SECRET"))

    # URL koneksi basis data
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://'+ USERNAME +':'+PASSWORD+'@'+HOST + '/'+ DATABASE

    # Menyimpan perubahan otomatis terhadap model di dalam basis data
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Merekam kueri database
    SQLALCHEMY_RECORD_QUERIES = True
