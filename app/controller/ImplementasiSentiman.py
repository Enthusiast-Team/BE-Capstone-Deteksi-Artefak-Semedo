import pandas as pd
import pymysql
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TweetTokenizer
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels



db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': '',
    'database': 'review',
}
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Insert data into the table
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            cursor.execute(sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()


# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


# def read_mysql_table(table, host='localhost', user='root', password='', database='review'):
#     # Establish a connection to the MySQL database
#     connection = pymysql.connect(
#         host=host,
#         user=user,
#         password=password,
#         database=database
#     )

#     # Create a cursor object to execute SQL queries
#     cursor = connection.cursor()

#     query = f"SELECT * FROM {table}"
#     cursor.execute(query)
#     result = cursor.fetchall()

#     # Convert the result to a Pandas DataFrame
#     df = pd.DataFrame(result)

#     # Assign column names based on the cursor description
#     df.columns = [column[0] for column in cursor.description]

#     # Close the cursor and the database connection
#     cursor.close()
#     connection.close()

#     return df


# table_name = 'input_review'
# df = read_mysql_table(table_name)
# df.head()

# # import pandas as pd
# # dframe=pd.read_csv("ulasan-rating5.csv")
# # print(dframe[1:4])

# df.head()

# #menyimpan tweet. (tipe data series pandas)
# data_content = df['review']

# # casefolding
# data_casefolding = data_content.str.lower()
# data_casefolding.head()

# filtering_url = [re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", str(tweet)) for tweet in data_casefolding]
# #cont
# filtering_cont = [re.sub(r'\(cont\)'," ", tweet)for tweet in filtering_url]
# #punctuatuion
# filtering_punctuation = [re.sub('[!"”#$%&’()*+,-./:;<=>?@[\]^_`{|}~]', ' ', tweet) for tweet in filtering_cont]
# #  hapus #tagger
# filtering_tagger = [re.sub(r'#([^\s]+)', '', tweet) for tweet in filtering_punctuation]
# #numeric
# filtering_numeric = [re.sub(r'\d+', ' ', tweet) for tweet in filtering_tagger]

# data_filtering = pd.Series(filtering_numeric)

# # #tokenize
# tknzr = TweetTokenizer()
# data_tokenize = [tknzr.tokenize(tweet) for tweet in data_filtering]
# data_tokenize

# #slang word
# path_dataslang = open("modelai/kamus kata baku-clear.csv")
# dataslang = pd.read_csv(path_dataslang, encoding = 'utf-8', header=None, sep=";")

# def replaceSlang(word):
#   if word in list(dataslang[0]):
#     indexslang = list(dataslang[0]).index(word)
#     return dataslang[1][indexslang]
#   else:
#     return word

# data_formal = []
# for data in data_tokenize:
#   data_clean = [replaceSlang(word) for word in data]
#   data_formal.append(data_clean)
# len_data_formal = len(data_formal)


# nltk.download('stopwords')
# default_stop_words = nltk.corpus.stopwords.words('indonesian')
# stopwords = set(default_stop_words)

# def removeStopWords(line, stopwords):
#   words = []
#   for word in line:
#     word=str(word)
#     word = word.strip()
#     if word not in stopwords and word != "" and word != "&":
#       words.append(word)

#   return words
# reviews = [removeStopWords(line,stopwords) for line in data_formal]

# file_path = 'modelai/reviews.pkl'

# # Read the pickle file
# with open(file_path, 'rb') as file:
#     data_train = pickle.load(file)

# data_train

# # pembuatan vector kata
# vectorizer = TfidfVectorizer()
# train_vector = vectorizer.fit_transform(data_train)
# reviews2 = [" ".join(r) for r in reviews]

# import joblib

# load_model = joblib.load(open('/modelai/revisi_hasil_sentimen.pkl','rb'))

# result = []
# for test in reviews2:
#     test_data = [str(test)]
#     test_vector = vectorizer.transform(test_data).toarray()
#     pred = load_model.predict(test_vector)
#     result.append(pred[0])

# result

# from sklearn.utils.multiclass import unique_labels
# unique_labels(result)

# df['label'] = result

# def delete_all_data_from_table(table, host='localhost', user='root', password='', database='review'):
#     # Establish a connection to the MySQL database
#     connection = pymysql.connect(
#         host=host,
#         user=user,
#         password=password,
#         database=database
#     )
#         # Create a cursor object to execute SQL queries
#     cursor = connection.cursor()

#     # Delete all data from the specified table
#     query = f"DELETE FROM {table}"
#     cursor.execute(query)

#     # Commit the changes
#     connection.commit()

#     # Close the cursor and the database connection
#     cursor.close()
#     connection.close()

# delete_all_data_from_table('input_review')

# def insert_df_into_hasil_model(df, host='localhost', user='root', password='', database='review'):
#     # Establish a connection to the MySQL database
#     connection = pymysql.connect(
#         host=host,
#         user=user,
#         password=password,
#         database=database
#     )
#      # Create a cursor object to execute SQL queries
#     cursor = connection.cursor()

#     # Insert each row from the DataFrame into the 'hasil_model' table
#     for index, row in df.iterrows():
#         query = "INSERT INTO hasil_model (id_review, nama, tanggal, review, label) VALUES (%s, %s, %s, %s, %s)"
#         cursor.execute(query, (row['id_review'], row['nama'], row['tanggal'], row['review'], row['label']))

#     # Commit the changes
#     connection.commit()

#     # Close the cursor and the database connection
#     cursor.close()
#     connection.close()

# insert_df_into_hasil_model(df)

# table_name = 'hasil_model'
# hasil_df = read_mysql_table(table_name)
# hasil_df.to_csv('./modelai/hasil_model_baru.csv')