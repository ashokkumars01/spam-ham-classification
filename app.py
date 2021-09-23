from flask import Flask,render_template,url_for,request
from flask_mysqldb import MySQL
import pandas as pd 
import pickle
#from sklearn.feature_extraction.text import CountVectorizer
from logger_app import logs
import logging
import yaml

# load the model from disk
try:
    logging.info("Loading a model and transform file")
    filename = 'Extratrees_classifier.pkl'
    model = pickle.load(open(filename, 'rb'))
    cv=pickle.load(open('transform.pkl','rb'))
except Exception as e:
    print(e)
    logging.exception(e)

app = Flask(__name__)

# Configure db
'''try:
    logging.info("Configure Database")
    db = yaml.load(open('db.yaml'))
    app.config['MYSQL_HOST'] = db['mysql_host']
    app.config['MYSQL_USER'] = db['mysql_user']
    app.config['MYSQL_PASSWORD'] = db['mysql_password']
    app.config['MYSQL_DB'] = db['mysql_db']
except Exception as e:
    print(e)
    logging.exception(e)

mysql = MySQL(app)'''

@app.route('/')
def home():
    logging.info("Rendering a Home page")
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        logging.info("Reading a Message from user")
        if request.method == 'POST':
            Message = request.form['message']
            data = [Message]
            vect = cv.transform(data).toarray()
            my_prediction = model.predict(vect)
            if my_prediction ==0:
                Label = 'ham'
            elif my_prediction == 1:
                Label = 'spam'
            #mycursor = mysql.connection.cursor()
            #mycursor.execute("INSERT INTO prediction(Label, Message) VALUES(%s, %s)",(Label, Message))
            #mysql.connection.commit()
            #mycursor.close()
        return render_template('result.html',prediction = my_prediction)
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ == '__main__':
    app.run(debug=True)
