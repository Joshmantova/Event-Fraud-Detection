import pandas as pd
import numpy as np
import psycopg2 as pg2
from  sqlalchemy import create_engine

from flask import Flask, render_template

import sys
sys.path.append('../src/') #The following custom functions need to be imported from another folder
from nlp_model import spacy_tokenizer
from clean_data_and_predict import Data

app = Flask(__name__)

def add_data(hunters_df):
    conn = pg2.connect(dbname='fraud', user='postgres', host='localhost', port='5432')
    cur = conn.cursor()
    sql = hunters_df.to_sql('fraud_predictions', index=False, con=conn, if_exists='append')
    conn.commit()
    conn.close()
    # engine = create_engine('postgresql://postgres@localhost:5432')
    # hunters_df.to_sql('fraud_predictions', index=False, con=engine, if_exists='append')

def retrieve_data():
    conn = pg2.connect(dbname='fraud', user='postgres', host='localhost', port='5432')
    cur = conn.cursor()
    query = "SELECT * FROM fraud_predictions"
    cur.execute(query)
    data = []
    for r in cur.fetchall():
        data.append(r)
    conn.close()
    return data

# home page
@app.route('/', methods = ['POST', 'GET'])
def index():
    context = {"title":"Home"}
    return render_template('index.html', context=context) + 'Go to /score for predictions'

@app.route('/hello', methods = ['GET'])
def hello():
    return 'Hello, World! '

@app.route('/score', methods=['GET', 'POST'])
def score():
    data = Data()
    raw_data = data.raw_data
    clean_data = data.clean()
    prediction = data.predict(clean_data)
    if prediction < .25:
        ordinal_prediction = 'Low'
    elif prediction > .25 and prediction < .70:
        ordinal_prediction = 'Moderate'
    elif prediction > .70:
        ordinal_prediction = 'High'
    return f'''
    <div class = 'container'>
        <h1>Here is an incoming event and our prediction:</h1>
        <h3>Event name: {raw_data['name'][0]} </h3>
        <h3>Organization name: {raw_data['org_name'][0]} </h3>
        <h3>Current Fraud Potential: {ordinal_prediction} </h3>
        <h3> Likelihood of fraud: {prediction} </h3>
        <h3>Raw data: {raw_data.drop('description', axis=1).to_html(notebook=True)} </h3>
        <h3>Event description: {raw_data['description'][0]}
    </div>
    '''

@app.route('/data', methods=['GET', 'POST'])
def data():
    db_data = retrieve_data()
    return '''
    <div class = 'container'>
        <h1>Here is the model interaction for predictions</h1>
        <h3>Current Data: {0} </h3>
        <h3>
    </div>  
    '''.format(db_data)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8000, threaded=True)
