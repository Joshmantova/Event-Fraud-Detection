import requests
import pandas as pd
from src.clean_data_and_predict import Data
from src.nlp_model import spacy_tokenizer
import os

def test_data_source():
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    resp = requests.get(url)
    assert resp.status_code == 200

def test_clean_data():
    # df = pd.read_csv("tests/raw_test_data.csv")
    print(os.curdir)
    d = Data()
    d.clean()
    assert True