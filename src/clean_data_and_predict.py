import pandas as pd
import numpy as np
import datetime as dt

import joblib
import os
import zipfile
#need to import the feature engineering function to work with the pipe
from nlp_model import spacy_tokenizer

class Data:

    def __init__(self, path_to_json_data=None):
        if path_to_json_data==None:
            self.raw_data = pd.read_json('http://galvanize-case-study-on-fraud.herokuapp.com/data_point', orient='index').T
        else:
            self.raw_data = pd.read_json(path_to_json_data)
        self.pipe = None
    
    def _load_prediction_pipe(self):
        if 'prediction_pipe.joblib' not in os.listdir('models/'):
            with zipfile.ZipFile('models/prediction_pipe.joblib.zip', 'r') as zip_ref:
                zip_ref.extractall('models/')
        pipe = joblib.load('models/prediction_pipe.joblib')
        return pipe

    def predict(self, X):
        self.pipe = self._load_prediction_pipe()
        return self.pipe.predict_proba(X)[0][1]

    #Defining several functions that will be used below to transform the data
    def transform_description_to_proba(self, descriptions):
        "Takes NLP pipe trained on training data only to transform descriptions to prob of fraud"
        nlp_pipe = joblib.load('models/nlp_pipe.joblib')
        return nlp_pipe.predict_proba(descriptions)[:, 1]

    def get_ticket_price(self, val):
        """Calculates average ticket price across all training events"""
        price = []
        for ticket in val:
            price.append(ticket['cost'])
        if len(price):
            return np.mean(price)
        else:
            return 0

    def get_ticket_available(self, val):
        """Calculates number of tickets available"""
        available = []
        for ticket in val:
            available.append(ticket['quantity_total'])
        if sum(available):
            return sum(available)
        else:
            return 1

    def get_avg_quantity_sold(self, val):
        "Gets avg number of tickets sold across all ticket types"
        amounts = [v['quantity_sold'] for v in val]
        return np.mean(amounts)

    def get_num_payouts(self, val):
        "Gets number of previous payouts"
        number_of_payouts = len(val)
        return number_of_payouts

    def payee_name_in_org_name(self, row):
        "Calculating a bool feature to represent the payee name being in the org name"
        return row['payee_name'] in row['org_name']

    def venue_country_is_source_country(self, row):
        "Checking if the venue country is the country the event was created in"
        return row['venue_country']==row['country']

    def clean(self, _df=None):
        """Engineer several features based on EDA that may be useful in prediction including
        transforming the string descriptions to a probability of fraud"""
        if _df==None:
            df = self.raw_data.copy()
        else:
            df = _df.copy()
        #Several columns need to become datetimes in order to perform the necessary operations
        cols_to_dt = [
        'event_created', 'event_end', 'event_start', 'user_created', 'approx_payout_date'
        ]
        for col in cols_to_dt:
            df[col] = df[col].astype(int).apply(dt.datetime.utcfromtimestamp)
        #setting up popular countries list for future use
        popular_countries = ['US', 'GB', 'AU']

        #Using the previously defined functions to transform the data
        df['avg_ticket_price'] = df['ticket_types'].apply(self.get_ticket_price)
        df['total_tickets_available'] = df['ticket_types'].apply(self.get_ticket_available)
        df['number_of_payouts'] = df['previous_payouts'].apply(self.get_num_payouts)
        df['avg_quantity_sold'] = df['ticket_types'].apply(self.get_avg_quantity_sold)

        #Combining multiple columns:
        df['payee_name_in_org_name'] = df.apply(self.payee_name_in_org_name, axis=1)
        df['venue_country_is_source_country'] = df.apply(self.venue_country_is_source_country, axis=1)
        df['avg_cost_per_ticket'] = df['avg_ticket_price'] / df['total_tickets_available']

        #Transforming description:
        df['description'] = self.transform_description_to_proba(df['description'])

        #Datetime calculations:
        df['days_to_event'] = (df['event_start'] - df['event_created']).dt.days
        df['event_length'] = (df['event_end'] - df['event_start']).dt.days

        df['pop_country'] = df['country'].isin(popular_countries)

        #The following columns have already had information extracted from them, were deemed not useful through EDA,
        #Or I could not tell what the column represented and thus it may introduce data leakage
        cols_to_del = ['gts', 'email_domain', 'num_order',
                    'object_id', 'org_facebook', 'org_twitter',
                    'previous_payouts', 'payee_name', 'org_name',
                    'venue_latitude', 'venue_longitude', 'venue_name',
                    'event_published', 'channels', 'sale_duration2',
                        'user_type', 'ticket_types', 'venue_address',
                    'sale_duration', 'channels', 'approx_payout_date',
                    'name', 'org_desc']
        cols_to_del = cols_to_del + cols_to_dt

        df.drop(cols_to_del, axis=1, inplace=True)
        df.replace(to_replace='', value=np.nan, inplace=True)

        #changing dtypes:
        num_cols = [
        'body_length', 'name_length', 'num_payouts', 'user_age', 'days_to_event',
        'event_length', 'avg_ticket_price', 'total_tickets_available',
        'avg_cost_per_ticket', 'number_of_payouts', 'avg_quantity_sold',
        'description'
        ]

        cat_cols = [col for col in df.columns if col not in num_cols]
        df[cat_cols] = df[cat_cols].astype('category')
        df[num_cols] = df[num_cols].astype(float)

        return df
    
if __name__=='__main__':
    data = Data('../data/data.json')
    clean_data = data.clean()
    clean_data['fraud'] = clean_data['acct_type'].str.contains('fraud')
    clean_data.drop('acct_type', axis=1, inplace=True)
    clean_data.to_csv('clean_data.csv', index=False)