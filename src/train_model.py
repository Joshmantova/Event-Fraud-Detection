import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as smote_pipeline

import datetime as dt

import joblib

def engineer_features(_df):
    """Engineer several features based on EDA that may be useful in prediction including
    transforming the string descriptions to a probability of fraud"""
    df = _df.copy()
    #Several columns need to become datetimes in order to perform the necessary operations
    cols_to_dt = [
    'event_created', 'event_end', 'event_start', 'user_created', 'approx_payout_date'
    ]
    for col in cols_to_dt:
        df[col] = df[col].astype(int).apply(dt.datetime.utcfromtimestamp)
    #setting up popular countries list for future use
    popular_countries = ['US', 'GB', 'AU']

    #Defining several functions that will be used below to transform the data
    def transform_description_to_proba(descriptions):
        "Takes NLP pipe trained on training data only to transform descriptions to prob of fraud"
        nlp_pipe = joblib.load('../models/nlp_pipe.joblib')
        return nlp_pipe.predict_proba(descriptions)[:, 1]

    def get_ticket_price(val):
        """Calculates average ticket price across all training events"""
        price = []
        for ticket in val:
            price.append(ticket['cost'])
        if len(price):
            return np.mean(price)
        else:
            return 0

    def get_ticket_available(val):
        """Calculates number of tickets available"""
        available = []
        for ticket in val:
            available.append(ticket['quantity_total'])
        if sum(available):
            return sum(available)
        else:
            return 1

    def get_avg_quantity_sold(val):
        "Gets avg number of tickets sold across all ticket types"
        amounts = [amount for amount in val['quantity_sold']]
        return np.mean(amounts)

    def get_num_payouts(val):
        "Gets number of previous payouts"
        number_of_payouts = len(val)
        return number_of_payouts

    def payee_name_in_org_name(row):
        "Calculating a bool feature to represent the payee name being in the org name"
        return row['payee_name'] in row['org_name']

    def venue_country_is_source_country(row):
        "Checking if the venue country is the country the event was created in"
        return row['venue_country']==row['country']

    #Using the previously defined functions to transform the data
    df['avg_ticket_price'] = df['ticket_types'].apply(get_ticket_price)
    df['total_tickets_available'] = df['ticket_types'].apply(get_ticket_available)
    df['number_of_payouts'] = df['previous_payouts'].apply(get_num_payouts)
    df['avg_quantity_sold'] = df['ticket_types'].apply(get_avg_quantity_sold)

    #Combining multiple columns:
    df['payee_name_in_org_name'] = df.apply(payee_name_in_org_name, axis=1)
    df['venue_country_is_source_country'] = df.apply(venue_country_is_source_country, axis=1)
    df['avg_cost_per_ticket'] = df['avg_ticket_price'] / df['total_tickets_available']

    #Transforming description:
    df['description'] = transform_description_to_proba(df['description'])

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
    #converting our cleaning function to a sklearn transformer to be placed in a pipeline
    # eng_feats = FunctionTransformer(func=engineer_features)

    #Let's now create our main pipeline
    num_cols = ['body_length', 'description', 'name_length', 'num_payouts', 'user_age', 
                'avg_ticket_price', 'total_tickets_available', 'number_of_payouts', 
                'avg_quantity_sold', 'avg_cost_per_ticket', 'days_to_event', 'event_length']

    cat_cols = ['country', 'currency', 'delivery_method', 'fb_published', 'has_analytics', 
                'has_header', 'has_logo', 'listed', 'payout_type', 'show_map', 'venue_country', 
                'venue_state', 'payee_name_in_org_name', 'venue_country_is_source_country', 'pop_country']

    num_transformer = Pipeline(steps=[
        ('imputer', KNNImputer()),
        ('std', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    trans = ColumnTransformer(transformers=[
        ('numeric_transformer', num_transformer, num_cols),
        ('categorical_transformer', cat_transformer, cat_cols)
    ])

    pipe = smote_pipeline(steps=[
        # ('fe', eng_feats),
        ('transformer', trans),
        ('smoter', SMOTE(n_jobs=-1)),
        ('cls', RandomForestClassifier(n_jobs=-1, n_estimators=1000))
    ])

    #Let's now save this pipe to be used later
    joblib.dump(pipe, '../models/untrained_pipe.joblib')

    #let's also train the model and save the trained pipe
    df = pd.read_csv('clean_data.csv')
    #the dataframe contains a column named acct_type that includes the word fraud if the event was fraud.
    #lets create our labels from that.
    y = df['fraud']
    X = df.drop(['fraud'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        stratify=y)
    
    # X_train_processed = engineer_features(X_train)
    print('Fitting the model')
    pipe.fit(X_train, y_train)
    
    # print('Getting cross val f1 scores')
    #let's assess the model:
    # f1_scores = cross_val_score(pipe, X_train, y_train, cv=3,
    #                             scoring='f1', n_jobs=-1)
    # print(f'The average f1 score across folds was: {f1_scores.mean()}')
    # print(f'The list of f1 scores is: {f1_scores}')

    print('Getting accuracy, recall, precision, and f1 scores')
    train_predictions = pipe.predict(X_train)
    test_predictions = pipe.predict(X_test)

    train_acc = accuracy_score(y_train, train_predictions) #1.0
    test_acc = accuracy_score(y_test, test_predictions) #.99

    train_recall = recall_score(y_train, train_predictions) #1.0
    test_recall = recall_score(y_test, test_predictions) #.93

    train_precision = precision_score(y_train, train_predictions) #1.0
    test_precision = precision_score(y_test, test_predictions) #.97

    train_f1 = f1_score(y_train, train_predictions) #1.0
    test_f1 = f1_score(y_test, test_predictions) #.95

    print(f"Training accuracy was: {train_acc}\nTest accuracy was: {test_acc}")
    print(f"Training recall was: {train_recall}\nTest recall was: {test_recall}")
    print(f"Training precision was: {train_precision}\nTest precision was: {test_precision}")
    print(f"Training f1 was: {train_f1}\nTest f1 was: {test_f1}")

    joblib.dump(pipe, '../models/prediction_pipe.joblib')