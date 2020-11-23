import pandas as pd
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as smote_pipeline

import joblib
import string

def spacy_tokenizer(text):
    punctuations = string.punctuation
    stop_words = STOP_WORDS
    parser = English()
    mytokens = parser(text)
    mytokens = [word.lemma_.lower().strip() for word in mytokens if str(word) not in stop_words and str(word) not in punctuations]
    return mytokens

if __name__=='__main__':
    #Let's read in the data first:
    df = pd.read_json('../Data/data.json')
    #This data has a column named acct_type that represents what kind of account the member has
    #Accounts with fraud in the name are associated with a fraudulent event, so let's get our labels from that
    y = df['acct_type'].str.contains('fraud')
    #Let's now create our X matrix:
    X = df.drop(['acct_type'], axis=1)
    #Train test split to be able to assess overfitting:
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        stratify=y)    

    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer,
                                ngram_range=(2, 2),
                                max_features=100000,
                                token_pattern=None,
                                max_df=0.95)
    nlp_pipe = smote_pipeline(steps=[
        ('vectorizer', vectorizer),
        ('smoter', SMOTE(n_jobs=-1)),
        ('cls', MultinomialNB())
    ])
    print(nlp_pipe)
    print('Training...')
    nlp_pipe.fit(X_train['description'], y_train)
    print('Model Trained!')
    print(nlp_pipe)
    #Saving the model
    joblib.dump(nlp_pipe, '../models/nlp_pipe.joblib')
    print('model saved')
    #Let's assess the NLP model
    scores = cross_val_score(nlp_pipe, X_train['description'], y_train,
                            scoring='f1', cv=5, n_jobs=-1)
    print(f"The 5 fold cross validated F1 score mean is: {scores.mean()}\nThe STD across the 5 folds is: {scores.std()}")
    print(f"The individual scores are: {scores}")

    #mean: .38

    #grid searching
    grid = {'vectorizer__ngram_range': [(2, 2), (2, 3), (3, 3)],
            'vectorizer__max_features': [100000, 500000],
            'vectorizer__max_df': [.95, .97, .99]}
    search = GridSearchCV(nlp_pipe, param_grid=grid, scoring='f1', n_jobs=-1, cv=5)
    search.fit(X_train['description'], y_train)
    best_score = search.best_score_
    if best_score > .38:
        print(best_score)
        joblib.dump(search.best_estimator_, '../models/nlp_pipe.joblib')
    else:
        print(best_score)
