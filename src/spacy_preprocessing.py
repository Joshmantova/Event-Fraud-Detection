import spacy
from spacy.tokens import DocBin
import pandas as pd
from bs4 import BeautifulSoup

def strip_html_tags(descrip: str) -> str:
    soup = BeautifulSoup(descrip, features="html.parser")
    text = soup.text

    return text


def descrip_to_spacy(data: pd.DataFrame, output_path: str) -> None:
    "Takes in dataframe with description and label to save DocBin to disk"
    tuples = data.apply(lambda row: (strip_html_tags(row["description"]), row["fraud"]), axis=1).to_list()
    nlp = spacy.blank("en")
    db = DocBin()

    for doc, label in nlp.pipe(tuples, as_tuples=True):
        if label:
            doc.cats["FRAUD"] = 1
            doc.cats["NOTFRAUD"] = 0
        else:
            doc.cats["FRAUD"] = 0
            doc.cats["NOTFRAUD"] = 1
        
        db.add(doc)
    
    db.to_disk(output_path)

if __name__ == "__main__":
    df_train = pd.read_json("../data/df_train.json")
    df_val = pd.read_json("../data/df_val.json")

    train_descrips = df_train[["description", "fraud"]]
    val_descrips = df_val[["description", "fraud"]]

    descrip_to_spacy(train_descrips, "../data/train.spacy")

    descrip_to_spacy(val_descrips, "../data/val.spacy")
