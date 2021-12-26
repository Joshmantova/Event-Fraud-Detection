import pandas as pd

from sklearn.model_selection import train_test_split
import zipfile
import os


path_to_data = "../data/data.zip"


if "data.csv" not in os.listdir("../data/"):
    with zipfile.ZipFile(path_to_data, "r") as zip_ref:
        zip_ref.extractall("../data/")

df = pd.read_json("../data/data.json")
df['fraud'] = df['acct_type'].str.contains('fraud')

df_train, df_test = train_test_split(df, 
                                    test_size=0.20, 
                                    random_state=42, 
                                    stratify=df["fraud"]
                                    )

df_train, df_val = train_test_split(df_train,
                                    test_size=0.20,
                                    random_state=42,
                                    stratify=df_train["fraud"])

data_dict = {
    "df_train": df_train,
    "df_val": df_val,
    "df_test": df_test
}

for split_type, df in data_dict.items():
    print(df.shape)
    df.to_json(f"../data/{split_type}.json")
