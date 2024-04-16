from google.colab import drive
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

import pandas as pd

# Path to your dataset in Google Drive
file_path = "/content/drive/MyDrive/True.csv"

file_path = "/content/drive/MyDrive/Fake.csv"

# Read the dataset into a DataFrame
df_true = pd.read_csv("/content/drive/MyDrive/True.csv")
df_fake = pd.read_csv("/content/drive/MyDrive/Fake.csv")

df_fake.head()

df_true.head(5)

# Inserting a column "class" as target feature
df_fake["class"] = 0
df_true["class"] = 1

# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
df_fake = df_fake.head(23471)

df_true_manual_testing = df_true.tail(10)
df_true = df_true.head(21407)

# Adding class to manual testing datasets
df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

# Merging True and Fake Dataframes
df_merge = pd.concat([df_fake, df_true], axis=0)

# Removing columns which are not required
df = df_merge.drop(["title", "subject", "date"], axis=1)

# Random Shuffling the dataframe
df = df.sample(frac=1)

# Reset index
df.reset_index(inplace=True)
df.drop(["index"], axis=1, inplace=True)

df.columns

df.head()

#prerocessing of data 
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


#conversion of text column to string type
df["text"] = df["text"].astype(str)


#applying preprocessing_function(wordplot)
df["text"] = df["text"].apply(wordopt)
