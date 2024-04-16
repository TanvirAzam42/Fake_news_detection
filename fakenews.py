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



#extracting features
x = df["text"]
y = df["class"]

#split data into training and testing states.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Convert text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Logistic Regression
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, pred_dt))
print(classification_report(y_test, pred_dt))



# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
print("Gradient Boosting Classifier Accuracy:", accuracy_score(y_test, pred_gbc))
print(classification_report(y_test, pred_gbc))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))

# Model Testing
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(
        output_lable(pred_LR[0]),
        output_lable(pred_DT[0]),
        output_lable(pred_GBC[0]),
        output_lable(pred_RFC[0])
    ))

# Example usage:
news = input("Enter news text: ")
manual_testing(news)