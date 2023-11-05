import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv("fake_or_real_news.csv")
data['fake'] = data['label'].apply(lambda x: 0 if x == "real" else 1)
data = data.drop("label", axis=1)

X,y = data['text'], data['fake']
X_train, X_test, y_train, y_test = train_test_split(X, y test_size=0.2)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = LinearSVC()
clf.fit(X_test_vectorized, y_train)

clf.score(X_test_vectorized, y_test)

len(y_test) * 0.9479
len(y_test)

with open("mytext", "w", encoding="utf-8") as f:
    f.write(X_test.iloc[10])

with open("mytext.txt", "r", encoding="utf-8") as f:
    text = f.read()

vectorized_text = vectorizer.transform([text])

clf.predict(vectorized_text)
y_test.iloc