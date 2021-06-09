# Loading the libraries:

import numpy as np
import pandas as pd

# Importing the dataset:
    
df = pd.read_csv("IMDB Dataset.csv")

# Checking for null values:

df.isnull().sum()

# checking for unique values in sentiment column:

df["sentiment"].unique()

# Checking for empty strings:
    
blank = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blank.append(i)
            
blank
 # since there are no index values in blank, there are no empty strings 
     # in the dataset.
     
# Splitting into train-test sets:
    
X = df["review"]
y = df["sentiment"]    

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# Model Building:
    

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

# 1. LinearSVC:

from sklearn.svm import LinearSVC

text_classifier_svc = Pipeline([('tfidf', TfidfVectorizer()), ('svc', LinearSVC())])

text_classifier_svc.fit(X_train, y_train)

y_pred_svc = text_classifier_svc.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test, y_pred_svc)

# accuracy_score = 0.8998

print(confusion_matrix(y_test, y_pred_svc))

print(classification_report(y_test,y_pred_svc))

#                n     p
# precision     0.91  0.89
# recall        0.89  0.91


# 2 . Extra Trees Classifier:
    
from sklearn.ensemble import ExtraTreesClassifier

text_classifier_etc = Pipeline([('tfidf', TfidfVectorizer()), ('etc', ExtraTreesClassifier())])

text_classifier_etc.fit(X_train, y_train)

y_pred_etc = text_classifier_etc.predict(X_test)

accuracy_score(y_test, y_pred_etc)

# accuracy_score = 0.85

print(confusion_matrix(y_test, y_pred_etc))

print(classification_report(y_test, y_pred_etc))

#                n     p
# precision     0.86  0.86
# recall        0.86  0.86


# 3. Gradient Boosting Classifier:
    
from sklearn.ensemble import GradientBoostingClassifier

text_classifier_gbc = Pipeline([('tfidf', TfidfVectorizer()), ('gbc', GradientBoostingClassifier())])

text_classifier_gbc.fit(X_train, y_train)

y_pred_gbc = text_classifier_gbc.predict(X_test)

accuracy_score(y_test, y_pred_gbc)

# accuracy_score = 0.81

print(confusion_matrix(y_test, y_pred_gbc))

print(classification_report(y_test, y_pred_gbc))

#                n     p
# precision     0.85  0.78
# recall        0.76  0.87


# 4. Random Forest Classifier:
    
from sklearn.ensemble import RandomForestClassifier

text_classifier_rfc = Pipeline([('tfidf', TfidfVectorizer()), ('rfc', RandomForestClassifier())])

text_classifier_rfc.fit(X_train, y_train)

y_pred_rfc = text_classifier_rfc.predict(X_test)

accuracy_score(y_test, y_pred_rfc)

# accuracy_score = 0.84

print(confusion_matrix(y_test, y_pred_rfc))

print(classification_report(y_test, y_pred_rfc))
#                n     p
# precision     0.84  0.85
# recall        0.85  0.84

# 5. Logistic Regression:
    
from sklearn.linear_model import LogisticRegression

text_classifier_lg = Pipeline([('tfidf', TfidfVectorizer()), ('log_reg', LogisticRegression())])

text_classifier_lg.fit(X_train, y_train)

y_pred_lg = text_classifier_lg.predict(X_test)

accuracy_score(y_test, y_pred_lg)

# accuracy_score = 0.8968

print(confusion_matrix(y_test, y_pred_lg))

print(classification_report(y_test, y_pred_lg))
#                n     p
# precision     0.91  0.89
# recall        0.88  0.91

# 6. Decision Tree Classifier:
    
from sklearn.tree import DecisionTreeClassifier

text_classifier_dtc = Pipeline([('tfidf', TfidfVectorizer()), ('dtc', DecisionTreeClassifier())])

text_classifier_dtc.fit(X_train, y_train)

y_pred_dtc = text_classifier_dtc.predict(X_test)

accuracy_score(y_test, y_pred_dtc)

# accuracy_score = 0.7144

print(confusion_matrix(y_test, y_pred_dtc))

print(classification_report(y_test, y_pred_dtc))
#                n     p
# precision     0.71  0.72
# recall        0.72  0.71


# 7. KNeighbors Classifier:
    
from sklearn.neighbors import KNeighborsClassifier

text_classifier_knc = Pipeline([('tfidf', TfidfVectorizer()), ('knc', KNeighborsClassifier())])

text_classifier_knc.fit(X_train, y_train)

y_pred_knc = text_classifier_knc.predict(X_test)

accuracy_score(y_test, y_pred_knc)

# accuracy_score = 0.75

print(confusion_matrix(y_test, y_pred_knc))

print(classification_report(y_test, y_pred_knc))

#                n     p
# precision     0.77  0.74
# recall        0.72  0.79


# Conclusion:

# Basis the above analysis, it can be seen that the best performing 
    # algorithms are:
        # 1. Linear SVC
        # 2. LogosticRegression