import pandas as pd
from sklearn.svm import SVC


data = pd.read_csv('UCI_Credit_Card.csv')
print(data.columns)
print(data.isnull().sum())
y = data['default.payment.next.month']
X = data.drop(['ID', 'default.payment.next.month'], axis = 1)

clf = SVC(gamma='auto')

clf.fit(X,y)
print(clf.score(X,y))
