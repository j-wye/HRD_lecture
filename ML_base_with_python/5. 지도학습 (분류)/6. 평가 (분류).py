from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 데이터 불러오기
from sklearn.model_selection import StratifiedKFold
model = DecisionTreeClassifier(random_state=0)

kfold = StratifiedKFold(n_splits=5)
for train_idx, test_idx in kfold.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(accuracy_score(y_test, pred))



# 정확도
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

# 정밀도
from sklearn.metrics import precision_score
precision_score(y_test, pred)

# 재현율
from sklearn.metrics import recall_score
recall_score(y_test, pred)

# f1
from sklearn.metrics import f1_score
f1_score(y_test, pred)

# roc_auc
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
model = XGBClassifier(random_state=0, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)
pred = model.predict_proba(X_test)
roc_auc_score(y_test, pred[:,1])