import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal,uniform

#czytanie danych
iris = datasets.load_iris()
X = iris["data"]
y = iris["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train[:5])

svm = SVC()

#definicja hiperparametrów
param_distributions = {
    'C':reciprocal(20,200_000), #rozkład odwrotny
    'gamma':uniform(0.0001,0.1) #rozkład jednostajny
}
rnd_search_cv = RandomizedSearchCV(svm,param_distributions,n_iter=10,verbose=2,cv=3,random_state=42)

rnd_search_cv.fit(X_train,y_train)

#wyniki tuningu hiperparametrów
print(f"najlepsze hiperparametry: {rnd_search_cv.best_params_}")

final_model = rnd_search_cv.best_estimator_
print(final_model.score(X_test,y_test))
