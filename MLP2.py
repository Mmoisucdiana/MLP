# Ce folosim:
# MLP
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd




from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
X=data.data
y=data.target

# print(data.DESCR)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

model=MLPClassifier(hidden_layer_sizes=(100,),activation='tanh',alpha=0.0001,learning_rate='constant',max_iter=500, random_state=42)



model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test, y_pred)
print(accuracy)
#accuracy(0,97)-hidden_layer_sizes=(50,),activation='relu',alpha=0.0001,learning_rate='constant',max_iter=500, random_state=42
#accuracy(0,98)-hidden_layer_sizes=(50,),activation='tanh',alpha=0.0001,learning_rate='constant',max_iter=500, random_state=42)
#acuracy(0.99)-hidden_layer_sizes=(100,),activation='tanh',alpha=0.0001,learning_rate='constant',max_iter=500, random_state=42

#varianta mai robusta si mai bun estimator dpdv. al stabilitatii

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=500, random_state=42),
                           param_grid=param_grid,
                           cv=5,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

#Best parameters:
#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50,), 'learning_rate': 'constant'}
#Accuracy: 0.9736842105263158

