# Ce folosim:
# MLP
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.datasets import load_wine



# 1. Încarcă dataset-ul Wine
wine = load_wine()
X = wine.data
y = wine.target
# print(wine)



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

model = MLPClassifier(
    hidden_layer_sizes=(100,),    # un strat ascuns cu 100 neuroni
    activation='relu',           # funcția de activare ReLU
    alpha=0.0001,                # regularizare L2
    learning_rate='constant',    # rata de învățare constantă
    max_iter=500,                # număr maxim de epoci de antrenament
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred))

# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# #
# # Rulează căutarea
# grid_search.fit(X_train, y_train)
#
# #Afișează cei mai buni parametri și scorul
# print("Cei mai buni parametri găsiți:")
# print(grid_search.best_params_)
#
# print("\nAcuratețea pe setul de test cu cei mai buni parametri:")
# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"{accuracy:.4f}")