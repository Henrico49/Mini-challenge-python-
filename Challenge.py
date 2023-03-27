import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

##### Lecture du dataset #####
df = pd.read_csv("mobile_train.csv")

##### Récupération des labels #####
features = df.columns[:-1]

########## On regarde la répartition des classes ##########

"""
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(df[features].values)
fig = px.scatter(X_embedded, x=0, y=1, color=df['price_range'])
fig.show()
"""

##### préparation dataframe d'entrainement #####
digits_train = df.sample(frac=0.8)
digits_valid = df.drop(digits_train.index)

X = df[features]
Y = df['price_range']

X_train = digits_train[features]
Y_train = digits_train['price_range']

X_valid = digits_valid[features]
Y_valid = digits_valid['price_range']

########## recherche de l'importance de chaque labels lors de la prédiction ##########

def importanceLabel(liste_meilleur_label):
    # Diviser les données en ensemble d'entraînement et ensemble de validation
    X_train, X_valid, Y_train, Y_valid = train_test_split(df[features], df['price_range'], test_size=0.2, random_state=42)

    # Créer un arbre de décision
    tree = DecisionTreeClassifier(random_state=42)

    # Entraîner l'arbre de décision sur les données d'entraînement
    tree.fit(X_train, Y_train)

    # Obtenir les importances des features
    importances = tree.feature_importances_

    # Trier les labels par ordre d'importance décroissante
    indices = np.argsort(importances)[::-1]

    # Créer un DataFrame contenant les labels et leurs importances
    df_importances = pd.DataFrame({'feature': X_train.columns[indices], 'importance': importances[indices]})
    for i in range(0, 4):
        liste_meilleur_label.append(df_importances['feature'][i])
    # Créer un graphique à barres pour visualiser les importances des features
    fig = px.bar(df_importances, x='importance', y='feature', orientation='h', height=500, title='Importance des features')
    fig.show()
liste_meilleur_label = []
importanceLabel(liste_meilleur_label)

##### Normalisation #####
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_trainN = scaler.fit_transform(X_train)
X_validN = scaler.fit_transform(X_valid)
XN = scaler.fit_transform(X[liste_meilleur_label])



##### modèle KNN vu en cours #####

########## On définit la fonction de distance ##########

def distance_euclidienne(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


########## On définit la fonction de prédiction ##########

def neighbors(X_train, y_label, x_test, k):
    list_distances = []
    for i in range(np.shape(X_train)[0]):
        list_distances.append(distance_euclidienne(X_train.iloc[i], x_test))
    dataf = pd.DataFrame()
    dataf["label"] = y_label
    dataf["distance"] = list_distances
    dataf = dataf.sort_values(by="distance")
    return dataf.iloc[:k, :]


def predict(neighbors):
    return neighbors["label"].value_counts().idxmax()


def evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=True):
    TP = 0  # vrai
    FP = 0  # faux
    total = 0
    for i in range(X_valid.shape[0]):
        nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[i], k)
        if predict(nearest_neighbors) == Y_valid.iloc[i]:
            TP += 1
        else:
            FP += 1
        total += 1
    accuracy = TP / total
    if verbose:
        print("Accuracy:" + str(accuracy))
    return accuracy


##### Définir le meilleur nombre de voisin avec la bibliothèque KNeighborsClassifier de sklearn.neighbors #####

def meilleurNbrVoisin(df, n_iterations,nbrVoisinMax):

    # Stocker les précisions de chaque itération dans une liste
    all_accuracies = []
    for i in range(n_iterations):
        digits_train = df.sample(frac=0.8)
        digits_valid = df.drop(digits_train.index)

        X_train = digits_train[features]
        Y_train = digits_train['price_range']
        X_valid = digits_valid[features]
        Y_valid = digits_valid['price_range']
        list_accuracy = []
        for k in range(1, nbrVoisinMax):
            knn = KNeighborsClassifier(n_neighbors=k)

            # Entraîner le modèle sur les données d'entraînement
            knn.fit(X_train, Y_train)

            # Prédire les étiquettes pour les données de validation
            Y_pred = knn.predict(X_valid)
            accuracy = sum(Y_pred == Y_valid) / len(Y_valid)
            list_accuracy.append(accuracy)
        all_accuracies.append(list_accuracy)

    # Calculer la moyenne et l'écart-type de toutes les précisions pour chaque valeur de k
    mean_accuracies = np.mean(all_accuracies, axis=0)
    std_accuracies = np.std(all_accuracies, axis=0)

    # Tracer la courbe de précision moyenne avec des barres d'erreur
    plt.plot(range(1, nbrVoisinMax), mean_accuracies)
    plt.xlabel('k')
    plt.ylabel('Précision')
    plt.show()

########## test sur le vrai modèle ##########
def testModeleKNN(df,nbrVoisin):
    df2 = pd.read_csv("mobile_test_data.csv")

    knn1 = KNeighborsClassifier(n_neighbors=nbrVoisin)
    X = df2[features]
    # Entraîner le modèle sur les données d'entraînement
    knn1.fit(df[features], df['price_range'])

    # Prédire les étiquettes pour les données de validation
    Y_pred = knn1.predict(X)
    print(Y_pred)
    np.savetxt("testFullcolonne.csv", Y_pred, delimiter="\n")


########## recherche des meilleurs hyperparamètres ##########


def rechercheMeilleurHyperparametre():
    # Définir les valeurs des hyperparamètres à tester
    param_grid = {
        'hidden_layer_sizes': [(10,10,10,10,10,10,10,10,10,10,10,10),(20,20,20,20,20,20,20,20),
                               (50,50,50,50,50),(150),(10,10,10),(30,30,30),(10,10,10,10,10),(10,10),(30,30)],
        'alpha': [0.01,0.001,0.0001, 0.00001, 0.000001],
        'learning_rate': ['constant', 'invscaling', 'adaptive', 'learning_rate_init']

    }

    # Créer un objet GridSearchCV pour tester toutes les combinaisons des hyperparamètres
    grid_search = GridSearchCV(
        MLPClassifier(solver='adam', learning_rate='adaptive', max_iter=2000),
        param_grid,
        cv=7, # nombre de folds de la validation croisée
        n_jobs=-1, # utiliser tous les coeurs du processeur
        verbose=2, # afficher les détails de chaque itération
    )

    # Exécuter la recherche des meilleurs hyperparamètres sur les données d'entraînement
    grid_search.fit(XN, Y)

    # Afficher les meilleurs hyperparamètres et la précision correspondante
    print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
    print(f"Précision correspondante : {grid_search.best_score_}")

#rechercheMeilleurHyperparametre()

########## Test avec réseaux de neurone #########

def evaluation2(X_validN, Y_valid):
    clf = MLPClassifier(solver='adam', alpha=1e-05, hidden_layer_sizes=(10, 10, 10), learning_rate='adaptive',
                        max_iter=2000)
    clf.fit(X_trainN, Y_train)
    accuracy = clf.score(X_validN, Y_valid)
    print("Accuracy:", accuracy)


evaluation2(X_validN, Y_valid)


########## test sur le vrai modèle ##########
def testModeleMLP():
    df2 = pd.read_csv("mobile_test_data.csv")

    clf = MLPClassifier(solver='adam', alpha= 1e-05, hidden_layer_sizes= (10, 10, 10), learning_rate= 'adaptive', max_iter=2000)
    X2 = df2[liste_meilleur_label]
    X2N = scaler.fit_transform(X2[liste_meilleur_label])
    # Entraîner le modèle sur les données d'entraînement
    clf.fit(XN, Y)

    # Prédire les étiquettes pour les données de validation
    Y_pred = clf.predict(X2N)
    print(Y_pred)
    np.savetxt("testFullcolonne.csv", Y_pred, delimiter="\n")

#testModeleMLP()