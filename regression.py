import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


class Regression:

    def __init__(self, x_entrainement=None, y_entrainement=None, x_test=None, y_test=None, bavard=True,
                 taux_apprentissage=0.001, nb_apprentissages=50):
        self.x_entrainement = []
        self.y_entrainement = []
        self.x_entrainement = np.array(x_entrainement)
        self.y_entrainement = np.array(y_entrainement)
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)
        self.taux_apprentissage = taux_apprentissage
        self.nb_apprentissage = nb_apprentissages
        self.erreur = None
        self.bavard = bavard
        self.valide = False
        self.modele = None
        self.nb_variables = len(x_entrainement[0])
        self.y_pred = None
        self.x_pred = None

    def valider_modele(self):
        if self.bavard:
            print("LibIA: Validation du modèle.")
        valide = True
        return valide

    def afficher_modele(self):
        if self.valide:
            print("LibIA: Architecture de la régression:")
            print("LibIA: Type: Régression linéaire")
            print("LibIA: Utilité:  Prédiction ")
            print("LibIA: Nombre de variables " + str(self.nb_variables))
        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un modèle avant de l'afficher.")
            sys.exit()

    def construire_modele(self):
        if self.bavard:
            print("LibIA: Construction du modèle.")
        self.valide = self.valider_modele()
        if self.valide:
            self.modele = linear_model.LinearRegression()

    def entrainer_modele(self):
        if self.bavard:
            print("LibIA: Entraînement du modèle.")
        if self.valide:
            self.modele.fit(self.x_entrainement, self.y_entrainement)
            self.erreur = self.modele.score(self.x_entrainement, self.y_entrainement)
        else:
            print("LibIA: * ERREUR 10 * Vous devez avoir creé un modèle valide avant de l'entraîner.")
            sys.exit()

    def fermer_modele(self):
        if self.bavard:
            print("LibIA: Fermeture du modèle.")
        valide = False

    def predire_resultats(self, x_test=None, num=0):
        if self.bavard:
            print("LibIA: Prédiction des résultats.")
        x_test_initial = x_test
        if self.valide:
            if x_test is None:
                self.x_pred = [self.x_test[num]]
            else:
                self.x_pred = x_test
            self.x_pred = np.array(self.x_pred)
            self.y_pred = self.modele.predict(self.x_pred)
            print("LibIA: Donnee a predire_resultats: " + str(self.x_pred))
            print("LibIA: Prediction: " + str(self.y_pred))
            if x_test_initial is None:
                print("LibIA: La donnees correcte etait: " + str([self.y_test[num]]))
        else:
            print("LibIA: * ERREUR 15 * Vous devez avoir creé un modèle valide avant de faire des prédictions.")
            sys.exit()

    def afficher_resultats(self):
        if self.bavard:
            print("LibIA: Affichage des résultats.")
        if self.valide:

            le_min = min(self.x_test)
            le_max = max(self.x_test)

            x1 = [le_min]
            y1 = self.modele.predict(x1)
            x2 = [le_max]
            y2 = self.modele.predict(x2)

            plt.scatter(self.x_test, self.y_test, color='black')
            plt.plot([x1[0], x2[0]], [y1, y2], color='blue', linewidth=3)
            plt.scatter([x1, x2], [y1, y2], color='orange')
            plt.scatter([self.x_pred], [self.y_pred], marker='D', color='red')
            plt.show()
        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un modèle valide avant de l'afficher.")
            sys.exit()

    def afficher_erreur(self):
        if self.bavard:
            print("LibIA: Affichage de l'erreur.")
        print("LibIA: Erreur: " + str(self.erreur))

    def donner_erreur(self):
        return self.erreur
