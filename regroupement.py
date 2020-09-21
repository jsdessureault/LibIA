import sys

from matplotlib import pyplot as plt
from sklearn import cluster

import LibIA.libIA


class Regroupement:
    def __init__(self, type=None, x_entrainement=None, y_entrainement=None, nb_categories=1, nb_donnees_min_groupe=3,
                 bavard=True):
        self.x_entrainement = x_entrainement
        self.y_entrainement = y_entrainement
        self.nb_categories = nb_categories
        self.nb_donnees_min_groupe = nb_donnees_min_groupe
        self.bavard = bavard
        self.type = type

        self.modele = None
        self.y_pred = None
        self.valide = False

    def construire_modele(self):
        if self.bavard:
            print("LibIA: Construction du modèle.")
        if self.valide:

            if self.type == libIA.LibIA.TYPE_K_MOYENNE:
                self.modele = cluster.KMeans(n_clusters=self.nb_categories)
            #elif self.type == libIA.LibIA.TYPE_OPTIQUE:
            #    self.modele = cluster.OPTICS(min_samples=self.nb_donnees_min_groupe)
            else:
                print("LibIA: * ERREUR 16 * le type de regroupement spécifié est invalide.")
                sys.exit()

    def entrainer_modele(self):
        if self.bavard:
            print("LibIA: Entraînement du modèle")
        if self.valide:
            self.y_pred = self.modele.fit_predict(self.x_entrainement)
            # print(self.y_pred)
        else:
            print("LibIA: * ERREUR 10 * Vous devez avoir creé un modèle valide avant de l'entraîner.")
            sys.exit()

    def afficher_donnees(self):
        if self.valide:
            plt.scatter(self.x_entrainement[:, 0], self.x_entrainement[:, 1], c=self.y_entrainement)
            plt.title('Nuage de points a regrouper')
            plt.show()
        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un modèle valide avant de l'afficher.")
            sys.exit()

    def afficher_resultas(self):
        if self.valide:
            plt.scatter(self.x_entrainement[:, 0], self.x_entrainement[:, 1], c=self.y_pred)
            plt.title('Prediction des regroupements')
            plt.show()
        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un modèle valide avant de l'afficher.")
            sys.exit()

    '''    
    def afficher_regroupement_comparaison(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(121)
        plt.scatter(self.x_entrainement[:, 0], self.x_entrainement[:, 1], c=self.y_entrainement)
        plt.title('Patitionnement initiale')

        plt.subplot(122)
        plt.scatter(self.x_entrainement[:, 0], self.x_entrainement[:, 1], c=self.y_pred)
        plt.title('Prediction des partitions')

        plt.show()
    '''

    def valider_modele(self):
        if self.bavard:
            print("LibIA: Validation du modèle")
        self.valide = True

    def fermer_modele(self):
        if self.bavard:
            print("LibIA: Fermeture du modèle")
        valide = False

    def afficher_modele(self):
        if self.bavard:
            print("LibIA: Affichage du modèle")
        if self.valide:
            print("LibIA: Architecture de l'apprentissage non supervisée:")
            if self.type == libIA.LibIA.TYPE_K_MOYENNE:
                print("LibIA: Type: k-moyenne")
            print("LibIA: Utilite:  Regroupement de valeurs semblables")
            print("LibIA: Nombre de categories: " + str(self.nb_categories))
        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un modèle valide avant de l'afficher.")
            sys.exit()
