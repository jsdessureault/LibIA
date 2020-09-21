import sys

import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt

import donnees


class Perceptron:

    def __init__(self, x_entrainement=None, y_entrainement=None, x_test=None, y_test=None, bavard=True,
                 taux_apprentissage=0.001, nb_neurones_entrees=None, nb_neurones_sorties=None, nb_couches_cachees=None,
                 nb_neurones_cachees=None, nb_apprentissages=50, but="classification", verbose=True):
        self.x_entrainement = []
        self.y_entrainement = []
        self.x_entrainement = x_entrainement
        self.y_entrainement = y_entrainement
        self.x_test = x_test
        self.y_test = y_test
        self.nb_neurones_entrees = nb_neurones_entrees
        self.nb_neurones_sorties = nb_neurones_sorties
        self.nb_couches_cachees = nb_couches_cachees
        self.nb_neurones_cachees = nb_neurones_cachees
        self.taux_apprentissage = taux_apprentissage
        self.nb_neurones_entrees = nb_neurones_entrees
        self.nb_apprentissage = nb_apprentissages
        self.but = but
        self.histoire = None
        self.classes = None
        self.modele = None
        self.pointage = None
        self.erreur = None
        self.bavard = bavard
        self.valide = False
        self.y_pred = None
        self.x_pred = None
        self.verbose = verbose

    def valider_modele(self):
        valide = True
        if self.bavard:
            print("LibIA: Validation du modèle.")
        if self.nb_neurones_entrees != len(self.x_entrainement[0]):
            print(
                "LibIA: * ERREUR 14 * Le nombre de neurones d'entrée doit être identique au nombre de variables d'entrée.")
            sys.exit()
            valide = False
        return valide

    def afficher_modele(self):
        if self.bavard:
            print("LibIA: Affichage du modèle.")
        if self.valide:
            print("LibIA: Architecture du reseau de neurones:")
            print("LibIA: Type: Perceptron multicouche")
            print("LibIA: Utilite:  Classificatieur ")
            print("LibIA: Nombre de neurones d'entree: " + str(self.nb_neurones_entrees))
            print("LibIA: Nombre de neurones de sortie: " + str(self.nb_neurones_sorties))
            print("LibIA: Nombre de couches cachees: " + str(self.nb_couches_cachees))
            print("LibIA: Nombre de neurones dans les couches cachees: " + str(self.nb_neurones_cachees))
            # print("Fiche technique du perceptron:")
            # print(self.modele.summary())
            # print("Un fichier Perceptron.png a été crée.")
            # plot_model(self.modele, to_file='Perceptron.png', show_shapes=True, show_layer_names=True)
            arch = []
            arch.append(self.nb_neurones_entrees)
            for i in range(self.nb_couches_cachees):
                arch.append(self.nb_neurones_cachees)
            arch.append(self.nb_neurones_sorties)
            self.afficher_reseau_neurones(.1, .9, .1, .9, arch)

        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un modèle valide avant de l'afficher.")
            sys.exit()

    def construire_modele(self):
        if self.bavard:
            print("LibIA: Construction du modèle.")
        self.valide = self.valider_modele()
        if self.valide:
            self.modele = Sequential()
            self.modele.add(Dense(self.nb_neurones_cachees, input_dim=self.nb_neurones_entrees, activation='relu'))
            for i in range(self.nb_couches_cachees):
                self.modele.add(Dense(self.nb_neurones_cachees, activation='relu'))
            self.modele.add(Dense(self.nb_neurones_sorties, activation='sigmoid'))
            #opti = optimizers.Adam(lr=self.taux_apprentissage)
            opti = optimizers.Adam()
            self.modele.compile(loss='binary_crossentropy', optimizer=opti, metrics=['mse'])

    def entrainer_modele(self):
        if self.bavard:
            print("LibIA: Entraînement du modèle.")
        if self.valide:
            self.histoire = self.modele.fit(np.array(self.x_entrainement),
                                            np.array(self.y_entrainement),
                                            epochs=self.nb_apprentissage,
                                            batch_size=len(self.x_entrainement),
                                            verbose=2)
            self.pointage = self.modele.evaluate(np.array(self.x_entrainement), np.array(self.y_entrainement))
            self.erreur = self.pointage[1] * 100
            print(self.modele.metrics_names[1], str(self.erreur) + "%")
        else:
            print("LibIA: * ERREUR 10 * Vous devez avoir creé un modèle valide avant de l'entraîner.")
            sys.exit()

    def predire_resultats(self, x_test=None, encodage=None, num=None):
        if self.bavard:
            print("LibIA: Prédiction des résultats.")
        if self.valide:
            if x_test is None:
                if num is not None:
                    x_test = [self.x_test[num]]
                else:
                    x_test = self.x_test
            else:
                x_test = np.array(x_test)
            self.x_pred = np.array(x_test)
            self.y_pred = self.modele.predict(self.x_pred, batch_size=1)
            print("LibIA: Resultat dans le neurone de sortie: ")
            if encodage == "1parmisN":
                print(str(donnees.Donnees.decoder_un_parmi_N(donnees, self.y_pred)))
                return donnees.Donnees.decoder_un_parmi_N(donnees, self.y_pred)
            else:
                print(str(self.y_pred))
                return self.y_pred
        else:
            print("LibIA: * ERREUR 15 * Vous devez avoir creé un modèle valide avant de faire des prédictions.")
            sys.exit()
        return None

    def afficher_erreur(self):
        if self.bavard:
            print("LibIA: Affichage de l'erreur.")
        if self.valide:
            # plt.plot(self.histoire.history['mse'])
            plt.plot(self.histoire.history['mse'])
            plt.show()
        else:
            print("LibIA: * ERREUR 12 * Vous devez avoir creé un modèle valide avant de d'afficher l'erreur.")
            sys.exit()

    def donner_erreur(self):
        return self.erreur

    def fermer_modele(self):
        if self.bavard:
            print("LibIA: Fermeture du modele")
        valide = False

    def afficher_reseau_neurones(self, left, right, bottom, top, layer_sizes):
        # https://gist.github.com/craffel/2d727968c3aaebd10359

        print("LibIA: Affichage de l'architecture du réseau de neurones...")
        print("LibIA: Attention! Ce traitement peut etre long pour la gros réseaux de neurones.")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.gca()
        ax.axis('off')

        n_layers = len(layer_sizes)
        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(len(layer_sizes) - 1)
        # Nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size):
                circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                    color='w', ec='k', zorder=4)
                ax.add_artist(circle)
        # Edges
        for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
            layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
            for m in range(layer_size_a):
                for o in range(layer_size_b):
                    line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                      [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='k')
                    ax.add_artist(line)
        plt.show()
