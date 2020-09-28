# LibIA: Une librairie francophone pour l'apprentissage de l'intelligence artificielle.
# Par Jean-Sebastien Dessureault
# Et Jonathan Simard

import sys

import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import random

import gitLibIA.chronometre as chrono
import gitLibIA.perceptron
import gitLibIA.regression
import gitLibIA.regroupement
import gitLibIA.apprentissage_q
from gitLibIA.donnees import Donnees as donnees

import datetime


class Chronometre:

    def __init__(self, bavard=True):
        self.bavard = bavard
        self.temps_depart = None
        self.temps_arret = None

    def demarrer(self):
        self.temps_depart = datetime.datetime.now()
        if self.bavard:
            print("Chronometre demarré à : " + str(self.temps_depart))

    def arreter(self):
        self.temps_arret = datetime.datetime.now()
        duree = self.temps_arret - self.temps_depart
        if self.bavard:
            print("Chronometre arrêté à : " + str(self.temps_arret))
            print("Temps calculé: " + str(duree))
        return duree

    def donner_temps(self):
        return self.temps_arret - self.temps_depart

class LibIA:

    Vrai = True
    Faux = False
    Aucun = None

    bavard = True

    donnees = None
    jeux_de_donnees = None
    chrono = None

    ia = None
    jeu = None

    TYPE_PERCEPTRON = 0
    TYPE_REGRESSION = 1
    TYPE_APPRENTISSAGE_Q = 2
    TYPE_K_MOYENNE = 3

    # Jeux
    JEU_PACMAN = 0
    JEU_SPACE_INVADERS = 1
    JEU_ATLANTIS = 4
    JEU_BATTLEZONE = 5
    JEU_QUILLES = 6
    JEU_BOXE = 7
    JEU_BREAKOUT = 8
    JEU_ENDURO = 9
    JEU_HOCKEY = 10
    JEU_PITFALL = 11
    JEU_PONG = 12
    JEU_SKI = 13
    JEU_TENNIS = 14
    JEU_PINBALL = 15

    def __init__(self, bavard=True):
        self.bavard = bavard
        self.chrono = Chronometre(bavard)
        self.donnees = Donnees()

        if self.bavard:
            print("LibIA: Création de LibIA.")

    def creer_ia(self, type=None, taux_apprentissage=0.001, nb_neurones_entrees=2, nb_neurones_sorties=1, nb_couches_cachees=1, nb_neurones_cachees=2, nb_apprentissages=50, jeu=0, nb_categories=3, nb_donnees_min_groupe=3):
        try:
            if type == self.TYPE_PERCEPTRON:
                if self.bavard:
                    print("LibIA: Création d'un IA de type Perceptron.")
                self.ia = perceptron.Perceptron(x_entrainement=self.donnees.x_entrainement,
                                                y_entrainement=self.donnees.y_entrainement,
                                                x_test=self.donnees.x_test,
                                                y_test=self.donnees.y_test,
                                                taux_apprentissage=taux_apprentissage,
                                                nb_neurones_entrees=nb_neurones_entrees,
                                                nb_neurones_sorties=nb_neurones_sorties,
                                                nb_couches_cachees=nb_couches_cachees,
                                                nb_neurones_cachees=nb_neurones_cachees,
                                                nb_apprentissages=nb_apprentissages,
                                                bavard=self.bavard)
            elif type == self.TYPE_REGRESSION:
                if self.bavard:
                    print("LibIA: Création d'un IA de type Régression.")
                self.ia = regression.Regression(x_entrainement=self.donnees.x_entrainement,
                                                y_entrainement=self.donnees.y_entrainement,
                                                x_test=self.donnees.x_test,
                                                y_test=self.donnees.y_test,
                                                taux_apprentissage=taux_apprentissage,
                                                nb_apprentissages=nb_apprentissages,
                                                bavard=self.bavard)
            elif type == self.TYPE_APPRENTISSAGE_Q:
                if self.bavard:
                    print("LibIA: Création d'un IA de type Apprentissage Q.")
                self.ia = apprentissage_q.Apprentissage_Q(
                                                nb_apprentissages=nb_apprentissages,
                                                jeu=jeu,
                                                bavard=self.bavard)
            elif type == self.TYPE_K_MOYENNE:
                if self.bavard:
                    print("LibIA: Création d'un IA de type Regroupement K Moyenne.")
                self.ia = regroupement.Regroupement(
                                                type=self.TYPE_K_MOYENNE,
                                                x_entrainement=self.donnees.x_entrainement,
                                                y_entrainement=self.donnees.y_entrainement,
                                                nb_categories=nb_categories,
                                                bavard=self.bavard)
            else:
                print("LibIA: * ERREUR 1: * Le type d'IA spécifié est invalide. Vérifiez le paramètre 'type'.")
        except:
            print("LibIA: * ERREUR 2: * Une erreur est survenue lors de l'appel de 'créez IA'.  Vérifier les paramètres.")
            sys.exit()

    def entree_donnees(self):
        texte = input()
        return texte

    def activer_bavard(self, bavard=True):
        self.bavard = bavard
        if self.bavard:
            print("LibIA: Mode bavard actif.")

    def tirage_au_sort(self, minimum=0, maximum=10):
        return random.randint(minimum,maximum)


    def charger_donnees(self, jeux_de_donnees,  nb_nuages=10, nb_points_total=100, ecart_type=5):
        try:
            self.jeux_de_donnees = jeux_de_donnees
            self.donnees.charger_donnees(self.jeux_de_donnees, nb_nuages=nb_nuages, nb_points_total=nb_points_total, ecart_type=ecart_type, bavard=self.bavard)
        except:
            print("LibIA: * ERREUR 3: * Une erreur est survenue lors de l'appel de 'charger_donnees'.  Vérifier les paramètres.")
            sys.exit()

    def afficher_entrees_entrainement(self):
      print("Données en entrée (entraînement) (x)")
      print(self.donnees.x_entrainement)
      print("format: " + str(self.donnees.x_entrainement.shape))

    def afficher_sorties_entrainement(self):
      print("Données en sortie (entraînement) (y)")
      print(self.donnees.y_entrainement)
      print("format: " + str(self.donnees.y_entrainement.shape))

    def afficher_entrees_test(self):
      print("Données en entrée (test) (x)")
      print(self.donnees.x_test)
      print("format: " + str(self.donnees.x_test.shape))

    def afficher_sorties_test(self):
      print("Données en sortie (test) (y)")
      print(self.donnees.y_test)
      print("format: " + str(self.donnees.y_test.shape))

    def encoder_un_parmi_N(self, donnee):
        return to_categorical(donnee)

    def decoder_un_parmi_N(self, donnee):
        return np.argmax(donnee)

    def donner_en_clair(self, donnee):
        return np.around(donnee, decimals=2)

    def afficher_donnees(self, donnee):
        print(str(donnee))

    def afficher_division(self):
        print("--------------------------------------------------")

    def convertir_caractere(self, donnee):
        return str(donnee)

    def convertir_entier(self, donnee):
        return int(donnee)

    def convertir_decimal(self, donnee):
        return float(donnee)

    def afficher_jeux_donnees(self):
        self.donnees.afficher_jeux_donnees()

    def afficher_image2d(self, source="entrainement", axe="X", indice=0, tailleX=28, tailleY=28):
        try:
            if source is "entrainement":
                if axe is "X":
                    image = self.donnees.x_entrainement[indice]
                elif axe is "Y":
                    image = self.donnees.y_entrainement[indice]
                else:
                    print("LibIA: * ERREUR 4: * Une erreur est survenue lors de l'appel de 'afficher_image2d'.  L'axe spécifiée est invalide.")
                    sys.exit()
            elif source is "test":
                if axe is "X":
                    image = self.donnees.x_test[indice]
                elif axe is "Y":
                    image = self.donnees.y_test[indice]
                else:
                    print("LibIA: * ERREUR 4: * Une erreur est survenue lors de l'appel de 'afficher_image2d'.  L'axe spécifiée est invalide.")
                    sys.exit()
            else:
                print("LibIA: * ERREUR 5: * Une erreur est survenue lors de l'appel de 'afficher_image2d'.  La source spécifiée est invalide.")
                sys.exit()

            image = np.array(image.reshape((tailleX, tailleY)))
            fig, ax = plt.subplots()
            plt.title("Source: " + source + " Axe: " + axe + " Indice: " + str(indice))
            im = ax.imshow(image)
            plt.show()
        except:
            print("LibIA: * ERREUR 6: * Une erreur est survenue lors de l'appel de 'afficher_image2d'.  Vérifier les paramètres et opérations précédentes préalables.")
            sys.exit()



    def afficher_donnees_sources(self, source="entrainement", axe="X", indice=0):
        try:

            if source is "entrainement":
                if axe is "X":
                    donnee = self.donnees.x_entrainement[indice]
                elif axe is "Y":
                    donnee = self.donnees.y_entrainement[indice]
                else:
                    print("LibIA: * ERREUR 7: * Une erreur est survenue lors de l'appel de 'afficher_donnees_sources'.  L'axe spécifiée est invalide.")
                    sys.exit()
            elif source is "test":
                if axe is "X":
                    donnee = self.donnees.x_test[indice]
                elif axe is "Y":
                    donnee = self.donnees.y_test[indice]
                else:
                    print("LibIA: * ERREUR 7: * Une erreur est survenue lors de l'appel de 'afficher_donnees_sources'.  L'axe spécifiée est invalide.")
                    sys.exit()
            else:
                print("LibIA: * ERREUR 8: * Une erreur est survenue lors de l'appel de 'afficher_donnees_sources'.  La source spécifiée est invalide.")
                sys.exit()

            print("Source: " + source + " Axe: " + axe + " Indice: " + str(indice))
            print(donnee)

        except:
            print("LibIA: * ERREUR 9: * Une erreur est survenue lors de l'appel de 'afficher_donnees_sources'.  Vérifier les paramètres et opérations précédentes préalables.")
            sys.exit()

