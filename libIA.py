# LibIA: Une librairie francophone pour l'apprentissage de l'intelligence artificielle.
# Par Jean-Sebastien Dessureault
# Et Jonathan Simard

# Classe DONNÉES ---------------------------------------------------------------------------------------------------------------------------------------------------
#Données
import random
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn import datasets
from sklearn import model_selection
import pandas

class Donnees:
    jeux_de_donnees = []
    x_entrainement = []
    y_entrainement = []
    x_test = []
    y_test = []

    # Constantes
    XOU = 0
    ET = 1
    OU = 2
    NON = 3
    LNH = 4
    CHIFFRES_MANUSCRITS = 5
    MAISONS_BOSTON = 6
    PIMA = 7
    CANCER_SEIN = 8
    NUAGES_POINTS = 9
    MODE = 10
    NHL = 11
    MLB = 12

    def __init__(self):
        self.nb_nuages = None
        self.nb_points_total = None
        self.ecart_type = None
        self.initialiser_jeux_de_donnees()

    def encoder_un_parmi_N(self, donnee):
        return to_categorical(donnee)

    def decoder_un_parmi_N(self, donnee):
        return np.argmax(donnee)

    def charger_donnees(self, jeux_de_donnees, nb_nuages=5, nb_points_total=1000, ecart_type=5, bavard=True):

        self.nb_nuages = nb_nuages
        self.nb_points_total = nb_points_total
        self.ecart_type = ecart_type
        self.bavard = bavard

        if jeux_de_donnees == self.XOU:
            self.charger_XOU()
        if jeux_de_donnees == self.ET:
            self.charger_ET()
        if jeux_de_donnees == self.OU:
            self.charger_OU()
        if jeux_de_donnees == self.NON:
            self.charger_NON()
        if jeux_de_donnees == self.CHIFFRES_MANUSCRITS:
            self.charger_CHIFFRES()
        if jeux_de_donnees == self.MAISONS_BOSTON:
            self.charger_BOSTON()
        if jeux_de_donnees == self.PIMA:
            self.charger_PIMA()
        if jeux_de_donnees == self.CANCER_SEIN:
            self.charger_CANCER_SEIN()
        if jeux_de_donnees == self.NUAGES_POINTS:
            self.charger_NUAGES_POINTS()
        if jeux_de_donnees == self.MODE:
            self.charger_MODE()
        if jeux_de_donnees == self.NHL:
            self.charger_NHL()
        if jeux_de_donnees == self.MLB:
            self.charger_MLB()

    def initialiser_jeux_de_donnees(self):
        self.jeux_de_donnees.append([self.XOU, "Table de verite: Ou exclusif"])
        self.jeux_de_donnees.append([self.ET, "Table de verite: ET"])
        self.jeux_de_donnees.append([self.OU, "Table de verite: OU"])
        self.jeux_de_donnees.append([self.NON, "Table de verite: NON"])
        self.jeux_de_donnees.append([self.CHIFFRES_MANUSCRITS, "Chiffres ecrits a la main"])
        self.jeux_de_donnees.append([self.MAISONS_BOSTON, "Chiffres ecrits a la main"])
        self.jeux_de_donnees.append([self.PIMA, "Communautes natives Pima et diabete"])
        self.jeux_de_donnees.append([self.CANCER_SEIN, "Cancer du sein"])
        self.jeux_de_donnees.append([self.NUAGES_POINTS, "Nuages de points"])
        self.jeux_de_donnees.append([self.MODE, "Vetements de mode"])

    def afficher_jeux_donnees(self):
        print("LibIA: Ensembles de données disponibles:")
        for i in range(len(self.jeux_de_donnees)):
            print(self.jeux_de_donnees[i])

    def charger_XOU(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données XOU.")
        self.x_entrainement = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_entrainement = np.array([[0], [1], [1], [0]])
        self.x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_test = np.array([[0], [1], [1], [0]])

    def charger_ET(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données ET.")
        self.x_entrainement = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_entrainement = np.array([[0], [0], [0], [1]])
        self.x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_test = np.array([[0], [0], [0], [1]])

    def charger_OU(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données OU.")
        self.x_entrainement = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_entrainement = np.array([[0], [1], [1], [1]])
        self.x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_test = np.array([[0], [1], [1], [1]])

    def charger_NON(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données NON.")
        self.x_entrainement = np.array([[0], [1]])
        self.y_entrainement = np.array([[1], [0]])
        self.x_test = np.array([[0], [1]])
        self.y_test = np.array([[1], [0]])

    def charger_CHIFFRES(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données CHIFFRES.")
        (x_ent_tmp, y_ent_tmp), (x_test_tmp, y_test_tmp) = mnist.load_data()
        # Traitement des X
        self.x_entrainement = []
        self.x_test = []
        for i in range(len(x_ent_tmp)):
            self.x_entrainement.append(np.array(x_ent_tmp[i]).flatten())
        for i in range(len(x_test_tmp)):
            self.x_test.append(np.array(x_test_tmp[i]).flatten())
        self.x_entrainement = np.array(self.x_entrainement)
        self.x_test = np.array(self.x_test)
        # Traitement des Y
        self.y_entrainement = []
        self.y_test = []
        self.y_entrainement = self.encoder_un_parmi_N(y_ent_tmp)
        self.y_test = self.encoder_un_parmi_N(y_test_tmp)

    def charger_MODE(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données MODE.")
        (x_ent_tmp, y_ent_tmp), (x_test_tmp, y_test_tmp) = fashion_mnist.load_data()
        # Traitement des X
        self.x_entrainement = []
        self.x_test = []
        for i in range(len(x_ent_tmp)):
            self.x_entrainement.append(np.array(x_ent_tmp[i]).flatten())
        for i in range(len(x_test_tmp)):
            self.x_test.append(np.array(x_test_tmp[i]).flatten())
        self.x_entrainement = np.array(self.x_entrainement)
        self.x_test = np.array(self.x_test)
        # Traitement des Y
        self.y_entrainement = []
        self.y_test = []
        self.y_entrainement = self.encoder_un_parmi_N(y_ent_tmp)
        self.y_test = self.encoder_un_parmi_N(y_test_tmp)

    def charger_PIMA(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données PIMA.")
        x_ent_tmp, y_ent_tmp = datasets.load_diabetes(return_X_y=True)
        self.x_entrainement = []
        self.y_entrainement = []
        self.x_test = []
        self.y_test = []
        for i in range(len(x_ent_tmp) - 10):
            self.x_entrainement.append([x_ent_tmp[i][0], x_ent_tmp[i][1], x_ent_tmp[i][2], x_ent_tmp[i][3], x_ent_tmp[i][4], x_ent_tmp[i][5], x_ent_tmp[i][7], x_ent_tmp[i][7]])
            self.y_entrainement.append(y_ent_tmp[i])
        for i in range(len(x_ent_tmp) - 10, len(x_ent_tmp)):
            self.x_test.append([x_ent_tmp[i][0], x_ent_tmp[i][1], x_ent_tmp[i][2], x_ent_tmp[i][3], x_ent_tmp[i][4], x_ent_tmp[i][5], x_ent_tmp[i][7], x_ent_tmp[i][7]])
            self.y_test.append(y_ent_tmp[i])
        self.x_entrainement = np.array(self.x_entrainement)
        self.y_entrainement = np.array(self.y_entrainement)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

    def charger_BOSTON(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données BOSTON.")
        x_ent_tmp, y_ent_tmp = datasets.load_boston(return_X_y=True)
        self.x_entrainement = []
        self.y_entrainement = []
        self.x_test = []
        self.y_test = []
        # Extraction du nombre de chambre SEULEMENT (pour une seule dimension)
        for i in range(len(x_ent_tmp) - 10):
            self.x_entrainement.append([x_ent_tmp[i][5]])
            self.y_entrainement.append(y_ent_tmp[i])
        for i in range(len(x_ent_tmp) - 10, len(x_ent_tmp)):
            self.x_test.append([x_ent_tmp[i][5]])
            self.y_test.append(y_ent_tmp[i])
        self.x_entrainement = np.array(self.x_entrainement)
        self.y_entrainement = np.array(self.y_entrainement)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

    def charger_CANCER_SEIN(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données CANCER_SEIN.")
        data, target = datasets.load_breast_cancer(return_X_y=True)
        self.x_entrainement, self.x_test, self.y_entrainement, self.y_test = model_selection.train_test_split(data,
                                                                                                              target,
                                                                                                              test_size=0.10)

    def charger_NUAGES_POINTS(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données NUAGES_POINTS.")
        min = -100
        max = 100
        centres = []
        for i in range(self.nb_nuages):
            x = random.randint(min, max)
            y = random.randint(min, max)
            centres.append((x, y))
        self.x_entrainement, self.y_entrainement = datasets.make_blobs(n_samples=self.nb_points_total,
                                                                       center_box=(min, max),
                                                                       centers=centres,
                                                                       cluster_std=self.ecart_type)

    def charger_NHL(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données NHL.")
        url = 'https://raw.githubusercontent.com/jsdessureault/dataFilesForAI/master/NHL.csv'
        nhl = pandas.read_csv(url)
        nhl = np.array(nhl)
        random.shuffle(nhl)

        self.x_entrainement = []
        self.y_entrainement = []
        self.x_test = []
        self.y_test = []

        for i in range(len(nhl) - 10):
            self.x_entrainement.append([nhl[i][4], nhl[i][5]])
            self.y_entrainement.append(nhl[i][6])
        for i in range(len(nhl) - 10, len(nhl)):
            self.x_test.append([nhl[i][4], nhl[i][5]])
            self.y_test.append(nhl[i][6])
        self.x_entrainement = np.array(self.x_entrainement)
        self.y_entrainement = np.array(self.y_entrainement)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)


    def charger_MLB(self):
        if self.bavard:
            print("LibIA: Chargement de l'ensemble de données MLB.")
        url = 'https://raw.githubusercontent.com/jsdessureault/dataFilesForAI/master/MLB.csv'
        mlb = pandas.read_csv(url)
        mlb = np.array(mlb)
        random.shuffle(mlb)

        self.x_entrainement = []
        self.y_entrainement = []
        self.x_test = []
        self.y_test = []

        for i in range(len(mlb) - 10):
            self.x_entrainement.append([mlb[i][5], mlb[i][7], mlb[i][8], mlb[i][9], mlb[i][10]])
            self.y_entrainement.append(mlb[i][21])  # Moyenne 19, moyenne puissance 21
        for i in range(len(mlb) - 10, len(mlb)):
            self.x_test.append([mlb[i][5], mlb[i][7], mlb[i][8], mlb[i][9], mlb[i][10]])
            self.y_test.append(mlb[i][21])   # Moyenne 19, moyenne puissance 21
        self.x_entrainement = np.array(self.x_entrainement)
        self.y_entrainement = np.array(self.y_entrainement)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

# Classe CHRNONMÈTRE ---------------------------------------------------------------------------------------------------------------------------------------------------
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


# Classe APPRENTISSAGE_Q --------------------------------------------------------------------------------------------------------------------------------------------------
    
# https://gym.openai.com/
# https://gist.github.com/syuntoku14/b9527403697a6565237ff5a403517db2
# https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/
# https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
import base64
# %matplotlib inline
import glob
import io
import sys

import gym
import tensorflow.compat.v1 as tf
from IPython import display as ipythondisplay
from IPython.display import HTML
from gym import logger as gymlogger
from gym.wrappers import Monitor
tf.disable_v2_behavior()

class Apprentissage_Q:
            
    def __init__(self, nb_apprentissages=50, jeu=0, bavard=True):
        self.nb_apprentissage = nb_apprentissages
        self.jeu = jeu

        self.env = None
        self.observation = None
        self.action = None
        self.reward = None
        self.done = None
        self.info = None
        self.bavard = bavard

        self.valide = False

    def construire_modele(self):
        if self.bavard:
            print("LibIA: Construction du modèle.")

        self.valide = True
        gymlogger.set_level(40)  # error only
        self.env = self.convertir_env(gym.make(self.jeu_chaine(self.jeu)))
        print(self.env.action_space)
        self.observation = self.env.reset()

    def entrainer_modele(self):
        if self.bavard:
            print("LibIA: Entraînement du modèle.")
        if self.valide:
            while True:
                self.env.render()
                self.action = self.env.action_space.sample()
                self.observation, self.reward, self.done, self.info = self.env.step(self.action)
                if self.done:
                    break;
        else:
            print("LibIA: * ERREUR 10 * Vous devez avoir creé un réseau de neurones valide avant de l'entraîner.")
            sys.exit()

    def fermer_modele(self):
        if self.bavard:
            print("LibIA: Fermeture du modèle.")
        self.env.close()
        self.valide = False

    def valider_modele(self):
        if self.bavard:
            print("LibIA: Validation du modèle.")
        valide = True
        return valide

    def afficher_modele(self):
        if self.valide:
            print("LibIA: Architecture de l'apprentissage par renforcement. :")
            print("LibIA: Type: l'apprentissage Q")
            print(
                "LibIA: Utilité:  Apprentissage lorsqu'il n'y a pas de donnée à apprendre.  Récompenses et punitions lors de simulation. ")
        else:
            print("LibIA: * ERREUR 11 * Vous devez avoir creé un réseau de neurones valide avant de l'afficher.")
            sys.exit()

    def afficher_erreur(self):
        if self.valide:
            print("rien")
        else:
            print(
                "LibIA: * ERREUR 12 * Vous devez avoir creé un réseau de neurones valide avant de d'afficher l'erreur.")
            sys.exit()

    def donner_erreur(self):
        return self.erreur

    def afficher_resultat(self):
        if self.bavard:
            print("LibIA: Affichage du résultat.")
        self.montrer_video()

    def montrer_video(self):
        mp4list = glob.glob('video/*.mp4')
        if len(mp4list) > 0:
            mp4 = mp4list[0]
            video = io.open(mp4, 'r+b').read()
            encoded = base64.b64encode(video)
            ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                 </video>'''.format(encoded.decode('ascii'))))
        else:
            print("LibIA: * ERREUR 13 * Aucune vidéo ne semble avoir été générée.")
            sys.exit()

    def convertir_env(self, env):
        self.env = Monitor(env, './video', force=True)
        return self.env

    def jeu_chaine(self, no):
        if no == libIA.LibIA.JEU_PACMAN:
            return "MsPacman-v0"
        if no == libIA.LibIA.JEU_SPACE_INVADERS:
            return "SpaceInvaders-v0"
        if no == libIA.LibIA.JEU_ATLANTIS:
            return "Atlantis-v0"
        if no == libIA.LibIA.JEU_BATTLEZONE:
            return "BattleZone-v0"
        if no == libIA.LibIA.JEU_QUILLES:
            return "Bowling-v0"
        if no == libIA.LibIA.JEU_BOXE:
            return "Boxing-v0"
        if no == libIA.LibIA.JEU_BREAKOUT:
            return "Breakout-v0"
        if no == libIA.LibIA.JEU_ENDURO:
            return "Enduro-v0"
        if no == libIA.LibIA.JEU_HOCKEY:
            return "IceHockey-v0"
        if no == libIA.LibIA.JEU_PITFALL:
            return "Pitfall-v0"
        if no == libIA.LibIA.JEU_PONG:
            return "Pong-v0"
        if no == libIA.LibIA.JEU_SKI:
            return "Skiing-v0"
        if no == libIA.LibIA.JEU_TENNIS:
            return "Tennis-v0"
        if no == libIA.LibIA.JEU_PINBALL:
            return "VideoPinball-v0"

# Classe PERCEPTRON ---------------------------------------------------------------------------------------------------------------------------------------------------
import sys

import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt

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

# Classe REGRESSION ---------------------------------------------------------------------------------------------------------------------------------------------------
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
           
            le_min = np.min(self.x_test)
            le_max = np.max(self.x_test)

            x1 = [[le_min]]
            print(x1)
            y1 = self.modele.predict(x1)
            x2 = [[le_max]]
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

# Classe REGROUPEMENT ---------------------------------------------------------------------------------------------------------------------------------------------------
import sys
from matplotlib import pyplot as plt
from sklearn import cluster

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

        
# Classe LIBIA ---------------------------------------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import random

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
        # try:
        if type == self.TYPE_PERCEPTRON:
            if self.bavard:
                print("LibIA: Création d'un IA de type Perceptron.")
            self.ia = Perceptron(x_entrainement=self.donnees.x_entrainement,
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
            self.ia = Regression(x_entrainement=self.donnees.x_entrainement,
                                            y_entrainement=self.donnees.y_entrainement,
                                            x_test=self.donnees.x_test,
                                            y_test=self.donnees.y_test,
                                            taux_apprentissage=taux_apprentissage,
                                            nb_apprentissages=nb_apprentissages,
                                            bavard=self.bavard)
        elif type == self.TYPE_APPRENTISSAGE_Q:
            if self.bavard:
                print("LibIA: Création d'un IA de type Apprentissage Q.")
            self.ia = Apprentissage_Q(
                                            nb_apprentissages=nb_apprentissages,
                                            jeu=jeu,
                                            bavard=self.bavard)
        elif type == self.TYPE_K_MOYENNE:
            if self.bavard:
                print("LibIA: Création d'un IA de type Regroupement K Moyenne.")
            self.ia = Regroupement(
                                            type=self.TYPE_K_MOYENNE,
                                            x_entrainement=self.donnees.x_entrainement,
                                            y_entrainement=self.donnees.y_entrainement,
                                            nb_categories=nb_categories,
                                            bavard=self.bavard)
        else:
            print("LibIA: * ERREUR 1: * Le type d'IA spécifié est invalide. Vérifiez le paramètre 'type'.")
    #    except:
    #        print("LibIA: * ERREUR 2: * Une erreur est survenue lors de l'appel de 'créez IA'.  Vérifier les paramètres.")
    #        sys.exit()

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

