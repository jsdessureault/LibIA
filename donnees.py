import random
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn import datasets
from sklearn import model_selection
import pandas
import random


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


