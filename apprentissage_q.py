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

import LibIA.libIA

tf.disable_v2_behavior()

from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()


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
