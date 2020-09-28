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
