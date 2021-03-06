import LibIA.libIA



def regression_multiple_cancer_sein():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.chrono.demarrer()
    bob.charger_donnees(bob.donnees.CANCER_SEIN)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_REGRESSION, taux_apprentissage=0.001, nb_apprentissages=5000)
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_erreur()
    bob.afficher_entrees_test()
    bob.afficher_sorties_test()
    bob.ia.predire_resultats(num=0)
    bob.chrono.arreter()

def perceptron_chiffres():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.charger_donnees(bob.donnees.CHIFFRES_MANUSCRITS)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_PERCEPTRON, taux_apprentissage=0.001, nb_neurones_entrees=784, nb_couches_cachees=2,
                 nb_neurones_cachees=784, nb_neurones_sorties=10, nb_apprentissages=2)
    bob.ia.construire_modele()
    # bob.ia.afficher_modele()
    bob.chrono.demarrer()
    bob.ia.entrainer_modele()
    bob.chrono.arreter()
    bob.afficher_entrees_test()
    bob.afficher_sorties_test()
    bob.afficher_image2d(source="test", axe="X", indice=5434, tailleX=28, tailleY=28)
    prediction = bob.ia.predire_resultats(num=5434)
    bob.afficher_division()
    bob.afficher_donnees("Prediction brute: " + bob.convertir_caractere(prediction))
    bob.afficher_division()
    bob.afficher_donnees("Prediction en clair: " + bob.convertir_caractere(bob.donner_en_clair(prediction)))
    bob.afficher_division()
    bob.afficher_donnees("Prediction en chiffre: " + bob.convertir_caractere(bob.decoder_un_parmi_N(prediction)))
    bob.afficher_division()
    bob.ia.afficher_erreur()

def perceptron_mode():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.charger_donnees(bob.donnees.MODE)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_PERCEPTRON, taux_apprentissage=0.001, nb_neurones_entrees=784, nb_couches_cachees=2,
                 nb_neurones_cachees=784, nb_neurones_sorties=10, nb_apprentissages=100)
    bob.ia.construire_modele()
    #bob.ia.afficher_modele()
    bob.chrono.demarrer()
    bob.ia.entrainer_modele()
    bob.chrono.arreter()
    bob.afficher_entrees_test()
    bob.afficher_sorties_test()
    bob.afficher_image2d(source="test", axe="X", indice=5434, tailleX=28, tailleY=28)
    prediction = bob.ia.predire_resultats(num=5434)
    bob.afficher_division()
    bob.afficher_donnees("Prediction brute: " + bob.convertir_caractere(prediction))
    bob.afficher_division()
    bob.afficher_donnees("Prediction en clair: " + bob.convertir_caractere(bob.donner_en_clair(prediction)))
    bob.afficher_division()
    bob.afficher_donnees("Vetement predit: " + bob.convertir_caractere(bob.decoder_un_parmi_N(prediction)))
    bob.afficher_division()
    bob.ia.afficher_erreur()

def regroupement_nuage_points():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    nb_nuages = 3
    nb_points_total = 50
    ecart_type = 10
    bob.charger_donnees(bob.donnees.NUAGES_POINTS, nb_nuages=nb_nuages, nb_points_total=nb_points_total,
                        ecart_type=ecart_type)
    bob.afficher_entrees_entrainement()
    nb_categories = 4
    bob.creer_ia(type=libIA.LibIA.TYPE_K_MOYENNE, nb_categories=nb_categories)
    bob.ia.valider_modele()
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_donnees()
    bob.ia.afficher_resultas()
    bob.ia.fermer_modele()


def apprentissage_q_pitfall():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    jeu = libIA.LibIA.JEU_PITFALL
    taux_apprentissage = 0.001
    nb_apprentissage = 2
    bob.creer_ia(type=libIA.LibIA.TYPE_APPRENTISSAGE_Q, jeu=jeu, taux_apprentissage=taux_apprentissage,
                 nb_apprentissages=nb_apprentissage)
    bob.ia.valider_modele()
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_resultat()
    bob.ia.fermer_modele()

def regression_nhl():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.chrono.demarrer()
    bob.charger_donnees(bob.donnees.NHL)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_REGRESSION, taux_apprentissage=0.001, nb_apprentissages=20)
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_erreur()
    bob.afficher_entrees_test()
    bob.afficher_sorties_test()
    bob.ia.predire_resultats(num=5)
    #bob.ia.predire_resultats(x_test=[[98, 102]])
    bob.chrono.arreter()

def regression_mlb():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.chrono.demarrer()
    bob.charger_donnees(bob.donnees.MLB)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_REGRESSION, taux_apprentissage=0.001, nb_apprentissages=5000)
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_erreur()
    #bob.afficher_entrees_test()
    #bob.afficher_sorties_test()
    #bob.ia.predire_resultats(num=2)
    bob.ia.predire_resultats(x_test=[[98, 102, 3, 3, 3]])
    bob.chrono.arreter()
