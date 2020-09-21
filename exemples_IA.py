import LibIA.libIA

def exemple_chrono():
    # Combien de temps il faut pour compter jusqu'a 1 million?
    bob = libIA.LibIA()
    bob.chrono.demarrer()
    for i in range(1000000):
        # Affiche la valeur du compteur à tous les 100,000 tours de boucles
        if i % 100000 == 0:
            bob.afficher_donnees(bob.convertir_caractere(i))
    bob.chrono.arreter()

def exemple_donnees():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.afficher_jeux_donnees()
    bob.charger_donnees(bob.donnees.CHIFFRES_MANUSCRITS)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.afficher_donnees_sources(source="test", axe="X", indice=5)
    bob.afficher_image2d(source="test", axe="X", indice=8, tailleX=28, tailleY=28)
    bob.charger_donnees(bob.donnees.PIMA)
    bob.afficher_donnees_sources(source="entrainement", axe="X", indice=5)

def regression_multiple_pima():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.chrono.demarrer()
    bob.charger_donnees(bob.donnees.PIMA)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_REGRESSION, taux_apprentissage=0.001, nb_apprentissages=5000)
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_erreur()
    bob.afficher_entrees_test()
    bob.afficher_sorties_test()
    bob.ia.predire_resultats(num=5)
    bob.chrono.arreter()


def perceptron_xou():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    bob.chrono.demarrer()
    bob.charger_donnees(bob.donnees.XOU)
    bob.afficher_entrees_entrainement()
    bob.afficher_sorties_entrainement()
    bob.creer_ia(type=bob.TYPE_PERCEPTRON, taux_apprentissage=0.001, nb_neurones_entrees=2, nb_couches_cachees=1,
                 nb_neurones_cachees=12, nb_neurones_sorties=1, nb_apprentissages=5000)
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.afficher_entrees_test()
    bob.afficher_sorties_test()
    bob.ia.predire_resultats(x_test=[[1, 0]])
    bob.ia.afficher_erreur()
    bob.chrono.arreter()


def apprentissage_q_hockey():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    jeu = bob.JEU_HOCKEY
    nb_apprentissage = 2
    bob.creer_ia(type=bob.TYPE_APPRENTISSAGE_Q, jeu=jeu, nb_apprentissages=nb_apprentissage)
    bob.ia.valider_modele()
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_resultat()
    bob.ia.fermer_modele()


def regroupement_nuage_points():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)
    nb_nuages = 6
    nb_points_total = 500
    ecart_type = 25
    bob.charger_donnees(bob.donnees.NUAGES_POINTS, nb_nuages=nb_nuages, nb_points_total=nb_points_total,
                        ecart_type=ecart_type)
    bob.afficher_entrees_entrainement()
    nb_categories = 6
    bob.creer_ia(type=libIA.LibIA.TYPE_K_MOYENNE, nb_categories=nb_categories)
    bob.ia.valider_modele()
    bob.ia.construire_modele()
    bob.ia.afficher_modele()
    bob.ia.entrainer_modele()
    bob.ia.afficher_donnees()
    bob.ia.afficher_resultas()
    bob.ia.fermer_modele()

def exemple_1_parmis_n():
    bob = libIA.LibIA()

    # Exemple 1 - Encode
    print("Exemple 1")
    vecteur = [0, 1, 2, 3, 4]
    vecteur_1_parmi_n = bob.encoder_un_parmi_N(vecteur)
    print(vecteur_1_parmi_n)

    # Exemple 2 - Encode
    print("Exemple 2")
    vecteur = [0, 0, 2, 2, 4, 8]
    vecteur_1_parmi_n = bob.encoder_un_parmi_N(vecteur)
    print(vecteur_1_parmi_n)

    # Exemple 3 - Decode
    print("Exemple 3")
    vecteur = [0, 0, 2, 2, 4, 8]
    vecteur_1_parmi_n = bob.encoder_un_parmi_N(vecteur)
    valeur = bob.decoder_un_parmi_N(vecteur_1_parmi_n[2])
    print(valeur)

