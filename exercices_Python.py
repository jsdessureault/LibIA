import LibIA.libIA

def exercice_affiche():
    bob = libIA.LibIA()

    bob.afficher_donnees("Bonjour tout le monde!")
    age_capitaine = 67
    bob.afficher_donnees("Age du capitaine: " + bob.convertir_caractere(age_capitaine))
    bob.afficher_division()
    bob.afficher_donnees("En somme, le capitaine a " + bob.convertir_caractere(age_capitaine) + " ans.")
    bob.afficher_division()

def exercice_if():
    bob = libIA.LibIA()

    bob.afficher_donnees("Entrez l'âge de la personne: ")
    age = bob.entree_donnees()
    age = bob.convertir_entier(age)

    partisan_canadiens = bob.Vrai

    if age >= 18:
        bob.afficher_donnees("Majeur!")
    else:
        bob.afficher_donnees("Mineur!")

    if partisan_canadiens == bob.Vrai:
        bob.afficher_donnees("Un partisan des Canadiens!!")

    if age < 12 and partisan_canadiens == bob.Vrai:
        bob.afficher_donnees("Un jeune partisan des Canadiens.")

    if age < 18 or partisan_canadiens == bob.Vrai:
        bob.afficher_donnees("Mineur ou partisan des Canadiens.")

def exercice_for():
    bob = libIA.LibIA()

    min = 0
    max = 10
    saut = 1

    for i in range(min, max, saut):
        bob.afficher_donnees(i)

def exercice_while():
    bob = libIA.LibIA()

    bob.afficher_donnees("Entrez un nombre:")
    nombre = bob.entree_donnees()
    nombre = bob.convertir_entier(nombre)

    while nombre >= 0:
        bob.afficher_donnees("Compteur: " + bob.convertir_caractere(nombre))
        nombre = nombre - 1


def exerice_roche_papier_ciseaux():
    bob = libIA.LibIA()
    sophie = libIA.LibIA()

    victoires_bob = 0
    victoires_sophie = 0
    nb_victoires_requises = 5

    while victoires_bob < nb_victoires_requises and victoires_sophie < nb_victoires_requises:
        choix_bob = bob.tirage_au_sort(minimum=1, maximum=3)
        choix_sophie = sophie.tirage_au_sort(minimum=1, maximum=3)

        # 1: Roche
        # 2: Papier
        # 3: Ciseaux

        if choix_bob == choix_sophie:
            bob.afficher_donnees("Match null!")

        if choix_bob == 1 and choix_sophie == 2:
            sophie.afficher_donnees("j'ai (Sophie) gagné!")
            victoires_sophie = victoires_sophie + 1
        if choix_bob == 1 and choix_sophie == 3:
            bob.afficher_donnees("j'ai (Bob) gagné!")
            victoires_bob = victoires_bob + 1
        if choix_bob == 2 and choix_sophie == 1:
            bob.afficher_donnees("j'ai (Bob) gagné!")
            victoires_bob = victoires_bob + 1
        if choix_bob == 2 and choix_sophie == 3:
            sophie.afficher_donnees("j'ai (Sophie) gagné!")
            victoires_sophie = victoires_sophie + 1
        if choix_bob == 3 and choix_sophie == 1:
            sophie.afficher_donnees("j'ai (Sophie) gagné!")
            victoires_sophie = victoires_sophie + 1
        if choix_bob == 3 and choix_sophie == 2:
            bob.afficher_donnees("j'ai (Bob) gagné!")
            victoires_bob = victoires_bob + 1

    sophie.afficher_donnees("Marque finale")
    sophie.afficher_donnees("Sophie: " + sophie.convertir_caractere(victoires_sophie))
    bob.afficher_donnees("Bob: " + sophie.convertir_caractere(victoires_bob))
