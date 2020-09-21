import libIA

def exemple_variables_affichage_1():
    bob = libIA.LibIA()

    # 4 types de variable
    nom = "Bob"
    age = 22
    taille = 1.82
    parle_francais = bob.Vrai

    # Affiche une ligne qui divise l'information.
    bob.afficher_division()

    # La fonction affiche simplement les données à l'écran.
    bob.afficher_donnees("Voici les données!")
    bob.afficher_donnees("Nom: " + nom)
    # Lorsque la variables ne sont pas des chaînes de caractères, il faut les convertir.
    bob.afficher_donnees("Taille: " + bob.convertir_caractere(taille))
    bob.afficher_donnees("Age: " + bob.convertir_caractere(age))


def exemple_saisie_texte():
    bob = libIA.LibIA()

    bob.afficher_donnees("Entrez un nombre: ")
    nombre = bob.entree_donnees()

    bob.afficher_donnees("Le nombre saisi est le: " + bob.convertir_caractere(nombre))

def exemple_if_1():
    bob = libIA.LibIA()

    age = 22

    if age >= 18:
        bob.afficher_donnees("Majeur!")


def exemple_if_2():
    bob = libIA.LibIA()

    age = 22

    if age >= 18:
        bob.afficher_donnees("Majeur!")
    else:
        bob.afficher_donnees("Mineur!")


def exemple_if_3():
    bob = libIA.LibIA()

    partisan_canadiens = bob.Vrai

    if partisan_canadiens == bob.Vrai:
        bob.afficher_donnees("Est un partisan des Canadiens!!")
    else:
        bob.afficher_donnees("N'est pas un partisan des Canadiens")

def exemple_if_4():
    bob = libIA.LibIA()

    annee = 2020
    mois = 6

    if annee == 2020 and mois == 6:
        bob.afficher_donnees("Nous sommes en juin 2020.")


def exemple_for():
    bob = libIA.LibIA()

    min = 0
    max = 10
    saut = 1

    for i in range(min, max, saut):
        bob.afficher_donnees(i)

def exemple_while():
    bob = libIA.LibIA()

    sortie = bob.Faux

    while not sortie:
        bob.afficher_donnees("Entrez un chiffre (0 pour sortie):")
        nombre = bob.entree_donnees()
        bob.afficher_donnees("Vous avez entré: " + nombre)
        if nombre == "0":
            sortie = bob.Vrai


# a finir...
def exemple_vecteurs():
    bob = libIA.LibIA()
    bob.activer_bavard(bob.Vrai)

