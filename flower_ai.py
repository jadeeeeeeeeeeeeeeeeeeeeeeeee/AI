import numpy as np

'''x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5,0.5], [2,0.5], [5.5,1], [1,1], [4,1.5]), dtype=float) # données d'entrer
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float) # données de sortie /  1 = rouge /  0 = bleu'''

x_entrer = np.array(([0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 2, 1],
    [1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 0, 1], [2, 1, 1, 0, 1, 1], [0, 2, 1, 1, 1, 1], [1, 1, 0, 1, 1, 2],
    [2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 0], [2, 2, 2, 1, 0, 2], [0, 1, 2, 2, 2, 2], [2, 2, 0, 2, 2, 2]), dtype=int)
y = np.array(([0], [0], [0], [0], [0], [1], [1], [1], [1], [1], [2], [2], [2], [2], [2]), dtype=int)

# Changement de l'échelle de nos valeurs pour être entre 0 et 1
x_entrer = x_entrer/np.amax(x_entrer, axis=0) # On divise chaque entré par la valeur max des entrées
y = y/np.amax(y, axis=0)

# On récupère ce qu'il nous intéresse
X = np.split(x_entrer, [15])[0] # Données sur lesquelles on va s'entrainer, les 8 premières de notre matrice

l_question = ["Vous trouvez un portefeuille contenant une importante somme d'argent dans la rue. Que faites-vous ?",
    "Vous avez remarqué que votre collègue de travail semble très stressé et surchargé de tâches. Quelle est votre réaction ?",
    "Pendant une discussion en groupe, quel est votre comportement ?",
    "Vous avez promis d'aider un ami à déménager ce week-end, mais un autre ami vous propose une sortie imprévue. Que choisissez-vous de faire ?",
    "Vous découvrez que quelqu'un que vous connaissez a répandu des rumeurs malveillantes à votre sujet. Comment réagissez-vous ?",
    "Vous êtes témoin d'un incident où quelqu'un est en train de voler dans un magasin. Quelle est votre réaction ?"]

l_rep = [["Vous cherchez activement à retrouver le propriétaire pour lui rendre son bien.",
    "Vous gardez l'argent pour vous, mais vous le remettez à la police au cas où le propriétaire se manifeste.",
    "Vous gardez l'argent et ne dites à personne que vous l'avez trouvé."],
    ["Vous lui proposez votre aide et cherchez des solutions ensemble pour alléger sa charge de travail.",
    "Vous lui faites part de votre sympathie et lui offrez votre soutien moral.",
    "Vous ignorez la situation et continuez vos propres activités sans intervenir."],
    ["Vous écoutez attentivement les opinions des autres et participez de manière constructive à la conversation.",
    "Vous interagissez de manière équilibrée, partageant vos propres idées tout en écoutant celles des autres.",
    "Vous dominez la conversation, coupant la parole aux autres et imposant vos propres opinions."],
    ["Vous honorez votre engagement envers votre ami et reportez la sortie.",
    "Vous expliquez à votre ami déménageur que vous avez une autre opportunité et lui proposez une alternative.",
    "Vous annulez votre engagement sans préavis et vous rendez à la sortie avec votre autre ami."],
    ["Vous confrontez la personne de manière calme et essayez de comprendre ses motivations.",
    "Vous parlez avec la personne pour clarifier les faits et exprimer votre désaccord avec ses actions.",
    "Vous ignorez la personne et évitez toute confrontation directe."],
    ["Vous informez immédiatement un employé du magasin ou les autorités.",
    "Vous observez discrètement la situation pour recueillir des informations avant d'intervenir.",
    "Vous ignorez l'incident et continuez vos achats comme si de rien n'était."]]

'''xPrediction = np.split(x_entrer, [8])[1] # Valeur que l'on veut trouver'''

xPrediction = []

for i in range (6):
    print(l_question[i])
    for j in range (3) :
        print(j+1, ")", l_rep[i][j])
    xPrediction.append((float(input("Réponse 1, 2 ou 3 (voir la console)"))-1)/2)

xPrediction = np.array(xPrediction)


#Notre classe de réseau neuronal
class Neural_Network(object):
  def __init__(self):

  #Nos paramètres
    self.inputSize = 6 # Nombre de neurones d'entrer
    self.outputSize = 1 # Nombre de neurones de sortie
    self.hiddenSize = 10 # Nombre de neurones cachés

  #Nos poids
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (6x10) Matrice de poids entre les neurones d'entrer et cachés
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (10x1) Matrice de poids entre les neurones cachés et sortie


  #Fonction de propagation avant
  def forward(self, X):

    self.z = np.dot(X, self.W1) # Multiplication matricielle entre les valeurs d'entrer et les poids W1
    self.z2 = self.sigmoid(self.z) # Application de la fonction d'activation (Sigmoid)
    self.z3 = np.dot(self.z2, self.W2) # Multiplication matricielle entre les valeurs cachés et les poids W2
    o = self.sigmoid(self.z3) # Application de la fonction d'activation, et obtention de notre valeur de sortie final
    return o

  # Fonction d'activation
  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  # Dérivée de la fonction d'activation
  def sigmoidPrime(self, s):
    return s * (1 - s)

  #Fonction de rétropropagation
  def backward(self, X, y, o):

    self.o_error = y - o # Calcul de l'erreur
    self.o_delta = self.o_error*self.sigmoidPrime(o) # Application de la dérivée de la sigmoid à cette erreur

    self.z2_error = self.o_delta.dot(self.W2.T) # Calcul de l'erreur de nos neurones cachés
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # Application de la dérivée de la sigmoid à cette erreur

    self.W1 += X.T.dot(self.z2_delta) # On ajuste nos poids W1
    self.W2 += self.z2.T.dot(self.o_delta) # On ajuste nos poids W2

  #Fonction d'entrainement
  def train(self, X, y):

    o = self.forward(X)
    self.backward(X, y, o)

  #Fonction de prédiction
  def predict(self):

    print("Donnée prédite apres entrainement: ")
    print("Entrée : \n" + str(xPrediction))
    print("Sortie : \n" + str(self.forward(xPrediction)))

    if(self.forward(xPrediction)*3 < 1):
        print("Forte empathie et sens moral développé. Capacité d'écoute et de collaboration. Engagement envers les promesses et les relations. Confrontatif mais cherche à comprendre et à résoudre les conflits de manière civilisée.Sens de responsabilité civique et volonté d'intervenir pour maintenir l'ordre. Votre caractère peut donc se rapprocher de la princesse Raiponce, en effet vous êtes plutôt bienveillant et cherche à comprendre les problèmes pour les résoudre au mieux.")
    elif (self.forward(xPrediction)*3 < 2):
        print("Pragmatique avec une conscience sociale. Équilibré dans la communication et l'interaction sociale. Flexibilité tout en maintenant les liens avec les autres. Assertif mais ouvert au dialogue pour résoudre les problèmes. Stratégique dans l'approche mais prêt à agir. On peut donc penser que vous ressemblez beaucoup au personnage de Vaina, déterminé à réussir ses objectifs et prête à toujours agir.")
    else :
        print("Manque d'empathie et de moralité, privilégiant ses propres intérêts. Manque d'empathie et de considération pour les autres. Manque de considération pour les engagements et les autres. Évite les conflits mais peut-être au détriment de la résolution des problèmes. Manque d'implication sociale et désintérêt pour les normes de comportement.Vous ressemblez davantage au personnage de Cruella dans les 101 dalmatiens en raison de son manque de bienveillance et gentillesse.")


NN = Neural_Network()

for i in range(1000): #Choisissez un nombre d'itération, attention un trop grand nombre peut créer un overfitting !
    print("# " + str(i) + "\n")
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)

NN.predict()
