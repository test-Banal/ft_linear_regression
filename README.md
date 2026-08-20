# Python Project Template

Template Python moderne basé sur `uv`

# Project : ft_linear_regression
Decouvrir le champ de Machine Learning, via un projet utilisant un basique machine learning algorithm.
En se formalisant avec des concepts basiques derriere le machine learning
Ici linear function entraine avec un algorithme de descente de gradient

## Profils
Langage choisis : Python 
Librairie : NUmpy ? Matplotlib
Norme : flake8
Outils debug, warning : mypy, pytest, jupyterlab

### OBJECTIF : 
° global : predire le prix d'une voiture en fonction de son kilometrage. 
Creation de deux programmes : 
    ° first : predire le prix en fonction du kilometre : should prompt the mileage -> result : estimated price
                    estimatePrice(mileage) = O0 + (O1 * mileage) 
    °The second program will be used to train your model. It will read your dataset file  and perform a linear regression on the data. Once the linear regression has completed, you will save the variables theta0 and theta1 for use in the first program.
    You will be using the following formulas :
        tmpθ0 = learningRate ∗ 1   ∑(m−1) (estimateP rice(mileage[i]) − price[i])
                               m    i=0

tmpθ1 = learningRate ∗ 1 ∑(m−1) (estimateP rice(mileage[i]) − price[i]) ∗ mileage[i]
                       m   i=0

I let you guess what m is 
Note that the estimatePrice is the same as in our first program, but here it uses
your temporary, most recently computed theta0 and theta1.
Also, don’t forget to simultaneously update theta0 and theta1.

### Bonus :
    ° graphique
    °graphique linear regression
    ° programme qui calcule la precision de l'algorithme. 

### Angle d'attaque


### A LA FIN
essayer avec librairie genre : sklearn.linear_model.linearregrssion, numpy.polyfit, scipy optimisation ou modele ml deja implemente pour comparer mes resultats.