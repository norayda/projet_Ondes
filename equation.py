#Bibliotheques
import numpy as np # donne acces a la librairie numpy, et definit l'abbreviation np
from random import *
from __future__ import division
from scipy import *
from pylab import *
import scipy.linalg
from scipy.integrate import odeint
import matplotlib.pyplot as plt # donne acces a la librairie matplotlib.pyplot, et definit l'abbreviation plt
import math     # donne acces aux fonctions et constantes mathematiques
%matplotlib inline    
# ouvre les fenetres graphiques dans le notebook

#definition des fonctions
def deriv(syst, t):
    [x,y,v,u] = syst
    dxdt = v
    dydt = u
    dvdt = -(x+(x**2+y**2)*x)
    dudt = -(mu*y+(x**2+y**2)*y)
    return [dxdt,dydt,dvdt,dudt]                  # Derivees des variables

def trace(y,x, number):
    plt.subplot(number)
    plt.plot(y, x)
    plt.xlabel('Gphi(t)')
    plt.ylabel('phi(t)')
    plt.title("Courbe de psy par rapport a phi")
    plt.show()

# Parametres temps
start = -100
end = 100
numsteps = 5000
t = np.linspace(start,end,numsteps)

#Premiere version de solutions
#Constantes
mu=.3
# Conditions initiales et resolution
x0,y0 = 1.4, 1.4
v0,u0 = .000, 0.000
syst_CI=np.array([x0,y0,v0,u0])    # Tableau des CI
Sols=odeint(deriv,syst_CI,t)
# Recuperation des solutions
[x,y,v,u] = Sols . T        # Decomposition du tableau des solutions : Affectation avec transposition
#affichage
trace(y,x,111)

#Deuxieme version de solutions
#Constantes
mu=.5
# Conditions initiales et resolution
x0,y0 = -1., 2.
v0,u0 = 2., -.5
syst_CI=np.array([x0,y0,v0,u0])    # Tableau des CI
Sols=odeint(deriv,syst_CI,t)
# Recuperation des solutions
[x,y,v,u] = Sols . T        # Decomposition du tableau des solutions : Affectation avec transposition
#affichage
trace(y,x,111)


#Equation D'alembert
#Biblioteques
import numpy as np # donne acces a la librairie numpy, et definit l'abbreviation np
import scipy        # donne acces aux librairies scipy, scipy.linalg et scipy.integrate
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt # donne acces a la librairie matplotlib.pyplot, et definit l'abbreviation plt
import math     # donne acces aux fonctions et constantes mathematiques
%matplotlib inline    
# ouvre les fenetres graphiques dans le notebook

#definition des fonctions
mu=0.3
A =np.array([[-1.,0], [-mu,0]]) # la matrice
#La fonction second membre en respectant l'ordre des arguments
def second_membre(Y,t):
    return Y
def sol_exacte(phi0, Gphi0, t):
    Z=(phi0*math.sin(t), Gphi0*math.sin(t*math.sqrt(mu)))
    return Z
 
phi0 = 1.
Gphi0 = 1.
t_ =np.linspace(-100,100,10000)

phi=scipy.integrate.odeint(second_membre, phi0, t_)
Gphi=scipy.integrate.odeint(second_membre, Gphi0, t_)
Ye=[sol_exacte(phi0, Gphi0, t) for t in t_] #vecteur de la solution exacte sur le vecteur discretise t_
xo_=[] 
xe_=[]
    
for ye in Ye:
    xo_.append(ye[0])
    xe_.append(ye[1])
    
plt.subplot()
plt.plot(t_,xo_,label='phi exacte')
plt.plot(t_,xe_,label='Gphi exacte')
plt.xlabel('t')
plt.ylabel('phi(t), Gphi(t)')
plt.legend()
plt.show()

plt.subplot()
plt.plot(xe_,xo_,label='courbe des phases')
plt.xlabel('phi(t)')
plt.ylabel('Gphi(t)')
plt.legend()
plt.show()

# Dichotomie


#definition des fonctions
def deriv(syst, t):
    [x,y,v,u] = syst
    dxdt = v
    dydt = u
    dvdt = -(x+(x**2+y**2)*x)
    dudt = -(mu*y+(x**2+y**2)*y)
    return [dxdt,dydt,dvdt,dudt]                  # Derivees des variables

def trace(y,x, number):
    plt.subplot(number)
    plt.plot(y, x)
    plt.xlabel('Gphi(t)')
    plt.ylabel('phi(t)')
    plt.title("Courbe de psy par rapport a phi")
    plt.show()

# Parametres temps
start = -100
end = 100
numsteps = 5000
t = np.linspace(start,end,numsteps)

#Premiere version de solutions
#Constantes
mu=.3
# Conditions initiales et resolution
x0,y0 = 1.4, 1.4
v0,u0 = .000, 0.000
syst_CI=np.array([x0,y0,v0,u0])    # Tableau des CI
Sols=odeint(deriv,syst_CI,t)
# Recuperation des solutions
[x,y,v,u] = Sols . T        # Decomposition du tableau des solutions : Affectation avec transposition
#affichage
trace(y,x,111)

#Deuxieme version de solutions
#Constantes
mu=.5
# Conditions initiales et resolution
x0,y0 = -1., 2.
v0,u0 = 2., -.5
syst_CI=np.array([x0,y0,v0,u0])    # Tableau des CI
Sols=odeint(deriv,syst_CI,t)
# Recuperation des solutions
[x,y,v,u] = Sols . T        # Decomposition du tableau des solutions : Affectation avec transposition
#affichage
trace(y,x,111)


#Equation D'alembert
#Biblioteques
import numpy as np # donne acces a la librairie numpy, et definit l'abbreviation np
import scipy        # donne acces aux librairies scipy, scipy.linalg et scipy.integrate
import scipy.linalg
import scipy.integrate
import matplotlib.pyplot as plt # donne acces a la librairie matplotlib.pyplot, et definit l'abbreviation plt
import math     # donne acces aux fonctions et constantes mathematiques
%matplotlib inline    
# ouvre les fenetres graphiques dans le notebook

#definition des fonctions
mu=0.3
A =np.array([[-1.,0], [-mu,0]]) # la matrice
#La fonction second membre en respectant l'ordre des arguments
def second_membre(Y,t):
    return Y
def sol_exacte(phi0, Gphi0, t):
    Z=(phi0*math.sin(t), Gphi0*math.sin(t*math.sqrt(mu)))
    return Z
 
phi0 = 1.
Gphi0 = 1.
t_ =np.linspace(-100,100,10000)

phi=scipy.integrate.odeint(second_membre, phi0, t_)
Gphi=scipy.integrate.odeint(second_membre, Gphi0, t_)
Ye=[sol_exacte(phi0, Gphi0, t) for t in t_] #vecteur de la solution exacte sur le vecteur discretise t_
xo_=[] 
xe_=[]
    
for ye in Ye:
    xo_.append(ye[0])
    xe_.append(ye[1])
    
plt.subplot()
plt.plot(t_,xo_,label='phi exacte')
plt.plot(t_,xe_,label='Gphi exacte')
plt.xlabel('t')
plt.ylabel('phi(t), Gphi(t)')
plt.legend()
plt.show()

plt.subplot()
plt.plot(xe_,xo_,label='courbe des phases')
plt.xlabel('phi(t)')
plt.ylabel('Gphi(t)')
plt.legend()
plt.show()

#DICHOTOMIE

# Paramètres temps
start = -12
end = 12
numsteps = 3000
t = np.linspace(start,end,numsteps)

#Dichotomie
# Conditions initiales et résolution
mu=0.38100875
x0=0.59
y0=1.555875
v0=0.81
u0=-0.15587499999999999

syst_CI=np.array([x0,y0,v0,u0])    # Tableau des CI
Sols=odeint(deriv,syst_CI,t)
# Récupération des solutions
[x,y,v,u] = Sols . T        # Décomposition du tableau des solutions : Affectation avec transposition
#affichage
trace(y,x,111)

# Code pour générer des valeurs initiales pour les resultats


# Paramètres temps
start = -10
end = 10
numsteps = 2500
t = np.linspace(start,end,numsteps)

mu=random()*50 # valleurs allant de -30 à 30

def multiTracage30(n):
    for i in range(n):
        x0, y0= random()*30-15, random()*30-15
        v0, u0= random()*30-15, random()*30-15
        #affichage valeurs
        print('mu='); print(mu)
        print('x0=');print(x0); print('y0='); print(y0)
        print('v0=');print(v0); print('u0='); print(u0)
        
        syst_CI=np.array([x0,y0,v0,u0])    # Tableau des CI
        Sols=odeint(deriv30,syst_CI,t)
        # Récupération des solutions
        [x,y,v,u] = Sols . T        # Décomposition du tableau des solutions : Affectation avec transposition
        #affichage
        trace(y,x,111)
        
multitracage30(100)
