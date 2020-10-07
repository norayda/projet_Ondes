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
plt.plot(xe_,xo_,label='courbe des phases')
plt.xlabel('phi(t)')
plt.ylabel('Gphi(t)')
plt.legend()
plt.show()