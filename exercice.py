#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math
import matplotlib.pyplot as plt


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    m = int(64)
    return np.linspace(-1.4, 2.5, 64) #UNIFORM ARRANGEMENT, dont use random.uniform



def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    list_pol = []
    for coord in cartesian_coordinates:
        r = np.sqrt(coord[0] ** 2 + coord[1] ** 2)
        phi = np.arctan2(coord[1], coord[0])
        list_pol.append([r, phi])

    return np.array(list_pol)


def find_closest_index(values: np.ndarray, number: float) -> int:

    print(values)
    index = np.abs(values-number).argmin() # on va a travers array de valeurs pour trouver difference la plus petite

    return values[index]




def graphique(x_values):


# correction

    x = np.linspace(-1,1, num=250)
    #  y = x**2 * math.sin(1/x**2) + x  ==> NO!!!

    y = x**2 * np.sin(1/x**2) + x

    plt.plot(x, y)
    plt.show()



def pi():
# methode montecarlo = cercle de rayon (chiffre a exterieur = exterieur de rayon
   N = 10000

   a = np.random.random(size = N)
   b = np.random.random(size = N)
   z = (a ** 2 + b ** 2) < 1
   pi = 4*z.sum()/N



   return pi


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())

    cart = [[1,2], [2,5]]

    print(coordinate_conversion(np.array(cart)))

    print(find_closest_index(values=np.random.random(10), number=0.5))


    x_values = range(-1, 1)
    #graphique(x_values)

    print(pi())