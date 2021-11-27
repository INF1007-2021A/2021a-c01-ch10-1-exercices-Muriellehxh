#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sy


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    m = int(64)
    return np.linspace(-1.4, 2.5, 64)  # UNIFORM ARRANGEMENT, dont use random.uniform


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    list_pol = []
    for coord in cartesian_coordinates:
        r = np.sqrt(coord[0] ** 2 + coord[1] ** 2)
        phi = np.arctan2(coord[1], coord[0])
        list_pol.append([r, phi])

    return np.array(list_pol)


def find_closest_index(values: np.ndarray, number: float) -> int:
    print(values)
    index = np.abs(values - number).argmin()  # on va a travers array de valeurs pour trouver difference la plus petite

    return values[index]


def graphique(x_values):
    # correction

    x = np.linspace(-1, 1, num=250)
    #  y = x**2 * math.sin(1/x**2) + x  ==> NO!!!

    y = x ** 2 * np.sin(1 / x ** 2) + x

    plt.plot(x, y)
    plt.show()


def pi():
    # methode montecarlo = cercle de rayon (chiffre a exterieur = exterieur de rayon
    # YOU CANT CREATE EMPTY NP.ARRAY !! INSTEAD CREATE LIST, THEN UPDATE, THEN TURN INTO ARRAY AT END

    N = 10000

    x = np.random.uniform(low=-1, high=1, size=N)
    y = np.random.uniform(low=-1, high=1, size=N)

    z = (x ** 2 + y ** 2)

    list_int_x = []
    list_int_y = []
    list_ext_x = []
    list_ext_y = []

    for p_cercle in z:
        ind = np.where(z == p_cercle)
        if math.sqrt(p_cercle) < 1:
            list_int_x.append(x[ind])
            list_int_y.append(y[ind])
        else:
            list_ext_x.append(x[ind])
            list_ext_y.append(y[ind])

    np_int_x = np.array(list_int_x)
    np_int_y = np.array(list_int_y)
    np_ext_x = np.array(list_ext_x)
    np_ext_y = np.array(list_ext_y)

    plt.scatter(np_int_x, np_int_y)
    plt.scatter(np_ext_x, np_ext_y)

    # plt.show()


x= sy.Symbol('x')

y = np.exp(-x ** 2)

print(sy.integrate(y, (x)))

x = np.linspace(-4, 4, 1000)
plt.plot(x, f(x))
plt.axhline(color='red')
plt.fill_between(x, f(x), where=[(x>=-4) and (x<=4) for x in x])
plt.show()


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())

    cart = [[1, 2], [2, 5]]

    print(coordinate_conversion(np.array(cart)))

    print(find_closest_index(values=np.random.random(10), number=0.5))

    x_values = range(-1, 1)
    # graphique(x_values)

    print(pi())
