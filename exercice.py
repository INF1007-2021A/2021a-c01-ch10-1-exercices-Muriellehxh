#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sy


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:

    list_np = []
    for coord in cartesian_coordinates:
        r = np.sqrt(coord[0]**2 + coord[1]**2)
        phi = np.arctan2(coord[1], coord[0])
        list_np.append([r, phi])
    return np.array(list_np)


def find_closest_index(values: np.ndarray, number: float) -> int:   # not right but its fine

    list_diff = []
    for n in values:
        list_diff.append(abs(n-number))

    return values[list_diff.index(min(list_diff))]



def graphique(): # revoir !! ( tu peux utiliser l'equation direct, lamda marche pas avec )
    x = np.linspace(-1, 1, num=250)
    y = x ** 2 * np.sin(1 / x ** 2) + x

    plt.scatter(x,y)
    plt.show()


def pi():

    n = 10000
    x = np.random.rand(n)
    y = np.random.rand(n)

    y_inside = []
    x_inside = []
    y_outside = []
    x_outside = []

    for numb in range(0,n):
        if math.sqrt(x[numb]**2 + y[numb]**2) < 1 :
            y_inside.append(y[numb])
            x_inside.append(x[numb])
        else:
            y_outside.append(y[numb])
            x_outside.append(x[numb])


    plt.scatter(np.array(x_inside), np.array(y_inside))
    plt.scatter(np.array(x_outside), np.array(y_outside))

    plt.show()




if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())

    cart = [[1, 2], [2, 5]]

    print(coordinate_conversion(np.array(cart)))

    print(find_closest_index(values=np.random.random(10), number=0.5))


    # graphique()

    print(pi())
