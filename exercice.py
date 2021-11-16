#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    m = int(math.sqrt(64))
    return np.random.uniform(size=(m, m), low=-1.4, high=2.5)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    list_pol = []
    for coord in cartesian_coordinates:
        r = np.sqrt(coord[0] ** 2 + coord[1] ** 2)
        phi = np.arctan2(coord[1], coord[0])
        list_pol.append([r, phi])

    return np.array(list_pol)


def find_closest_index(values: np.ndarray, number: float) -> int:
    print(values)
    index = np.abs(values-number).argmin()

    return values[index]


# methode montecarlo = cercle de rayon (chiffre a exterieur = exterieur de rayon

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    lin = (linear_values())
    cart = [[1,2], [2,5]]
    print(coordinate_conversion(np.array(cart)))

    print(find_closest_index(values=np.random.random(10), number=0.5))