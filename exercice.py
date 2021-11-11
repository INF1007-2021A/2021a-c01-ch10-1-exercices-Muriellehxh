#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import math

# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    list_arr1 = []
    full_list = []
    for m in range(0,2):
        for i in np.arange(-1.3, 2.5):
            if len(list_arr1) <= math.sqrt(64):
                list_arr1.append(i)
        full_list.append(list_arr1)

    return np.array(full_list)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return 0


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    print(linear_values())
