import numpy as np
from PIL import Image


def compatibility(arr):
    M = np.array([[0] * 6, [0, .501, .210, .174, .056, .056], [0, .210, .501, .210, .210, .210],
                  [0, .174, .210, .501, .174, .174], [0, .056, .210, .174, .501, .501],
                  [0, .056, .210, .174, .501, .501]])
    S = np.zeros(arr.shape, dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            N = neighbours(arr, i, j)
            C = np.zeros(N.shape, dtype=int)
            C[:] = arr[i, j]
            S[i, j] = np.sum(M[N, C]) / np.size(N)

    return np.mean(S)


def neighbours(arr, i, j, d=1):
    x1, x2 = i - d, i + d
    y1, y2 = j - d, j + d
    if i - d < 0:
        x1 = 0
    if i + d >= arr.shape[0]:
        x2 = arr.shape[0] - 1
    if j - d < 0:
        y1 = 0
    if j + d >= arr.shape[1]:
        y2 = arr.shape[1] - 1
    return arr[x1:x2 + 1, y1:y2 + 1]


def compactness(arr, w=5):
    S = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            N = neighbours(arr, i, j, w)
            S[i, j] = (N == arr[i, j]).sum() / np.size(N)
    return np.mean(S)


def proportion(arr):
    S = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0:
                S[i, j] = 0
            elif arr[i, j] == 1:
                S[i, j] = proportion.marta[i, j]
            elif arr[i, j] == 2:
                S[i, j] = proportion.garden[i, j]
            elif arr[i, j] == 3:
                S[i, j] = proportion.urban[i, j]
            elif arr[i, j] == 4:
                S[i, j] = proportion.d_agri[i, j]
            elif arr[i, j] == 5:
                S[i, j] = proportion.w_agri[i, j]
    return np.sum(S)


proportion.marta = np.array(Image.open('input/fuzzy_range.tif'))
proportion.garden = np.array(Image.open('input/fuzzy_garden.tif'))
proportion.urban = np.array(Image.open('input/fuzzy_urban.tif'))
proportion.d_agri = np.array(Image.open('input/fuzzy_d_agri.tif'))
proportion.w_agri = np.array(Image.open('input/fuzzy_w_agri.tif'))


def difficulty_of_change(arr):
    M = np.array(
        [[0] * 6, [0, 0, .109, .144, .109, .109], [0, .234, 0, .144, .234, .144], [0, .450, .450, 0, .234, .234],
         [0, .109, .144, .059, 0, .144], [0, .144, .144, .234, .144, 0]])
    S = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            S[i, j] = M[difficulty_of_change.initial[i, j], arr[i, j]]
    return np.sum(S)


difficulty_of_change.initial = np.array(Image.open('input/landuse_raster.tif'))
