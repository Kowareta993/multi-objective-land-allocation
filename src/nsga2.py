import numpy as np
import random
from PIL import Image, ImageOps
import math
from functions import compactness, compatibility, difficulty_of_change, proportion
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import cohen_kappa_score

properties = {
    "constraints": {
        "area": {
            1: 83685600,
            2: 19832400,
            3: 153000,
            4: 1736100,
            5: 5581800
        },
        "min_area": 0.2,
        "max_area": 0.2,
        'w_slope': 5,
        'd_slope': 10
    },
    'w': 9,
    'h': 12,
    'init': np.array(Image.open('input/landuse_raster.tif')).astype(int),
    'random_ratio': 0.7,
    "cell_area": 30 * 30,
    "land_types": 6,
    "edge_mutation_size": 5,
    'beta': 10,
    'generations': 50,
    'population_size': 200,
    'initial_ratio': 0.5,
    'crossover': None
}


def map_arr(arr, func="normal"):
    w = properties['w']
    h = properties['h']
    x = int(np.ceil(arr.shape[0] / h))
    y = int(np.ceil(arr.shape[1] / w))
    A = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            mx = min((i + 1) * h, arr.shape[0])
            my = min((j + 1) * w, arr.shape[1])
            if func == "normal":
                A[i, j] = np.argmax(np.bincount(arr[i * h:mx, j * w:my].flatten()))
            if func == "mean":
                A[i, j] = np.mean(arr[i * h:mx, j * w:my])
            if func == "max":
                A[i, j] = np.max(arr[i * h:mx, j * w:my])
    return A


def unmap_arr(arr):
    w = properties['init'].shape[1]
    h = properties['init'].shape[0]
    x = properties['h']
    y = properties['w']
    A = np.zeros((h, w))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            mx = min((i + 1) * x, h)
            my = min((j + 1) * y, w)
            A[i * x:mx, j * y:my] = arr[i, j]
    return A


def set_properties():
    global properties
    properties['constraints']['min_usage'] = {}
    properties['constraints']['max_usage'] = {}
    cell_area = properties['cell_area'] * properties['w'] * properties['h']
    for key, val in properties['constraints']['area'].items():
        properties['constraints']['min_usage'][key] = (1 - properties['constraints']['min_area']) * val // cell_area
        properties['constraints']['max_usage'][key] = (1 + properties['constraints']['max_area']) * val // cell_area
    properties['slope'] = map_arr(np.array(Image.open('input/slope.tif')), 'max')
    proportion.marta = map_arr(np.array(Image.open('input/fuzzy_range.tif')), 'mean')
    proportion.garden = map_arr(np.array(Image.open('input/fuzzy_garden.tif')), 'mean')
    proportion.urban = map_arr(np.array(Image.open('input/fuzzy_urban.tif')), 'mean')
    proportion.d_agri = map_arr(np.array(Image.open('input/fuzzy_d_agri.tif')), 'mean')
    proportion.w_agri = map_arr(np.array(Image.open('input/fuzzy_w_agri.tif')), 'mean')

    difficulty_of_change.initial = map_arr(properties['init']).astype(int)


def dominates(fit, sol1, sol2):
    f1 = fit[sol1]
    f2 = fit[sol2]
    if (f1 == f2).sum() == f1.shape[0]:
        return False
    return (f1 >= f2).sum() == f1.shape[0]


def fast_nondominated_sort(P, fit):
    Psize = P.shape[0]
    S = [set() for _ in range(Psize)]
    F = [set() for _ in range(Psize + 1)]
    n = np.zeros(Psize)
    i = 0
    for p in range(Psize):
        for q in range(Psize):
            if dominates(fit, p, q):
                S[p].add(q)
            elif dominates(fit, q, p):
                n[p] += 1
        if n[p] == 0:
            F[i].add(p)
    while len(F[i]) > 0:
        H = set()
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    H.add(q)
        i += 1
        F[i] = H
    result = [np.array(list(Fi), dtype=int) for Fi in F if len(Fi) > 0]
    nfit = [np.array([fit[i] for i in j]) for j in result]
    return result, nfit


def penalty(p):
    x = 0.0
    if np.size(properties['slope'][p == 4]) > 0:
        if np.max(properties['slope'][p == 4]) > properties['constraints']['d_slope']:
            x += 1
    else:
        x += 1
    if np.size(properties['slope'][p == 5]) > 0:
        if np.max(properties['slope'][p == 5], 0) > properties['constraints']['w_slope']:
            x += 1
    else:
        x += 1
    counts = np.bincount(p.flatten())
    for i in range(1, properties['land_types']):
        if i < len(counts):
            if counts[i] < properties['constraints']['min_usage'][i]:
                x += abs(properties['constraints']['min_usage'][i] - counts[i]) / (properties['h'] * properties['w'])
            if counts[i] > properties['constraints']['max_usage'][i]:
                x += abs(properties['constraints']['max_usage'][i] - counts[i]) / (properties['h'] * properties['w'])
        else:
            x += 1
    return 1 + properties['beta'] * x / 7


def fitness(p):
    err = penalty(p)
    return np.array([proportion(p) / err, compatibility(p) / err, compactness(p) / err, -difficulty_of_change(p) * err])
    


def crowding_distance_assignment(L, fit):
    D = np.zeros(len(L))
    if len(L) == 0:
        return D
    for m in range(fit.shape[1]):
        idx = fit[:, m].argsort()
        D[idx[0]] = np.inf
        D[idx[-1]] = np.inf
        for i in range(1, len(L) - 1):
            D[i] += fit[idx[i + 1], m] - fit[idx[i - 1], m]
    return D


def crowding_distance_sort(P, F, fit):
    for i in range(len(F)):
        if len(F[i]) > 0:
            Di = crowding_distance_assignment(P[F[i]], fit[i])
            idx = np.argsort(-Di)
            F[i] = F[i][idx]
            fit[i] = fit[i][idx]
    return F, fit


def crossover(P):
    C = mutation(edge_crossover(P))
    for i in range(properties['edge_mutation_size']):
        constraint_edge_mutation(C)
    return C


def box_crossover(P):
    n = P.shape[0]
    h = P.shape[1]
    w = P.shape[2]
    parents1 = np.random.randint(0, n, size=n // 2)
    parents2 = np.random.randint(0, n, size=n // 2)
    H1 = np.random.randint(0, h, size=n // 2)
    H2 = np.random.randint(0, h, size=n // 2)
    W1 = np.random.randint(0, w, size=n // 2)
    W2 = np.random.randint(0, w, size=n // 2)
    masks = np.zeros((n // 2, h, w)).astype(int)
    for i in range(n//2):
        h1, h2 = min(H1[i], H2[i]), max(H1[i], H2[i])
        w1, w2 = min(W1[i], W2[i]), max(W1[i], W2[i])
        masks[i, h1:h2+1, w1:w2+1] = 1
    children1 = P[parents1]
    children1[masks] = P[parents2][masks]
    children2 = P[parents2]
    children2[masks] = P[parents1][masks]
    return np.concatenate((children1, children2), axis=0)

def crossover2(P):
    C = mutation(box_crossover(P))
    for i in range(properties['edge_mutation_size']):
        constraint_edge_mutation(C)
    return C

def mutation(C):
    return np.array(list(map(path_based_mutation, C)))


def initial_population(n):
    P = [None] * n
    current = map_arr(properties['init']).astype(int)
    im = to_img(current)
    im.save(f"results/init.tif")
    c = properties['constraints']
    s = properties['land_types']
    weights = [(c['min_usage'][i] + c['max_usage'][i]) / 2 for i in range(1, s)]
    for i in range(n):
        # P[i] = np.random.randint(1, properties["land_types"], size=current.shape)
        P[i] = np.random.choice(list(range(1, s)), current.shape,
                                p=[weights[i - 1] / sum(weights) for i in range(1, s)])
        if i < n * properties['initial_ratio']:
            k = int(properties['random_ratio'] * np.size(current))
            mask = np.array([True] * k + [False] * (np.size(current) - k))
            np.random.shuffle(mask)
            mask = np.reshape(mask, current.shape)
            P[i][mask] = current[mask]
        P[i][current == 0] = 0
    return np.array(P)


def to_img(arr):
    A = unmap_arr(arr)
    narr = np.zeros((A.shape[0], A.shape[1], 3)).astype('uint8')
    narr[A == 0, :] = 255
    narr[A == -1, :] = 0

    narr[A == 1, 0] = 0
    narr[A == 1, 1] = 97
    narr[A == 1, 2] = 61

    narr[A == 2, 0] = 85
    narr[A == 2, 1] = 255
    narr[A == 2, 2] = 0

    narr[A == 3, 0] = 255
    narr[A == 3, 1] = 255
    narr[A == 3, 2] = 0

    narr[A == 4, 0] = 207
    narr[A == 4, 1] = 0
    narr[A == 4, 2] = 255

    narr[A == 5, 0] = 0
    narr[A == 5, 1] = 77
    narr[A == 5, 2] = 168
    
    img = Image.fromarray(narr, 'RGB')
    return img


def save_solution(p, fname):
    try:
        os.mkdir(f'results/bests')
    except:
        pass
    im = to_img(p)
    im.save(f"results/bests/{fname}.tif")

def save_kappa(p, gen, i):
    try:
        os.mkdir(f'results/{gen}')
    except:
        pass
    im = to_img(p)
    im.save(f"results/{gen}/{i}.tif")


def save_front(F, gen):
    try:
        os.mkdir(f'results/{gen}')
        os.mkdir(f'results/{gen}/fronts')
    except:
        pass
    i = 1
    for p in F:
        im = to_img(p)
        im.save(f"results/{gen}/fronts/{i}.tif")
        i += 1


def check_neighbours(arr, i, j):
    if arr[i, j] == 0:
        return False
    x1, x2 = i - 1, i + 1
    y1, y2 = j - 1, j + 1
    if i == 0:
        x1 = i
    if i == arr.shape[0] - 1:
        x2 = i
    if j == 0:
        y1 = j
    if j == arr.shape[1] - 1:
        y2 = j
    return (arr[x1:x2 + 1, y1:y2 + 1] != arr[i, j]).sum() - (arr[x1:x2 + 1, y1:y2 + 1] == 0).sum() > 0


def edge_cells(arr):
    mask = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            mask[i, j] = check_neighbours(arr, i, j)
    return mask.astype(np.bool_)


def edge_crossover(P):
    n = P.shape[0]
    parents1 = np.random.randint(0, n, size=n // 2)
    parents2 = np.random.randint(0, n, size=n // 2)
    edges = np.array(list(map(edge_cells, P)))
    children = np.concatenate([P[parents1, :], P[parents2, :]])
    edges = np.concatenate([edges[parents1, :], edges[parents1, :]])
    children2 = np.concatenate([P[parents2, :], P[parents1, :]])
    children[edges] = children2[edges]
    return children


def path_based_mutation(arr, w=3, s=7):
    m = arr.shape[0] - w
    n = arr.shape[1] - w
    idx = random.randint(0, m * n - 1)
    i = idx // n
    j = idx % n
    window = arr[i:i + 3, j:j + 3]
    selected = np.array([True] * s + [False] * (w * w - s))
    np.random.shuffle(selected)
    selected = selected.reshape((w, w))
    ss = window[selected]
    m = np.argmax(np.bincount(ss))
    if m == 0:
        return arr
    selected = np.logical_and(selected, window != 0)
    window[selected] = m
    arr[i:i + 3, j:j + 3] = window[:]
    return arr


def constraint_edge_mutation(P):
    C = properties['constraints']
    idx = random.randint(0, P.shape[0] - 1)
    usage = np.bincount(P[idx].flatten())
    edges = edge_cells(P[idx])
    n = edges[edges == True].sum()
    ii = random.randint(0, n)
    i = 0
    j = 0
    k = -1
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if edges[i, j]:
                ii -= 1
            if ii == 0:
                k = int(P[idx, i, j])
    if k == -1:
        print("PANIC")
        exit(1)
    if k == 0:
        return
    if usage[k] < C['min_usage'][k]:
        return
    if usage[k] > C['max_usage'][k]:
        rnd = np.random.choice([i for i in range(1, properties['land_types']) if i != k], 1)
        P[idx, i, j] = rnd
    else:
        P[idx, i, j] = random.randint(1, properties['land_types'] - 1)


def save_slopes():
    A = np.zeros(properties['slope'].shape)
    A[properties['slope'] < properties['constraints']['w_slope']] = 5
    A[map_arr(properties['init']) == 0] = -1
    to_img(A).save('results/w_slope.tif')
    A = np.zeros(properties['slope'].shape)
    A[properties['slope'] < properties['constraints']['d_slope']] = 4
    A[map_arr(properties['init']) == 0] = -1
    to_img(A).save('results/d_slope.tif')


def scatter_3d(data, labels, i):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(data[0], data[1], data[2])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    fig.savefig(f'results/charts/3d_{i}.png')

def scatter_2d(data, labels, i):
    fig = plt.figure()
    ax = plt.axes()
    ax.scatter(data[0], data[1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    fig.savefig(f'results/charts/2d_{i}.png')

def plot(data, labels, i):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(data[0], data[1])
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    fig.savefig(f'results/charts/plot_{i}.png')

def table(row_headers, column_headers, cell_text, i):
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
    fig = plt.figure(linewidth=2)
    ax = plt.axes()
    ax.axis('off')
    the_table = ax.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowColours=rcolors,
                      rowLoc='right',
                      colColours=ccolors,
                      colLabels=column_headers,
                      loc='center')
    fig.savefig(f'results/charts/table_{i}.png', bbox_inches='tight', dpi=150)            

def kappa(F, i, gen):
    init = map_arr(properties['init'])
    init = init.reshape((init.size,))
    scores = [cohen_kappa_score(init, p.reshape((p.size, ))) for p in F]
    best = F[np.argmax(scores)]
    save_kappa(best, gen, i)
    
def report(P, H):
    try:
        os.mkdir('results/charts')
    except:
        pass
    fit = np.array([np.array([proportion(p), compatibility(p), compactness(p), -difficulty_of_change(p)]) for p in P])
    labels = ['proportion', 'compatibility', 'compactness', 'difficulty of change']
    scatter_3d([fit[:, 0], fit[:, 1], fit[:, 2]], [labels[0], labels[1], labels[2]], 1)
    scatter_3d([fit[:, 0], fit[:, 1], -fit[:, 3]], [labels[0], labels[1], labels[3]], 2)
    scatter_3d([fit[:, 0], -fit[:, 3], fit[:, 2]], [labels[0], labels[3], labels[2]], 3)
    scatter_3d([-fit[:, 3], fit[:, 1], fit[:, 2]], [labels[3], labels[1], labels[2]], 4)
    scatter_2d([fit[:, 0], fit[:, 1]], [labels[0], labels[1]], 1)
    scatter_2d([fit[:, 0], fit[:, 2]], [labels[0], labels[2]], 2)
    scatter_2d([fit[:, 0], -fit[:, 3]], [labels[0], labels[3]], 3)
    scatter_2d([fit[:, 1], fit[:, 2]], [labels[1], labels[2]], 4)
    scatter_2d([fit[:, 1], -fit[:, 3]], [labels[1], labels[3]], 5)
    scatter_2d([fit[:, 2], -fit[:, 3]], [labels[2], labels[3]], 6)
    plot([range(H.shape[0]), [proportion(p) for p in H[:, 0, :, :]]], ['iteration', labels[0]], 1)
    plot([range(H.shape[0]), [compatibility(p) for p in H[:, 1, :, :]]], ['iteration', labels[1]], 2)
    plot([range(H.shape[0]), [compactness(p) for p in H[:, 2, :, :]]], ['iteration', labels[2]], 3)
    plot([range(H.shape[0]), [difficulty_of_change(p) for p in H[:, 3, :, :]]], ['iteration', labels[3]], 4)
    args = np.argmax(fit, axis=0)
    save_solution(P[args[0]], labels[0])
    save_solution(P[args[1]], labels[1])
    save_solution(P[args[2]], labels[2])
    save_solution(P[args[3]], labels[3])
    normalized = fit / np.ndarray.sum(fit, axis=0)
    weighted = np.argmax(np.sum(normalized, axis=1))
    table(labels, ['max ' + label for label in labels] + ['weighted max'], [[f"{normalized[args[i], j]:.4f}" for i in range(4)] + [f"{normalized[weighted, j]:.4f}"] for j in range(4)], 1)
    init = map_arr(properties['init']).astype(int)
    fit_init = np.array([proportion(init), compatibility(init), compactness(init), difficulty_of_change(init)])
    cells = np.zeros((4, 3))
    cells[:, 0] = fit_init
    cells[:, 1] = np.array([fit[args[i], i] for i in range(4)])
    cells[:, 2] = (cells[:, 1] - cells[:, 0]) / cells[:, 0] * 100
    table(labels, ['initial answer', 'best answer', 'improvement'], cells, 2)

try:
    os.mkdir('results')
except:
    pass

properties['crossover'] = crossover2


set_properties()
print(properties['constraints']['max_usage'])
print(properties['constraints']['min_usage'])
save_slopes()

pop_size = properties['population_size']
max_gen = properties['generations']
P = initial_population(pop_size)

# save_population(P, 0)
gen_no = 0
fit = np.array(list(map(fitness, P)))
data = []
while gen_no < max_gen:
    print(f"{gen_no}")
    C = properties['crossover'](P)
    fit_C = np.array(list(map(fitness, C)))
    P = np.concatenate([P, C], axis=0)
    fit = np.concatenate([fit, fit_C], axis=0)
    F, fit = fast_nondominated_sort(P, fit)
    F, fit = crowding_distance_sort(P, F, fit)
    for i in range(len(F)):
        kappa(P[F[i]], i + 1, gen_no)
    args = np.argmax(fit[0], axis=0)
    data.append(P[F[0][args]])
    idx = np.concatenate(F)
    fit = np.concatenate(fit)
    P = P[idx[:pop_size]]
    fit = fit[:pop_size]
    gen_no = gen_no + 1


F, fit = fast_nondominated_sort(P, fit)
F, fit = crowding_distance_sort(P, F, fit)
data = np.array(data)
report(P[F[0]], data)
for i in range(len(F)):
    kappa(P[F[i]], i + 1, gen_no)