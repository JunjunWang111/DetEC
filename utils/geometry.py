# utils/geometry.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_local_density(coords, k=20, sigma=2.0):
    n = len(coords)
    if n <= 1:
        return np.zeros(n)
    k_actual = min(k, n-1)
    nbrs = NearestNeighbors(n_neighbors=k_actual+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    dists = distances[:, 1:]
    weights = np.exp(- (dists ** 2) / (2 * sigma ** 2))
    rho = weights.mean(axis=1)
    return rho

def compute_curvature(coords, k=20):
    n = len(coords)
    if n <= 3:
        return np.zeros(n)
    k_actual = min(k, n-1)
    nbrs = NearestNeighbors(n_neighbors=k_actual+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    curvatures = []
    for i in range(len(coords)):
        neigh_coords = coords[indices[i, 1:]]
        centered = neigh_coords - neigh_coords.mean(axis=0)
        cov = centered.T @ centered / (k_actual-1)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]
        curv = eigenvalues[2] / (eigenvalues.sum() + 1e-8)
        curvatures.append(curv)
    return np.array(curvatures)

def compute_local_frames(coords, k=20):
    n = len(coords)
    if n <= 3:
        return np.zeros((n, 3, 3))
    k_actual = min(k, n-1)
    nbrs = NearestNeighbors(n_neighbors=k_actual+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    frames = []
    for i in range(len(coords)):
        neigh_coords = coords[indices[i, 1:]]
        centered = neigh_coords - coords[i]
        U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)
        frames.append(U.T)
    return np.array(frames)
