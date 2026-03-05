import numpy as np
import libmr
import sys
import scipy.spatial.distance
import sklearn.metrics.pairwise
import time
import argparse
import itertools as it
import warnings
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count

# Suppress math warnings from scikit-learn
warnings.filterwarnings('ignore', category=RuntimeWarning)

@contextmanager
def timer(message):
    """
    Simple timing method. Logging should be used instead for large scale experiments.
    """
    print(message)
    start = time.time()
    yield
    stop = time.time()
    print("...elapsed time: {:.4f} seconds".format(stop-start))


def matrix_logarithm(C):
    """Computes the matrix logarithm for an SPD matrix via eigendecomposition."""
    # Ensure perfect symmetry to avoid numerical floating point errors
    C = (C + C.T) / 2.0
    evals, evecs = np.linalg.eigh(C)
    # Clip eigenvalues to avoid log(0) on edges of the manifold
    evals = np.maximum(evals, 1e-10)
    log_evals = np.diag(np.log(evals))
    return evecs @ log_evals @ evecs.T

def transform_to_tangent_space(X):
    """Reshapes flattened matrices, applies logm, and flattens them again."""
    # Dynamically find the window size (W) from the flattened vector length (W^2)
    W = int(np.sqrt(X.shape[1]))
    X_mat = X.reshape(-1, W, W)
    
    # Prepare array for the mapped tangent vectors
    X_log = np.zeros((X_mat.shape[0], X.shape[1]))
    
    for i in range(X_mat.shape[0]):
        X_log[i] = matrix_logarithm(X_mat[i]).flatten()
        
    return X_log

def log_euclidean_cdist(X, Y):
    X_log = transform_to_tangent_space(X)
    Y_log = transform_to_tangent_space(Y)
    return sklearn.metrics.pairwise.pairwise_distances(X_log, Y_log, metric="euclidean", n_jobs=1)

def log_euclidean_pdist(X):
    X_log = transform_to_tangent_space(X)
    return sklearn.metrics.pairwise.pairwise_distances(X_log, metric="euclidean", n_jobs=1)


def euclidean_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)
def euclidean_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)
def cosine_cdist(X, Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)
def cosine_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)

# Add our new metric to the lookup dictionary
dist_func_lookup = {
    "cosine": {"cdist": cosine_cdist, "pdist": cosine_pdist},
    "euclidean": {"cdist": euclidean_cdist, "pdist": euclidean_pdist},
    "log_euclidean": {"cdist": log_euclidean_cdist, "pdist": log_euclidean_pdist}
}

parser = argparse.ArgumentParser()
parser.add_argument("--tailsize", type=int, help="number of points that constitute 'extrema'", default=50)
parser.add_argument("--cover_threshold", type=float, help="probabilistic threshold to designate redundancy between points", default=0.5)
# Set the default distance to our new log_euclidean metric!
parser.add_argument("--distance", type=str, default="log_euclidean", choices=dist_func_lookup.keys())
parser.add_argument("--nfuse", type=int, help="number of extreme vectors to fuse over", default=4)
parser.add_argument("--margin_scale", type=float, help="multiplier by which to scale the margin distribution", default=0.5)

args = parser.parse_args()
tailsize = args.tailsize
cover_threshold = args.cover_threshold
cdist_func = dist_func_lookup[args.distance]["cdist"]
pdist_func = dist_func_lookup[args.distance]["pdist"]
num_to_fuse = args.nfuse
margin_scale = args.margin_scale

def set_cover_greedy(universe, subsets, cost=lambda x:1.0):
    universe = set(universe)
    subsets = [set(s) for s in subsets]
    covered = set()
    cover_indices = []
    while covered != universe:
        max_index = np.array([len(x - covered) for x in subsets]).argmax()
        covered |= subsets[max_index]
        cover_indices.append(max_index)
    return cover_indices

def set_cover(points, weibulls, solver=set_cover_greedy):
    universe = list(range(len(points)))
    d_mat = pdist_func(points)
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel, list(zip(d_mat, weibulls))))
    p.close()
    p.join()
    thresholded = list(zip(*np.where(probs >= cover_threshold)))
    subsets = {k: tuple(set(x[1] for x in v)) for k,v in it.groupby(thresholded, key=lambda x:x[0])}
    subsets = [subsets.get(i, ()) for i in universe]
    keep_indices = solver(universe, subsets)
    return keep_indices

def reduce_model(points, weibulls, labels, labels_to_reduce=None):
    if cover_threshold >= 1.0:
        return points, weibulls, labels
    ulabels = np.unique(labels)
    if labels_to_reduce is None:
        labels_to_reduce = ulabels
    labels_to_reduce = set(labels_to_reduce)
    keep = np.array([], dtype=int)
    for ulabel in ulabels:
        ind = np.where(labels == ulabel)
        if ulabel in labels_to_reduce: 
            print("...reducing model for label {}".format(ulabel))
            keep_ind = set_cover(points[ind], [weibulls[i] for i in ind[0]])
            keep = np.concatenate((keep, ind[0][keep_ind]))
        else:
            keep = np.concatenate((keep, ind[0]))
    points = points[keep]
    weibulls = [weibulls[i] for i in keep]
    labels = labels[keep]
    return points, weibulls, labels

def weibull_fit_parallel(args):
    global tailsize
    dists, row, labels = args
    nearest = np.partition(dists[np.where(labels != labels[row])], tailsize)
    mr = libmr.MR()
    mr.fit_low(nearest, tailsize)
    return str(mr)

def weibull_eval_parallel(args):
    dists, weibull_params = args
    mr = libmr.load_from_string(weibull_params)
    probs = mr.w_score_vector(dists)
    return probs

def fuse_prob_for_label(prob_mat, num_to_fuse):
    return np.average(np.partition(prob_mat, -num_to_fuse, axis=0)[-num_to_fuse:,:], axis=0)

def fit(X, y):
    global margin_scale
    d_mat = margin_scale * pdist_func(X)
    p = Pool(cpu_count())
    row_range = list(range(len(d_mat)))
    args = list(zip(d_mat, row_range, [y for i in row_range]))
    with timer("...getting weibulls"):
        weibulls = p.map(weibull_fit_parallel, args)
    p.close()
    p.join()
    return weibulls

def predict(X, points, weibulls, labels):
    global num_to_fuse
    d_mat = cdist_func(points, X).astype(np.float64)
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel, list(zip(d_mat, weibulls))))
    p.close()
    p.join()
    ulabels = np.unique(labels)
    fused_probs = []
    for ulabel in ulabels:
        fused_probs.append(fuse_prob_for_label(probs[np.where(labels == ulabel)], num_to_fuse))
    fused_probs = np.array(fused_probs)
    max_ind = np.argmax(fused_probs, axis=0)
    predicted_labels = ulabels[max_ind]
    return predicted_labels, fused_probs

def load_data(fname):
    """Robust Python 3 loader using numpy"""
    raw_data = np.loadtxt(fname, delimiter=",", dtype=str)
    labels = raw_data[:, 0]
    data = raw_data[:, 1:].astype(float)
    return data, labels

def get_accuracy(predictions, labels):
    return sum(predictions == labels) / float(len(predictions))

def letter_test(train_fname, test_fname):
    with timer("...loading train data"):
        Xtrain, ytrain = load_data(train_fname)
        print("Train shape:", Xtrain.shape)
        
    with timer("...loading test data"):
        Xtest, ytest = load_data(test_fname)
        print("Test shape:", Xtest.shape)      
        
    with timer("...fitting train set"):
        weibulls = fit(Xtrain, ytrain)
        
    with timer("...reducing model"):
        Xtrain, weibulls, ytrain = reduce_model(Xtrain, weibulls, ytrain)
    print("...model size: {}".format(len(ytrain)))
    
    with timer("...getting predictions"):
        predictions, probs = predict(Xtest, Xtrain, weibulls, ytrain)
        
    with timer("...evaluating predictions"):
        accuracy = get_accuracy(predictions, ytest)
        
    print("accuracy: {:.4f}".format(accuracy))
    return accuracy

# UPDATE THESE PATHS to point to your new SPD files
#test_data_path = "/Users/bamorim/Documents/GitHub/Extreme-Value-Machine/test_spd_w4.txt"
#train_data_path = "/Users/bamorim/Documents/GitHub/Extreme-Value-Machine/train_spd_w4.txt"

#test_data_path = "test_spd_w4.txt"
#train_data_path = "train_spd_w4.txt"

test_data_path = "test_synthetic_spd_w4.txt"
train_data_path = "train_synthetic_spd_w4.txt"

if __name__ == "__main__":
    letter_test(train_data_path, test_data_path)