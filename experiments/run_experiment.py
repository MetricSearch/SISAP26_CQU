import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
import json
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import falconn
import time

MF_DINO2 = "mf_dino2"
MF_DINO2_GENERAL_EUC = "mf_dino2_general_euc"
GLOVE = "glove"
GLOVE_GENERAL_EUC = "glove_general_euc"
GOOAQ = "gooaq"
UNIFORM = "uniform"
PUBMED = "pubmed"
MF_EDGEHISTOS = "mf_edgehistos"
LAION = "laion"
LAION_GENERAL_EUC = "laion_general_euc"
SIFT = "sift"
SIFT_GENERAL_EUC = "sift_general_euc"

AVAILABLE_DATASETS = [MF_DINO2, MF_DINO2_GENERAL_EUC, MF_EDGEHISTOS, GLOVE, GLOVE_GENERAL_EUC, GOOAQ, PUBMED, UNIFORM, LAION, LAION_GENERAL_EUC, SIFT, SIFT_GENERAL_EUC]

# Dataset layout. Override DATASET_ROOT when the datasets are stored elsewhere.
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", "/datasets"))
SIFT_DATASET_PATH = DATASET_ROOT / "sift" / "sift_base.fvecs"
MF_DINO2_DATASET_DIR = DATASET_ROOT / "mf_dino2"
GLOVE_DATASET_PATH = DATASET_ROOT / "twitter_glove" / "twitter_glove_100d.npy"
MF_EDGEHISTOS_DATASET_PATH = (
    DATASET_ROOT / "mf_mpeg7_edgehistos" / "all_edgehistogram_data.npy"
)
PUBMED_DATASET_PATH = DATASET_ROOT / "pubmed" / "benchmark-dev-pubmed23.h5"
GOOAQ_DATASET_PATH = DATASET_ROOT / "gooaq" / "benchmark-dev-gooaq.h5"
LAION_DATASET_PATH = (
    DATASET_ROOT / "laion" / "laion-10M" / "laion2B-en-pca96v2-n=10M.h5"
)

RESULTS_DIR = Path(
    os.environ.get("RESULTS_ROOT", Path(__file__).resolve().parent)
)

VANILLA_PATH = "vanilla_lsh.json"
MTQ_PATH = "mtq_lsh.json"
CQU_LSH_PATH = "cqu_lsh.json"

# Elide recall threshold
RECALL_THRESHOLD = 0.0
N_QUERIES = 250

# https://gist.github.com/danoneata/49a807f47656fedbb389
def get_sift_general_euc(c_contiguous=True):
    fv = np.fromfile(SIFT_DATASET_PATH, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError(f"Non-uniform vector sizes in {SIFT_DATASET_PATH}")
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
            
    print(f"Pre-uniqued size: {fv.shape[0]}")
    fv = np.unique(fv, axis=0)
    print(f"Post uniqued size: {fv.shape[0]}")
        
    return fv

def get_sift(c_contiguous=True):
    vecs = get_sift_general_euc(c_contiguous)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

def get_mf_dino2():
    data = np.zeros((0, 384), dtype=np.float32)

    # Load Dino2
    for i in range(100):
        fp = loadmat(MF_DINO2_DATASET_DIR / f"{i}.mat")['features']
        fp /= np.linalg.norm(fp, axis=1, keepdims=True)
        data = np.vstack((data, fp))

    return data

def get_mf_dino2_general_euc():
    data = np.zeros((0, 384), dtype=np.float32)

    # Load Dino2
    for i in range(100):
        fp = loadmat(MF_DINO2_DATASET_DIR / f"{i}.mat")['features']
        data = np.vstack((data, fp))

    return data

def get_glove_general_euc():
    data = np.load(GLOVE_DATASET_PATH)
    data = data.astype(np.float32)
    # We do NOT l2 norm
    # data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data

def get_mpeg7_edgehistos():
    data = np.load(MF_EDGEHISTOS_DATASET_PATH)
    data = data.astype(np.float32)

    data -= np.mean(data)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data

def get_uniform():
    data = np.random.normal(0, 1, size=(1_000_000, 200)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data

def get_glove():
    data = np.load(GLOVE_DATASET_PATH)
    data = data.astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)

    return data

def get_pubmed():
    with h5py.File(PUBMED_DATASET_PATH, "r") as f:
        data = f["train"][:]
        
    data = np.ascontiguousarray(data)
    # data = data.astype(np.float32)
    # data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    print(f"Loaded pubmed")
    return data

def get_gooaq():
    with h5py.File(GOOAQ_DATASET_PATH, "r") as f:
        data = f["train"][:]
    data = data.astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data

def get_laion():
    with h5py.File(LAION_DATASET_PATH, 'r') as hdf:
        data = hdf['pca96'][:]
        data /= np.linalg.norm(data, axis=1, keepdims=True)

    data = np.ascontiguousarray(data)        
    print(f"Pre-uniqued size: {data.shape[0]}")
    
    data = np.unique(data, axis=0)
    print(f"Post uniqued size: {data.shape[0]}")
        
    return data

def get_laion_general_euc():
    with h5py.File(LAION_DATASET_PATH, 'r') as hdf:
        data = hdf['pca96'][:]
        data = data[:3_000_000]
        
    print(f"Pre-uniqued size: {data.shape[0]}")
    
    data = np.unique(data, axis=0)
    print(f"Post uniqued size: {data.shape[0]}")
        
    return data

def scramble_separate_queries(data, n_queries):
    """
    Scrambles the data by swapping n_queries random elements to the end.
    Returns (data, queries)
    """
    n = len(data)
    # Swap the chosen random rows with the last n_queries rows so that the
    # chosen rows end up at the end (and the previous tail rows move to the
    # chosen positions).
    #
    # To make this a true swap (no overlap issues), sample only from the
    # non-tail part.
    ids = np.random.choice(n - n_queries, n_queries, replace=False)
    tail_ids = np.arange(n - n_queries, n)

    for src, dst in zip(ids, tail_ids):
        tmp = data[src].copy()
        data[src] = data[dst]
        data[dst] = tmp
    
    return data[:n - n_queries], data[n - n_queries:]

def brute_force_knn(q, data, k: int, work=None):
    if work is None:
        work = np.empty(data.shape[0], dtype=np.float32)

    np.dot(data, q, out=work) 
    work *= -1.0                

    idx = np.argpartition(work, kth=k - 1)[:k]

    idx = idx[np.argsort(work[idx])]
    return idx

# def brute_force_knn(q, data, k: int):
#     gt_dists = 1 - np.dot(data, q)
#     gt_ids = np.argpartition(gt_dists, k)[:k]
#     return gt_ids

def build_and_measure_memory_deterministic(params, data):
    """
    Builds the index 10 times and measures space requirements for each.
    """
    start = time.time_ns()
    index = falconn.LSHIndex(params)
    index.setup(data)
    end = time.time_ns()
    build_time_ms = (end - start) / 1_000_000
    
    size_mb = index.get_composite_hash_table_size_bytes() / (1024 * 1024)
    
    return build_time_ms, size_mb, index

def build_falconn(data, num_hash_tables, probes_per_table, num_hash_bits):
    n = data.shape[0]
    d = data.shape[1]
    
    params = falconn.get_default_parameters(n, d, falconn.DistanceFunction.NegativeInnerProduct, True)

    # Setup the index
    params.num_setup_threads = 1
    params.lsh_family = falconn.LSHFamily.CrossPolytope
    params.dimension = d    
    params.distance_function = falconn.DistanceFunction.NegativeInnerProduct
    params.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    params.num_rotations = 1
    params.l = num_hash_tables
    

    falconn.compute_number_of_hash_functions(int(num_hash_bits), params)
    
    print(f"Hashes: {params.k}")
    print(f"Last CP dimension: {params.last_cp_dimension}")
    
    # Escape hatch for invalid config - if num probes is less than num tables
    if probes_per_table < params.l:
        return -1, -1, None
    
    return build_and_measure_memory_deterministic(params, data)

def measure_falconn(index, queries, gt_nns, total_probes, warmup=None, patience=None):
    nns_to_find = 100
    
    query_object = index.construct_query_object(total_probes)
    
    results = []
    
    start = time.time_ns()
    
    if warmup is None and patience is None: # Regular LSH
        for qid in range(N_QUERIES):
            res = query_object.find_k_nearest_neighbors(
                                                    queries[qid], 
                                                    k=nns_to_find)
            results.append(res)
    elif warmup is not None: # MTQ LSH
        for qid in range(N_QUERIES):
            res = query_object.find_k_nearest_neighbours_mtq(
                                                        queries[qid], 
                                                        k=nns_to_find,
                                                        warmup=warmup)
            results.append(res)
    else: # CQU-LSH
        for qid in range(N_QUERIES):
            res = query_object.find_k_nearest_neighbours_cqu_lsh(
                                                            queries[qid], 
                                                            k=nns_to_find,
                                                            patience=patience)
            results.append(res)
            
    end = time.time_ns()
    
    time_elapsed_ms = (end - start) / 1_000_000
    # Time per query (ms)
    time_ms_per_query = time_elapsed_ms / N_QUERIES
    
    recalls = []
        
    # Measure recall
    for i in range(len(results)):
        approx_result = results[i]
        gt = gt_nns[i]
        
        recall = len(np.intersect1d(approx_result, gt)) / len(gt)
        recalls.append(recall)
    
    
    return np.mean(recalls), time_ms_per_query

def write_to_json(path, observation):
    """
    Observation should be a dict of
    
    {
        recall: int
        size: int
        time: float
    }
    """
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []
        
    all_results.append(observation)
    
    # Writeback
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2, default=int)
    
    
def run_experiments(dataset_name, n_queries, data):    
    data = np.ascontiguousarray(data)
    
    data, queries = scramble_separate_queries(data, n_queries)    
    print(f"Dataset {dataset_name} | Data Shape: {data.shape} | Queries Shape: {queries.shape}")
    k = 100
    
    true_nns = []
    work = np.empty(data.shape[0], dtype=np.float32)
    true_nns = np.empty((len(queries),k), dtype=np.int32)
    
    for i, q in enumerate(tqdm(queries)):
        true_nns[i] = brute_force_knn(q, data, k, work=work).astype(np.int32, copy=False)
    
    # Defines how much we should look. Go between [1/8, 1024] points per cell on average
    log2_data_size = np.log2(data.shape[0])
    lb_bits = int(log2_data_size - 5)
    ub_bits = int(log2_data_size + 6)
    
    # Defines hyperparameter space
    param_dist = {
        'num_hash_tables': [1, 2, 3, 4],   # 4
        'num_hash_bits': np.arange(lb_bits, ub_bits, 1), 
        'probes_per_table': [1, 25, 50, 75, 100, 150, 200, 250, 300], # 10
    }
    
    print(f"Found knns")
    
    # Total = 16,000
    ps = ParameterGrid(param_dist)

    for params in tqdm(ps):
        build_time_ms, size_mb, index = build_falconn(
            data,
            num_hash_tables=int(params["num_hash_tables"]),
            probes_per_table=int(params["probes_per_table"]),
            num_hash_bits=int(params["num_hash_bits"]),
        )

        if index is None:
            continue
        
        params["build_time_ms"] = build_time_ms
        
        print(f"Running vanilla with params={params}")
        total_probes = int(params["probes_per_table"]) * int(params["num_hash_tables"])

        # Run experiments        
        vanilla_recall, vanilla_time = measure_falconn(index, queries, true_nns, total_probes)

        # Output to JSON files
        if vanilla_recall > RECALL_THRESHOLD:
            new_param_dist = params
            new_param_dist['recall'] = vanilla_recall
            new_param_dist['size_mb'] = size_mb
            new_param_dist['time_ms'] = vanilla_time
            write_to_json(RESULTS_DIR / dataset_name / VANILLA_PATH, new_param_dist)
                        
        # Handles MTQ params
        for warmup in [0, 2 * k, 4 * k, 8 * k, 16 * k, 32 * k]:
            params['warmup'] = warmup

            print(f"Running MTQ with params={params}")
            mtq_recall, mtq_time = measure_falconn(index, queries, true_nns, total_probes, params['warmup'])

            
            print(f"MTQ recall: {mtq_recall}")
            if mtq_recall > RECALL_THRESHOLD:
                new_param_dist = params
                new_param_dist['recall'] = mtq_recall
                new_param_dist['size_mb'] = size_mb
                new_param_dist['time_ms'] = mtq_time
                write_to_json(RESULTS_DIR / dataset_name / MTQ_PATH, new_param_dist)

            for patience in [0, 1, 2, 3, 4]:
                
                # If patience is bigger than hash tables, some tables are checked twice pointlessly
                if patience > params['num_hash_tables']:
                    continue
                
                params['patience'] = patience
                
                # CQU-LSH updates the query after the first pass over all tables.
                if warmup == 0:
                    print(f"Running CQU-LSH with params={params}")
                    cqu_recall, cqu_time = measure_falconn(
                        index,
                        queries,
                        true_nns,
                        total_probes,
                        patience=params['patience'],
                    )
                    
                    new_param_dist = params
                    new_param_dist['recall'] = cqu_recall
                    new_param_dist['size_mb'] = size_mb
                    new_param_dist['time_ms'] = cqu_time
                    write_to_json(
                        RESULTS_DIR / dataset_name / CQU_LSH_PATH,
                        new_param_dist,
                    )
                    

def get_parser():
    parser = argparse.ArgumentParser(description=f'Cross polytope k-nn search experiments. Available datasets: {AVAILABLE_DATASETS}')

    parser.add_argument('dataset', type=str,
                        help='The name of the dataset to load')

    return parser

if __name__ == "__main__":
    np.random.seed(91289)
    

    parser = get_parser()
    args = parser.parse_args()

    dataset = args.dataset
    
        
    if dataset == MF_DINO2:
        # Run the dino2 experiments
        run_experiments(dataset, N_QUERIES, get_mf_dino2())
    elif dataset == MF_DINO2_GENERAL_EUC:
        # Run the dino2 experiments
        run_experiments(dataset, N_QUERIES, get_mf_dino2_general_euc())
    elif dataset == GLOVE:
        run_experiments(dataset, N_QUERIES, get_glove())
    elif dataset == GLOVE_GENERAL_EUC:
        run_experiments(dataset, N_QUERIES, get_glove_general_euc())
    elif dataset == GOOAQ:
        run_experiments(dataset, N_QUERIES, get_gooaq())
    elif dataset == UNIFORM:
        run_experiments(dataset, N_QUERIES, get_uniform())
    elif dataset == PUBMED:
        run_experiments(dataset, N_QUERIES, get_pubmed())
    elif dataset == MF_EDGEHISTOS:
        run_experiments(dataset, N_QUERIES, get_mpeg7_edgehistos())
    elif dataset == LAION:
        run_experiments(dataset, N_QUERIES, get_laion())
    elif dataset == LAION_GENERAL_EUC:
        run_experiments(dataset, N_QUERIES, get_laion_general_euc())
    elif dataset == SIFT:
        run_experiments(dataset, N_QUERIES, get_sift())
    elif dataset == SIFT_GENERAL_EUC:
        run_experiments(dataset, N_QUERIES, get_sift_general_euc())

    else:
        raise RuntimeError(f"Unknown dataset. Datasets are: {AVAILABLE_DATASETS}")
