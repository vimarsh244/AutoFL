from torch.utils.data import Subset
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import as_classification_dataset, AvalancheDataset
# from avalanche.benchmarks.utils import make_classification_dataset

# def split_dataset(dataset, n_experiences):
#     n = len(dataset)
#     indices = list(range(n))
#     tls = [0 for _ in range(len(dataset))] # one task label for each sample
# #    avl_dataset = make_classification_dataset(dataset, task_labels=tls)
#     chunk_size = n // n_experiences
#     splits = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(n_experiences-1)]
#     splits.append(indices[(n_experiences-1)*chunk_size:])  # last chunk may be larger
#     return [Subset(dataset, idxs) for idxs in splits]
# 
def split_dataset(dataset, n_experiences):
    n = len(dataset)
    indices = list(range(n))
    chunk_size = n // n_experiences
    splits = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(n_experiences-1)]
    splits.append(indices[(n_experiences-1)*chunk_size:])  # Last split takes the remainder

    avalanche_experiences = [AvalancheDataset(Subset(dataset, idxs)) for idxs in splits]
    return avalanche_experiences
