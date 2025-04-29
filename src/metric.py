import numpy as np

def recall_at_k(true_labels, predicted_labels, k):
    """
    Compute Recall@k

    Args:
    - true_labels (list): A list where each element is a list of relevant document indices for a query.
    - predicted_labels (list): A list where each element is a list of predicted document indices for a query.
    - k (int): The number of top documents to consider for the metric.

    Returns:
    - recall (float): The average Recall@k score over all queries.
    """
    recalls = []

    for true, pred in zip(true_labels, predicted_labels):
        if type(true) == str:
            true = [true]
        if type(true) == int:
            true = [true]

        pred_k = pred[:k]
        if not true:
            recalls.append(0.0)
        else:
            recall = len(set(pred_k) & set(true)) / len(true)
            recalls.append(recall)
    return np.mean(recalls)


def ndcg_at_k(true_labels, predicted_labels, k):
    """
    Compute NDCG@k

    Args:
    - true_labels (list): A list where each element is a list of relevant document indices for a query.
    - predicted_labels (list): A list where each element is a list of predicted document indices for a query.
    - k (int): The number of top documents to consider for the metric.

    Returns:
    - ndcg (float): The average NDCG@k score over all queries.
    """

    def dcg(rel_scores):
        rel_scores = np.array(rel_scores)
        discounts = np.log2(np.arange(len(rel_scores)) + 2)
        return np.sum((2 ** rel_scores - 1) / discounts)

    def idcg(n_relevant):
        # The ideal DCG is obtained by taking the highest possible relevance scores
        ideal_rel = [1] * n_relevant + [0] * (k - n_relevant)
        return dcg(ideal_rel)

    ndcgs = []
    for true, pred in zip(true_labels, predicted_labels):
        if type(true) == str:
            true = [true]
        if type(true) == int:
            true = [true]

        rel_scores = [(1 if p in true else 0) for p in pred[:k]]
        actual_idcg = idcg(min(len(true), k))
        actual_dcg = dcg(rel_scores)
        if actual_idcg == 0:
            ndcgs.append(0.0)
        else:
            ndcgs.append(actual_dcg / actual_idcg)
    return np.mean(ndcgs)


