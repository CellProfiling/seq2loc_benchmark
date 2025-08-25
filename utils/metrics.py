import numpy as np
from sklearn.metrics import matthews_corrcoef, average_precision_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import jaccard_score, roc_auc_score
from sklearn.metrics import label_ranking_average_precision_score, coverage_error


# Taken from DeepLoc2.0
# taken from https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf == 0:
        return 0
    else:
        return sup / np.sqrt(inf)


def get_mcc_threshold_perclass(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0

    y_pred = (y_prob >= best_proba).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    # print(score, best_mcc)
    # plt.plot(mccs)
    return best_proba


def get_mcc_threhsold(y_true, y_pred):
    thresholds = np.array(
        [
            get_mcc_threshold_perclass(y_true[:, i], y_pred[:, i])
            for i in range(y_true.shape[1])
        ]
    )
    return thresholds


def roc_auc(y_true, y_pred):
    # remove categories with no positive samples
    rocauc_perclass = roc_auc_score(y_true, y_pred, average=None)
    rocauc_macro = roc_auc_score(y_true, y_pred, average="macro")
    rocauc_micro = roc_auc_score(y_true, y_pred, average="micro")
    return rocauc_perclass, rocauc_macro, rocauc_micro


def get_all_metrics(y_true, y_pred, y_pred_bin, categories):
    cat_count = y_true.sum(axis=0)
    zero_count_cats = np.where(cat_count == 0)[0]

    y_true = np.delete(y_true, zero_count_cats, axis=1)
    y_pred = np.delete(y_pred, zero_count_cats, axis=1)
    y_pred_bin = np.delete(y_pred_bin, zero_count_cats, axis=1)
    cat_count = np.delete(cat_count, zero_count_cats)

    cat_perclass = [cat for i, cat in enumerate(categories) if i not in zero_count_cats]

    macro_ap = average_precision_score(y_true, y_pred, average="macro")
    micro_ap = average_precision_score(y_true, y_pred, average="micro")

    mcc_perclass = np.array(
        [
            matthews_corrcoef(y_true[:, i], y_pred_bin[:, i])
            for i in range(y_true.shape[1])
        ]
    )

    acc = (y_true == y_pred_bin).mean()
    acc_samples = (y_true == y_pred_bin).all(axis=1).mean()
    acc_perclass = (y_true == y_pred_bin).mean(axis=0)

    recall_perclass = np.array(
        [
            recall_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
            for i in range(y_true.shape[1])
        ]
    )
    precision_perclass = np.array(
        [
            precision_score(y_true[:, i], y_pred_bin[:, i], zero_division=0)
            for i in range(y_true.shape[1])
        ]
    )

    f1_perclass = f1_score(y_true, y_pred_bin, average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)

    jaccard_perclass = jaccard_score(y_true, y_pred_bin, average=None, zero_division=0)
    jaccard_macro = jaccard_score(y_true, y_pred_bin, average="macro", zero_division=0)
    jaccard_micro = jaccard_score(y_true, y_pred_bin, average="micro", zero_division=0)

    rocauc_perclass, rocauc_macro, rocauc_micro = roc_auc(y_true, y_pred)

    mlrap = label_ranking_average_precision_score(y_true, y_pred)
    cov_error = coverage_error(y_true, y_pred)

    num_labels = y_pred_bin.sum(axis=1).mean()

    return {
        "category_perclass": cat_perclass,
        "count_perclass": cat_count,
        "macro_ap": macro_ap,
        "micro_ap": micro_ap,
        "mcc_perclass": mcc_perclass,
        "acc": acc,
        "acc_samples": acc_samples,
        "acc_perclass": acc_perclass,
        "recall_perclass": recall_perclass,
        "precision_perclass": precision_perclass,
        "f1_perclass": f1_perclass,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "jaccard_perclass": jaccard_perclass,
        "jaccard_macro": jaccard_macro,
        "jaccard_micro": jaccard_micro,
        "rocauc_perclass": rocauc_perclass,
        "rocauc_macro": rocauc_macro,
        "rocauc_micro": rocauc_micro,
        "mlrap": mlrap,
        "coverage_error": cov_error,
        "num_labels": num_labels,
    }


def get_all_fold_metrics(y_true, y_pred_list, thresholds_list, categories):
    avg_y_pred = np.mean(y_pred_list, axis=0)

    avg_y_pred_bin = []
    for y_pred, thresholds in zip(y_pred_list, thresholds_list):
        y_pred_bin = (y_pred > thresholds).astype(np.int16)
        avg_y_pred_bin.append(y_pred_bin)
    avg_y_pred_bin = np.mean(avg_y_pred_bin, axis=0)
    avg_y_pred_bin = (avg_y_pred_bin > 0.5).astype(np.int16)

    return get_all_metrics(y_true, avg_y_pred, avg_y_pred_bin, categories)
