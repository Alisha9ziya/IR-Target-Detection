import numpy as np

def compute_metrics(pred_mask, gt_mask):
    # Convert to boolean
    pred = pred_mask > 0
    gt = gt_mask > 0

    # Confusion components
    TP = np.sum(pred & gt)
    FP = np.sum(pred & (~gt))
    FN = np.sum((~pred) & gt)

    # Metrics
    PD = TP / (TP + FN + 1e-6)   # Probability of Detection
    FA = FP / (pred.size + 1e-6) # False Alarm Rate (normalized)

    return PD, FA