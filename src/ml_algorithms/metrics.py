# src/ml_algorithm/metrics.py
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# =====================
# Basic metric functions
# =====================
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred):
    true_positives = np.sum(y_true * y_pred)
    false_positives = np.sum(y_pred) - true_positives
    false_negatives = np.sum(y_true) - true_positives

    return np.array([[true_positives, false_positives], [false_negatives, true_positives]])


# =================================
# Compute ROC and AUC for thresholds
# =================================
def compute_roc_auc(y_true, y_probs, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_list = []
    fpr_list = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        TPR = TP / P if P > 0 else 0
        FPR = FP / N if N > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    auc = np.trapz(tpr_list, fpr_list)
    return np.array(fpr_list), np.array(tpr_list), thresholds, auc


def plot_roc_curve(y_true, y_probs, figsize=(8, 6)):
    fpr, tpr, thresholds, auc = compute_roc_auc(y_true, y_probs)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    project_root = pathlib.Path(__file__).parents[2]  # Adjust according to your folder depth
    image_dir = project_root / "images"
    image_dir.mkdir(exist_ok=True)  # Make sure folder exists
    plt.savefig(image_dir / "ROC.png")
    plt.clf()


# ====================================
# Compute precision and recall per threshold
# ====================================

def compute_precision_recall(y_true, y_probs, num_thresholds=100):
    thresholds = np.linspace(0, 1, num_thresholds)
    precision_list = []
    recall_list = []

    P = np.sum(y_true == 1)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / P if P > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    return np.array(precision_list), np.array(recall_list), thresholds



def plot_precision_recall_curve(y_true, y_probs, figsize=(8, 6)):
    precision, recall, thresholds = compute_precision_recall(y_true, y_probs)
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)

    project_root = pathlib.Path(__file__).parents[2]  # Adjust according to your folder depth
    image_dir = project_root / "images"
    image_dir.mkdir(exist_ok=True)  # Make sure folder exists
    plt.savefig(image_dir / 'precision-recall_curve.png')
    plt.clf()


def plot_threshold_vs_metrics(y_true, y_probs, num_thresholds=100, figsize=(8, 6)):
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)

        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        TPR = TP / P if P > 0 else 0
        FPR = FP / N if N > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / P if P > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)
        precision_list.append(precision)
        recall_list.append(recall)

    plt.figure(figsize=figsize)
    plt.plot(thresholds, tpr_list, label='TPR (Recall)')
    plt.plot(thresholds, fpr_list, label='FPR')
    plt.plot(thresholds, precision_list, label='Precision')
    plt.plot(thresholds, recall_list, label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Metric value')
    plt.title('Threshold vs Metrics')
    plt.legend()
    plt.grid(True)

    project_root = pathlib.Path(__file__).parents[2]  # Adjust according to your folder depth
    image_dir = project_root / "images"
    image_dir.mkdir(exist_ok=True)  # Make sure folder exists
    plt.savefig(image_dir / 'Threshold_vs_Metrics.png', dpi=300, bbox_inches='tight')
    plt.clf()




# ===============================
# Plotly interactive plotting functions
# ===============================

def plot_roc_curve_plotly(y_true, y_probs):
    """
    Create an interactive ROC curve plot using Plotly.
    Shows TPR vs FPR for various thresholds, with AUC in legend.
    """
    fpr, tpr, thresholds, auc = compute_roc_auc(y_true, y_probs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='darkorange')
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier',
        line=dict(color='navy', dash='dash')
    ))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Recall)',
        template='plotly_dark',
        width=700, height=500
    )
    return fig


def plot_precision_recall_curve_plotly(y_true, y_probs):
    """
    Create an interactive Precision-Recall curve using Plotly.
    """
    precision, recall, thresholds = compute_precision_recall(y_true, y_probs)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines', line=dict(color='blue')
    ))
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        template='plotly_dark',
        width=700, height=500
    )
    return fig

def plot_threshold_vs_metrics_plotly(y_true, y_probs, num_thresholds=100):
    """
    Plot interactive curves of threshold vs TPR, FPR, Precision, and Recall.
    Useful for selecting the best threshold visually.
    """
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_list = []
    fpr_list = []
    precision_list = []
    recall_list = []

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        TPR = TP / P if P > 0 else 0
        FPR = FP / N if N > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / P if P > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)
        precision_list.append(precision)
        recall_list.append(recall)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=tpr_list, mode='lines', name='TPR (Recall)'))
    fig.add_trace(go.Scatter(x=thresholds, y=fpr_list, mode='lines', name='FPR'))
    fig.add_trace(go.Scatter(x=thresholds, y=precision_list, mode='lines', name='Precision'))
    fig.add_trace(go.Scatter(x=thresholds, y=recall_list, mode='lines', name='Recall'))
    fig.update_layout(
        title='Threshold vs Metrics',
        xaxis_title='Threshold',
        yaxis_title='Metric Value',
        template='plotly_dark',
        width=700, height=500
    )
    return fig

# ======================================
# Optional Matplotlib backup functions
# Commented out, use if needed for static image generation
# ======================================

# def plot_roc_curve(y_true, y_probs, figsize=(8, 6)):
#     """Static ROC curve using Matplotlib (backup)."""
#     fpr, tpr, thresholds, auc = compute_roc_auc(y_true, y_probs)
#     plt.figure(figsize=figsize)
#     plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='darkorange')
#     plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate (Recall)')
#     plt.title('ROC Curve')
#     plt.legend()
#     plt.grid(True)
#     project_root = pathlib.Path(__file__).parents[2]
#     image_dir = project_root / "images"
#     image_dir.mkdir(exist_ok=True)
#     plt.savefig(image_dir / "ROC.png")
#     plt.clf()

# def plot_precision_recall_curve(y_true, y_probs, figsize=(8, 6)):
#     """Static Precision-Recall curve using Matplotlib (backup)."""
#     precision, recall, thresholds = compute_precision_recall(y_true, y_probs)
#     plt.figure(figsize=figsize)
#     plt.plot(recall, precision, color='blue')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.grid(True)
#     project_root = pathlib.Path(__file__).parents[2]
#     image_dir = project_root / "images"
#     image_dir.mkdir(exist_ok=True)
#     plt.savefig(image_dir / 'precision-recall_curve.png')
#     plt.clf()

# def plot_threshold_vs_metrics(y_true, y_probs, num_thresholds=100, figsize=(8, 6)):
#     """Static Threshold vs Metrics plot using Matplotlib (backup)."""
#     thresholds = np.linspace(0, 1, num_thresholds)
#     tpr_list = []
#     fpr_list = []
#     precision_list = []
#     recall_list = []
#     P = np.sum(y_true == 1)
#     N = np.sum(y_true == 0)
#     for thresh in thresholds:
#         y_pred = (y_probs >= thresh).astype(int)
#         TP = np.sum((y_pred == 1) & (y_true == 1))
#         FP = np.sum((y_pred == 1) & (y_true == 0))
#         FN = np.sum((y_pred == 0) & (y_true == 1))
#         TPR = TP / P if P > 0 else 0
#         FPR = FP / N if N > 0 else 0
#         precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
#         recall = TP / P if P > 0 else 0
#         tpr_list.append(TPR)
#         fpr_list.append(FPR)
#         precision_list.append(precision)
#         recall_list.append(recall)
#     plt.figure(figsize=figsize)
#     plt.plot(thresholds, tpr_list, label='TPR (Recall)')
#     plt.plot(thresholds, fpr_list, label='FPR')
#     plt.plot(thresholds, precision_list, label='Precision')
#     plt.plot(thresholds, recall_list, label='Recall')
#     plt.xlabel('Threshold')
#     plt.ylabel('Metric value')
#     plt.title('Threshold vs Metrics')
#     plt.legend()
#     plt.grid(True)
#     project_root = pathlib.Path(__file__).parents[2]
#     image_dir = project_root / "images"
#     image_dir.mkdir(exist_ok=True)
#     plt.savefig(image_dir / 'Threshold_vs_Metrics.png', dpi=300, bbox_inches='tight')
#     plt.clf()