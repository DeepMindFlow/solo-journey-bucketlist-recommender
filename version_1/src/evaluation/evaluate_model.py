import logging
import numpy as np
from src.ml_algorithms.logistic_regression import LogisticRegression
from src.ml_algorithms.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                                       #plot_roc_curve, compute_roc_auc, plot_precision_recall_curve, plot_threshold_vs_metrics
                                       plot_roc_curve_plotly, plot_precision_recall_curve_plotly,plot_threshold_vs_metrics_plotly )
from src.feature_engineering import run_feature_engineering

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Train-Test split 80% training/ 20% Testing
X_train, X_test, y_train, y_test = run_feature_engineering()

X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=float)

logistic_reg_model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
logistic_reg_model.fit(X_train, y_train)

y_predictions = logistic_reg_model.predict_proba(X_test)
logging.info(f"Sample raw probs: {y_predictions[:10]}")
logging.info(f"Min prob: {np.min(y_predictions)}")
logging.info(f"Max prob: {np.max(y_predictions)}")
threshold = 0.35
y_pred = (y_predictions >= threshold).astype(int)

logging.info(f"\nLearned Weights: {logistic_reg_model.weights}")
logging.info(f"Min weight: {logistic_reg_model.weights.min():.4f}")
logging.info(f"Max weight: {logistic_reg_model.weights.max():.4f}")
logging.info(f"Mean weight: {logistic_reg_model.weights.mean():.4f}")
logging.info(f"First 10 weights: {logistic_reg_model.weights[:10]}")

logging.info("\n=== Step 3: Inspecting Predicted Probabilities ===")
logging.info(f"First 10 probabilities: {y_predictions[:10]}")
logging.info(f"Min Probability: {np.min(y_predictions):.4f}")
logging.info(f"Max Probability: {np.max(y_predictions):.4f}")
logging.info(f"Average Probability: {np.mean(y_predictions):.4f}")
logging.info("===================================================\n")

y_true = y_test.astype(int)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
confusion_matrix = confusion_matrix(y_true, y_pred)

'''plot_roc_curve(y_true, y_predictions)
plot_precision_recall_curve(y_true, y_predictions)
plot_threshold_vs_metrics(y_true, y_predictions)'''

# Instead of static matplotlib plots, generate Plotly interactive plots:
fig_roc = plot_roc_curve_plotly(y_true, y_predictions)
fig_precision_recall = plot_precision_recall_curve_plotly(y_true, y_predictions)
fig_threshold_vs_metrics = plot_threshold_vs_metrics_plotly(y_true, y_predictions)

# For local testing you can do:
# fig_roc.show()
# fig_pr.show()
# fig_threshold.show()

# Or save interactive HTML if desired:
fig_roc.write_html("images/roc_curve.html")
fig_precision_recall.write_html("images/precision_recall_curve.html")
fig_threshold_vs_metrics.write_html("images/threshold_vs_metrics.html")

logging.info("\n==== MODEL EVALUATION METRICS ====")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Confusion Matrix:\n{confusion_matrix}")
logging.info("---------------------------------")
logging.info("==== ROC AUC PLOT METRICS ARE FOUND IN IMAGES SUBFOLDER ====")
logging.info("---------------------------------")