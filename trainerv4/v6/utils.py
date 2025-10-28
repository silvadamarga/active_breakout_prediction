#!/usr/bin/env python3
"""
v6 - Utility Functions
Helpers for logging, GPU config, analysis, saving results.
"""

import os
import logging
import pickle
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from typing import List, Dict, Tuple, Any

# Import config (assuming it's in the same directory level)
from .config import (
    LOG_FILENAME_TPL, JSON_SUMMARY_FILENAME_TPL, EQUITY_CURVE_FILENAME_TPL,
    RANDOM_STATE # Needed if setting seeds here
)

# Added imports needed for inspect_label_quality
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ---------------- Logging Setup ----------------
# Moved to main.py to avoid potential circular imports if utils needs logging


# ---------------- GPU Configuration ----------------
def configure_gpu_memory_growth():
    """Configures TensorFlow GPU memory growth."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"Enabled memory growth for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logging.error(f"Could not set GPU memory growth: {e}")
    else:
        logging.info("No GPUs detected by TensorFlow.")


# ---------------- Diagnostics & Analysis ----------------
def inspect_label_quality(y_true: np.ndarray, y_pred_proba: np.ndarray, save_prefix='diagnostic'):
    """
    Analyzes label distribution and calculates ROC/PR AUC for multi-class.

    Args:
        y_true (np.ndarray): True labels (0, 1, 2).
        y_pred_proba (np.ndarray): Predicted probabilities (N, 3).
        save_prefix (str): Prefix for saving plot filenames.
    """
    if y_true is None or y_pred_proba is None or len(y_true) == 0 or len(y_pred_proba) == 0:
        logging.warning("inspect_label_quality: Input arrays are empty or None. Skipping.")
        return {}

    unique_labels, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    label_map = {0: 'Loss', 1: 'Timeout', 2: 'Win'}
    logging.info("Label counts -> total: %d", total)
    for label_val, count in zip(unique_labels, counts):
        logging.info("  %s: %d (%.2f%%)", label_map.get(label_val, 'Unknown'), count, count / total * 100 if total else 0)

    # --- Multi-class ROC/PR AUC Calculation ---
    results = {}
    n_classes = y_pred_proba.shape[1]

    if len(unique_labels) < 2:
        logging.warning("Only one class present in y_true. Skipping ROC/PR calculations.")
        return results
    if n_classes < 2:
        logging.warning(f"y_pred_proba has only {n_classes} columns. Cannot calculate multi-class metrics.")
        return results

    try:
        # One-vs-Rest ROC AUC (Macro Average)
        roc_auc_ovr_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        results['roc_auc_ovr_macro'] = roc_auc_ovr_macro
        logging.info(f"Model Stats -> ROC AUC (OvR Macro): {roc_auc_ovr_macro:.4f}")

        # Plot ROC for each class
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_pred_proba[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {label_map.get(i,"?")} (AUC={class_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curves (Macro AUC={roc_auc_ovr_macro:.3f})')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend()

        # Plot PR for each class (Micro average PR AUC is equivalent to Average Precision)
        plt.subplot(1, 2, 2)
        avg_precision = 0
        for i in range(n_classes):
             precision, recall, _ = precision_recall_curve((y_true == i).astype(int), y_pred_proba[:, i])
             class_pr_auc = auc(recall, precision)
             avg_precision += class_pr_auc * (counts[unique_labels == i][0] / total if total > 0 else 0) # Weighted average
             plt.plot(recall, precision, label=f'Class {label_map.get(i,"?")} (AUC={class_pr_auc:.2f})')

        results['pr_auc_avg_precision'] = avg_precision # Store micro-average PR AUC
        logging.info(f"Model Stats -> Average Precision (PR AUC Micro): {avg_precision:.4f}")
        plt.title(f'PR Curves (Avg Precision={avg_precision:.3f})')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_roc_pr_curves.png'); plt.close()
        logging.info(f"Saved {save_prefix}_roc_pr_curves.png")

    except Exception as e:
        logging.error(f"Error calculating/plotting ROC/PR curves: {e}", exc_info=False)

    return results


def compute_breakout_sample_weights(y):
    """Computes balanced class weights for the breakout model target."""
    classes = np.unique(y)
    if len(classes) < 2: return np.ones_like(y, dtype=float) # Avoid division by zero if only one class
    weights = compute_class_weight('balanced', classes=classes, y=y)
    # Create a mapping from class label to weight
    weight_map = {int(cls): float(w) for cls, w in zip(classes, weights)}
    # Apply weights based on the mapping
    sample_weights = np.array([weight_map.get(label, 1.0) for label in y], dtype=np.float32)
    return sample_weights


def analyze_performance_by_regime(trades_df: pd.DataFrame, regime_map: dict) -> dict:
    """Analyzes backtest trade performance grouped by market regime."""
    if trades_df is None or trades_df.empty: return {}
    logging.info("--- Performance by Regime ---")
    regime_names = {k: v for k, v in regime_map.items()} # Ensure correct mapping
    # Handle potential float regimes if prediction wasn't integer
    trades_df['entry_regime_int'] = trades_df['entry_regime'].fillna(-1).astype(int)
    trades_df['regime_name'] = trades_df['entry_regime_int'].map(regime_names).fillna('Unknown')

    perf_stats = {}
    for name, group in trades_df.groupby('regime_name'):
        total_trades = len(group)
        if total_trades == 0: continue
        win_rate = (group['profit'] > 0).mean() * 100 if total_trades > 0 else 0.0
        avg_profit = group['profit'].mean() if total_trades > 0 else 0.0
        total_profit = group['profit'].sum()
        stats = {
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'average_profit': avg_profit,
            'total_profit': total_profit
        }
        perf_stats[name] = stats
        logging.info(f"Regime '{name}': {total_trades} trades, Win Rate: {win_rate:.2f}%, Avg Profit: {avg_profit:.2f}, Total Profit: {total_profit:.2f}")
    return perf_stats


# ---------------- Saving & Reporting ----------------
def save_results_summary(
    json_filename: str,
    run_id: str,
    strategy_params: dict,
    financial_metrics: dict,
    statistical_metrics: dict,
    best_hps: dict
):
    """Saves a JSON summary of the run."""
    summary = {
        'run_id': run_id,
        'run_timestamp': datetime.now().isoformat(),
        'strategy_parameters': strategy_params,
        'best_hyperparameters': best_hps,
        'financial_performance': financial_metrics,
        'model_statistics': statistical_metrics
    }
    try:
        with open(json_filename, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_summary = json.loads(json.dumps(summary, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
            json.dump(serializable_summary, f, indent=4)
        logging.info(f"Saved run summary to {json_filename}")
    except Exception as e:
        logging.error(f"Failed to save JSON summary to {json_filename}: {e}")

def calculate_financial_metrics(portfolio_series: pd.Series, trades_df: pd.DataFrame, initial_cash: float) -> dict:
    """Calculates standard backtest performance metrics."""
    if portfolio_series is None or portfolio_series.empty:
        return {'error': 'Portfolio series is empty'}

    final_val = float(portfolio_series.iloc[-1])
    total_return = (final_val / initial_cash - 1.0) * 100 if initial_cash > 0 else 0.0

    peak = portfolio_series.cummax()
    drawdown = (portfolio_series - peak) / peak
    max_dd = float(drawdown.min() * 100) if not drawdown.empty else 0.0

    daily_returns = portfolio_series.pct_change().fillna(0)
    sharpe = 0.0
    if daily_returns.std() > 1e-9: # Avoid division by zero or near-zero
         sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    total_trades = len(trades_df) if trades_df is not None else 0
    win_count = sum(trades_df['profit'] > 0) if total_trades > 0 else 0
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

    return {
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd,
        'sharpe_ratio': sharpe,
        'total_trades': total_trades,
        'win_rate_pct': win_rate,
        'final_portfolio_value': final_val
    }

def plot_equity_curve(portfolio_series: pd.Series, filename: str, run_id: str):
    """Plots and saves the equity curve."""
    if portfolio_series is None or portfolio_series.empty:
        logging.warning("Cannot plot equity curve: Portfolio series is empty.")
        return
    try:
        plt.figure(figsize=(15, 7))
        plt.plot(portfolio_series.index, portfolio_series.values)
        plt.title(f'Combined Equity Curve (Run: {run_id})')
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Date')
        plt.grid(True)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gcf().autofmt_xdate() # Rotate date labels
        plt.savefig(filename)
        plt.close()
        logging.info(f"Saved equity curve to {filename}")
    except Exception as e:
        logging.error(f"Failed to plot or save equity curve to {filename}: {e}")

