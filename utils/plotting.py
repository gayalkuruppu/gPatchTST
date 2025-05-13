import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['font.size'] = 14


def plot_ROC_curves(all_models_full_CV_results, plot_title, metric_type, suptitle):

    fig, ax = plt.subplots(1,1)
    
    # NOTE: this can be same task but multiple classifiers or same classifier but multiple tasks!
    for model_type in all_models_full_CV_results.keys():
    
        all_fitted_gs, all_outer_fold_subject_groups, all_outer_fold_true_y, all_best_fit_pred_y_probs = all_models_full_CV_results[model_type]['full_cv_results']
        
        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        num_outer_folds = len(all_fitted_gs)
        for fold_idx in range(num_outer_folds):
            
            # all_true_y, all_pred_y_probs = all_outer_fold_true_y[fold_idx], all_best_fit_pred_y_probs[fold_idx]
            # fpr, tpr, thresholds = roc_curve(all_true_y, all_pred_y_probs)
            
            fpr = all_models_full_CV_results[model_type]['model_metrics_across_folds'][metric_type]['fprs'][fold_idx]
            tpr = all_models_full_CV_results[model_type]['model_metrics_across_folds'][metric_type]['tprs'][fold_idx]
            thresholds = all_models_full_CV_results[model_type]['model_metrics_across_folds'][metric_type]['thresholds'][fold_idx]
            
            # ax.plot(fpr, tpr, lw=1, alpha=0.4,
            #         color="grey",
            #         # label=f"ROC repeat {repeat_idx}"
            #        )
            
            # interp for uniformity of values across repeats/models
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            all_tprs.append(interp_tpr)
                
        all_tprs = np.array(all_tprs)
        mean_tpr = np.mean(all_tprs, axis=0)
        mean_auroc = np.mean(all_models_full_CV_results[model_type]['model_metrics_across_folds'][metric_type]['aurocs'])
    
        std_tpr = np.std(all_tprs, axis=0)
        std_auroc = np.std(all_models_full_CV_results[model_type]['model_metrics_across_folds'][metric_type]['aurocs'])
        
        # mean +- std ROC curve
        ax.plot(mean_fpr, mean_tpr, lw=2, alpha=0.8, label=f"{model_type} (AUC = {round(mean_auroc, 2)} $\pm$ {round(std_auroc, 2)})")
    
        # filling in std regions
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            alpha=0.2,
            # label=r"$\pm$ 1 std. dev.",
        )
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(plot_title, fontsize=16)
    ax.legend(loc="lower right")
    fig.suptitle(suptitle, fontsize=18)
    plt.show()

    return fig, ax