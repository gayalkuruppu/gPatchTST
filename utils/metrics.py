from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

def _compute_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, all_outer_fold_subject_groups, metric_type, num_outer_folds):
    
    model_metrics_across_folds = {
        "fprs": [],
        "tprs": [],
        "thresholds": [],        
        "aurocs": [],        
        "accs": [],        
        "bal_accs": [],        
    }
    
    for fold_idx in range(num_outer_folds):
        
        all_true_y = all_outer_fold_true_y[fold_idx]
        all_pred_y_probs = all_best_fit_pred_y_probs[fold_idx]

        if metric_type == "subject_level":
            '''
            epoch-level predicted probabilities are averaged over for each subject in the fold
            '''
            all_subject_groups = all_outer_fold_subject_groups[fold_idx]
    
            # average all pred_y_probs belonging to same subject group
            df = pd.DataFrame({
                "true_y": all_true_y,
                "pred_y_prob": all_pred_y_probs,
                "subject_group": all_subject_groups,
            })
    
            updated_true_y = []
            updated_pred_y_probs = []
            for subject_id in np.unique(all_subject_groups):
                subject_df = df[df["subject_group"] == subject_id]
                assert len(list(subject_df["true_y"].unique())) == 1
                updated_true_y.append(list(subject_df["true_y"].unique())[0])
                updated_pred_y_probs.append(subject_df["pred_y_prob"].mean())
    
            all_true_y = np.array(updated_true_y)
            all_pred_y_probs = np.array(updated_pred_y_probs)

        auroc = roc_auc_score(all_true_y, all_pred_y_probs)
        # obtain optimal threshold from predicted y probs: Cut-off is determied using Youden's index defined as sensitivity + specificity - 1.
        fpr, tpr, thresholds = roc_curve(all_true_y, all_pred_y_probs)
        idx = np.argmax(tpr - fpr)
        thres = thresholds[idx]
        # calculate acc/balacc metrics after optimal threshold is applied
        acc = accuracy_score(all_true_y, (all_pred_y_probs>=thres).astype(int))
        bal_acc = balanced_accuracy_score(all_true_y, (all_pred_y_probs>=thres).astype(int))

        model_metrics_across_folds["fprs"].append(fpr)
        model_metrics_across_folds["tprs"].append(tpr)
        model_metrics_across_folds["thresholds"].append(thres)
        model_metrics_across_folds["aurocs"].append(auroc)
        model_metrics_across_folds["accs"].append(acc)
        model_metrics_across_folds["bal_accs"].append(bal_acc)
          
    return model_metrics_across_folds


def _compute_regression_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, all_outer_fold_subject_groups, metric_type, num_outer_folds):

    model_metrics_across_folds = {
        "mean_squared_errors": [],
        "mean_absolute_errors": [],
        "r2_scores": [],        
    }

    for fold_idx in range(num_outer_folds):
        
        # CAUTION: y_probs == y_pred for LinRegression!
        all_true_y = all_outer_fold_true_y[fold_idx]
        all_pred_y = all_best_fit_pred_y_probs[fold_idx]

        if metric_type == "sequence_level":
            mse = mean_squared_error(all_true_y, all_pred_y)
            mae = mean_absolute_error(all_true_y, all_pred_y)
            r2 = r2_score(all_true_y, all_pred_y)

            model_metrics_across_folds["mean_squared_errors"].append(mse)
            model_metrics_across_folds["mean_absolute_errors"].append(mae)
            model_metrics_across_folds["r2_scores"].append(r2)

        elif metric_type == "subject_level":
            raise ValueError("TODO: not implemented yet!")

        else:
            raise ValueError("FIXME: unknown metric type!")

    return model_metrics_across_folds


def get_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, all_outer_fold_subject_groups, cv_params, model_type):
    '''
    compute and print heldout metrics after full CV run
    '''
    model_metrics_across_folds = {}
    
    model_metrics_across_folds['epoch_level'] = _compute_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs,
                                                                 all_outer_fold_subject_groups, 
                                                                 metric_type="epoch_level",
                                                                 num_outer_folds=cv_params['cv_folds']['outer'], 
                                                                )
    
    model_metrics_across_folds['subject_level'] = _compute_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs,
                                                                 all_outer_fold_subject_groups, 
                                                                 metric_type="subject_level",
                                                                 num_outer_folds=cv_params['cv_folds']['outer'], 
                                                                )
    
    for metrics_type in model_metrics_across_folds.keys():
        print(f"\n\n --------------- {metrics_type.upper()} ---------------")
        metrics = model_metrics_across_folds[metrics_type]
        print(f"\n** {model_type}: Mean +- std across {cv_params['cv_folds']['outer']} OUTER CV folds **")
        print(f"AUROC: {np.mean(metrics['aurocs'])} +- {np.std(metrics['aurocs'])}")
        print(f"Accuracy: {np.mean(metrics['accs'])} +- {np.std(metrics['accs'])}")
        print(f"Bal. Accuracy: {np.mean(metrics['bal_accs'])} +- {np.std(metrics['bal_accs'])}")
        print("****\n\n")

    return model_metrics_across_folds


def get_regression_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, all_outer_fold_subject_groups, cv_params, model_type):
    '''
    compute and print heldout metrics after full CV run
    '''
    model_metrics_across_folds = {}

    # sequence = single-channel 10s epoch
    model_metrics_across_folds['sequence_level'] = _compute_regression_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs,
                                                                 all_outer_fold_subject_groups, 
                                                                 metric_type="sequence_level",
                                                                 num_outer_folds=cv_params['cv_folds']['outer'], 
                                                                )

    # model_metrics_across_folds['subject_level'] = _compute_regression_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs,
    #                                                              all_outer_fold_subject_groups, 
    #                                                              metric_type="subject_level",
    #                                                              num_outer_folds=cv_params['cv_folds']['outer'], 
    #                                                             )

    for metrics_type in model_metrics_across_folds.keys():
        print(f"\n\n --------------- {metrics_type.upper()} ---------------")
        metrics = model_metrics_across_folds[metrics_type]
        print(f"\n** {model_type}: Mean +- std across {cv_params['cv_folds']['outer']} OUTER CV folds **")
        print(f"MSE: {np.mean(metrics['mean_squared_errors'])} +- {np.std(metrics['mean_squared_errors'])}")
        print(f"MAE: {np.mean(metrics['mean_absolute_errors'])} +- {np.std(metrics['mean_absolute_errors'])}")
        print(f"R^2: {np.mean(metrics['r2_scores'])} +- {np.std(metrics['r2_scores'])}")
        print("****\n\n")

    return model_metrics_across_folds