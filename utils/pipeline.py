from utils.cross_validation import run_CV
from utils.metrics import get_metrics, get_regression_metrics
from utils.plotting import plot_ROC_curves
import numpy as np

def multiclass_clf_pipeline_runner(
    X, y, groups, 
    model_type,
    cv_type,
    cv_params,
    ):

    '''
    INFO
    '''
    _unique, _counts = np.unique(y, return_counts=True)
    print("** ALL DATA USED FOR OUTER CV:", X.shape, _unique, _counts)

    '''
    run CV
    '''
    full_cv_results = run_CV(
        features=X,
        labels=y,
        groups=groups,
        cv_type=cv_type,
        cv_params=cv_params,
        clf_name=model_type,
    )

    '''
    compute classification metrics
    '''
    _, all_outer_fold_subject_groups, all_outer_fold_true_y, all_best_fit_pred_y_probs = full_cv_results
    model_metrics_across_folds = get_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, 
                                             all_outer_fold_subject_groups, cv_params, model_type)
    
    all_results = {
        model_type: {
            'cv_type': cv_type,
            'cv_params': cv_params,
            'full_cv_results': full_cv_results,
            'model_metrics_across_folds': model_metrics_across_folds, 
        }
    }

    return all_results

def regression_pipeline_runner(
    X, y, groups, 
    model_type,
    cv_type,
    cv_params,
    ):

    '''
    run CV
    '''
    full_cv_results = run_CV(
        features=X,
        labels=y,
        groups=groups,
        cv_type=cv_type,
        cv_params=cv_params,
        clf_name=model_type,
    )

    '''
    compute regression metrics
    '''
    _, all_outer_fold_subject_groups, all_outer_fold_true_y, all_best_fit_pred_y_probs = full_cv_results
    model_metrics_across_folds = get_regression_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, 
                                             all_outer_fold_subject_groups, cv_params, model_type)
    
    all_results = {
        model_type: {
            'cv_type': cv_type,
            'cv_params': cv_params,
            'full_cv_results': full_cv_results,
            'model_metrics_across_folds': model_metrics_across_folds, 
        }
    }

    return all_results



def clf_pipeline_runner(
    X, y, groups, 
    model_type,
    cv_type,
    cv_params,
    return_plot=False,
    suptitle="",
    ):

    '''
    INFO
    '''
    _unique, _counts = np.unique(y, return_counts=True)
    print("** ALL DATA USED FOR OUTER CV:", X.shape, _unique, _counts)

    '''
    run CV
    '''
    full_cv_results = run_CV(
        features=X,
        labels=y,
        groups=groups,
        cv_type=cv_type,
        cv_params=cv_params,
        clf_name=model_type,
    )

    '''
    compute classification metrics
    '''
    _, all_outer_fold_subject_groups, all_outer_fold_true_y, all_best_fit_pred_y_probs = full_cv_results
    model_metrics_across_folds = get_metrics(all_outer_fold_true_y, all_best_fit_pred_y_probs, 
                                             all_outer_fold_subject_groups, cv_params, model_type)
    
    all_results = {
        model_type: {
            'cv_type': cv_type,
            'cv_params': cv_params,
            'full_cv_results': full_cv_results,
            'model_metrics_across_folds': model_metrics_across_folds, 
        }
    }

    if return_plot == False:
        return all_results

    else:
        '''
        make and return ROC plot
        '''
        fig, ax = plot_ROC_curves(
            all_results, 
            # plot_title=f"SUBJECT-LEVEL\nStrat{cv_params['cv_folds']['outer']}FoldCV + NestedHparamGridSearch \
            # (Strat{cv_params['cv_folds']['inner']}FoldCV)\n??(N=??) vs. ??(N=??)", 
            plot_title=f"N: {_counts}",
            metric_type="subject_level",
            suptitle=suptitle,
        )
        return fig, ax