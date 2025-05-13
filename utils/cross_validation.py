import sklearn
import numpy as np
from copy import deepcopy

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedGroupKFold

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import Pipeline

# from sklearn.utils.parallel import Parallel, delayed


def _run_KFoldCV(features, labels, groups,
                 clf_name, clf_hparam_grid, 
                 cv_folds, 
                 n_jobs,
                 random_state
                ):
    
    '''
    CAUTION: need to input full dataset!
    '''    
    
    all_fitted_gs = []
    all_outer_fold_subject_groups = []
    all_outer_fold_true_y = []
    all_best_fit_pred_y_probs = []

    '''
    TODO: FIXME: parallelize outer CV loop! (for heldout set variability)
    '''
    # https://stackoverflow.com/questions/55174340/how-to-use-joblib-with-scikitlearn-to-crossvalidate-in-parallel
    # out = Parallel(n_jobs=2, verbose=100, pre_dispatch='1.5*n_jobs')(delayed(train)(train_index, test_index) for train_index, test_index in skf.split(X, Y))

    # outer_cv = StratifiedKFold(n_splits=cv_folds['outer'], shuffle=True, random_state=random_state)
    subject_groups, recording_groups = groups
    outer_cv = StratifiedGroupKFold(n_splits=cv_folds['outer'], shuffle=True, random_state=random_state)
    for fold_idx, (train_val_idx, heldout_test_idx) in enumerate(outer_cv.split(features, labels, subject_groups)):
        
        print(f"Fold {fold_idx}:")
        
        # print(f"  Train: index={train_index}")
        # print(f"         subject_group={subject_groups[train_val_idx]}")
        # print(f"  Test:  index={heldout_test_idx}")
        # print(f"         subject_group={subject_groups[heldout_test_idx]}")

        '''
        define current outer CV fold data: features, labels, groups
        '''
        train_and_val_X = features[train_val_idx]
        train_and_val_y = labels[train_val_idx]
        
        heldout_test_X = features[heldout_test_idx]
        heldout_test_y = labels[heldout_test_idx]

        train_and_val_subject_groups = subject_groups[train_val_idx]
        train_and_val_recording_groups = recording_groups[train_val_idx]

        heldout_test_subject_groups = subject_groups[heldout_test_idx]
        heldout_test_recording_groups = recording_groups[heldout_test_idx]
        
        '''
        define preprocessing + fixed aspects/properties of each model type (ones that are NOT to be tuned)
        '''
        
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), unit_variance=False)
        # scaler = StandardScaler(with_mean=True, with_std=True)
        
        if clf_name == 'GaussianNB':
            clf = GaussianNB(priors=None, var_smoothing=1e-09)
        elif clf_name == 'SVM':
            clf = SVC(class_weight='balanced', probability=True)
        elif clf_name == 'LogReg':
            clf = LogisticRegression(solver='saga', penalty='elasticnet', class_weight='balanced', dual=False, n_jobs=n_jobs["model_fit"])
        elif clf_name == 'kNN':
            clf = KNeighborsClassifier(algorithm='auto', metric='minkowski', metric_params=None, n_jobs=n_jobs["model_fit"])
        else:
            raise ValueError("unknown clf type!")
       
        '''
        run kfold gridsearch hyperparam sweep using train_and_val of current outer cv fold
        NOTE: each hparam will be applied to inner k folds of the train+val data! 
        '''
        pipe = Pipeline([('scaler', scaler), (clf_name, clf)])

        # need to tune something for this estimator!
        if clf_hparam_grid:
            # inner_cv = StratifiedKFold(n_splits=cv_folds['inner'], shuffle=True, random_state=random_state)
            inner_cv = StratifiedGroupKFold(n_splits=cv_folds['inner'], shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(estimator=pipe, param_grid=clf_hparam_grid, 
                                       scoring='roc_auc', n_jobs=n_jobs["inner"], 
                                       refit=True, cv=inner_cv, 
                                       verbose=0, pre_dispatch='2*n_jobs', 
                                       return_train_score=False)
            fitted_gs = grid_search.fit(train_and_val_X, train_and_val_y, groups=train_and_val_subject_groups)
            print("best hparams found using inner CV loop:", fitted_gs.best_params_)
            best_model_pred_y_probs = fitted_gs.predict_proba(heldout_test_X)[:,1]
            all_fitted_gs.append(deepcopy(fitted_gs))
            all_outer_fold_subject_groups.append(heldout_test_subject_groups)
            all_outer_fold_true_y.append(heldout_test_y)
            all_best_fit_pred_y_probs.append(best_model_pred_y_probs)
            
        # nothing to tune for this estimator! (example GaussianNB)
        else:
            fit_model = pipe.fit(train_and_val_X, train_and_val_y)
            all_fitted_gs.append(deepcopy(fit_model))
            all_outer_fold_subject_groups.append(heldout_test_subject_groups)
            all_outer_fold_true_y.append(heldout_test_y)
            all_best_fit_pred_y_probs.append(fit_model.predict_proba(heldout_test_X)[:,1])

    nested_cv_results = [all_fitted_gs, all_outer_fold_subject_groups, all_outer_fold_true_y, all_best_fit_pred_y_probs]
    return nested_cv_results
    

def _run_repeated_LOOCV(features, labels, clf_name, clf_hparams, adjust_sampling_weight=False, n_random_repeats=1):

    '''
    CAUTION: Assumes each row of X and y refers to a unique subject (LOSO = leave one subject out)
    CAUTION: Need to take care of this before calling the function!
    '''
    
    repeats_results = {}
    
    for repeat_idx in range(n_random_repeats):

        cv = LeaveOneOut()
        
        all_clfs = []
        all_true_y = []
        all_pred_y_probs = []
        all_train_scalers = []
        
        # CAUTION: LOOCV loop is expensive depending on number of samples!!
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(features, labels)):
            
            # print(f"Fold {fold_idx} --> Test: index={test_idx}")
                
            '''
            TODO: FIXME: hyperparam sweep for each model!!
            '''
            if clf_name == 'GaussianNB':
                clf = GaussianNB(priors=None, var_smoothing=1e-09)
            elif clf_name == 'SVM':
                clf = SVC(kernel=clf_hparams['kernel'], class_weight='balanced', 
                          C=clf_hparams['C'], probability=True, 
                         gamma=clf_hparams['gamma'], degree=clf_hparams['degree'])
            elif clf_name == 'LogReg':
                clf = LogisticRegression(solver='saga', penalty='elasticnet', class_weight='balanced', 
                                         l1_ratio=clf_hparams['l1_ratio'], dual=False)                
            else:
                raise ValueError("TODO: unknown clf type!")
                
            sample_weight = None
            
            if adjust_sampling_weight:
                sampling_dict = {
                    0: 1./np.sum(labels[train_idx]==0), 
                    1: 1./np.sum(labels[train_idx]==1)
                }
                sample_weight = sklearn.utils.class_weight.compute_sample_weight(sampling_dict, labels[train_idx])
    
            '''
            apply scaling to train, use train statistics when scaling test set
            '''
            _train_x = features[train_idx]
            _train_y = labels[train_idx]
            _test_x = features[test_idx]
            _test_y = labels[test_idx]
            
            train_scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), unit_variance=False).fit(_train_x)
            # train_scaler = StandardScaler(with_mean=True, with_std=True).fit(_train_x)
            
            _train_x = train_scaler.transform(_train_x)
            _test_x = train_scaler.transform(_test_x)
    
    
            '''
            model fitting for single split inside LOOCV
            '''
            fit_model = clf.fit(_train_x, _train_y, sample_weight=sample_weight)

            all_clfs.append(deepcopy(fit_model))
            all_true_y.append(_test_y)
            all_pred_y_probs.append(fit_model.predict_proba(_test_x)[:,1])
            all_train_scalers.append(train_scaler)
        

        all_true_y = np.concatenate(all_true_y)
        all_pred_y_probs = np.concatenate(all_pred_y_probs)

        _results = [all_true_y, all_pred_y_probs, all_clfs, all_train_scalers]
        repeats_results[repeat_idx] = _results
    
    return repeats_results



def run_CV(features, labels, groups, cv_type, cv_params, clf_name):
    
    if cv_type == 'RepeatedLOOCV_NoHparamGridSearch':
        '''
        Leave One Subject Out without any Hparam Tuning - need to supply "best" hparams!
        Best hparams need to be determined outside of this function!
        '''        
        repeats_results = _run_repeated_LOOCV(features, labels, groups,
                                              clf_name, clf_hparams=cv_params['clf_hparams'], 
                                              adjust_sampling_weight=True, 
                                              n_random_repeats=cv_params['n_random_repeats']
                                             )
        return repeats_results

    elif cv_type == 'StratifiedKFold_NestedHparamGridSearch':
        '''
        Stratified KFold (outer cv) for train+val and heldout splits w/ nested Hparam Tuning within each train+val Fold (inner cv)
        '''
        nested_cv_results = _run_KFoldCV(features, labels, groups,
                                     clf_name, clf_hparam_grid=cv_params['hparam_grid'], 
                                     cv_folds=cv_params["cv_folds"],
                                     n_jobs=cv_params["n_jobs"],
                                     random_state=cv_params["random_state"]
                                    )
        return nested_cv_results

    else: 
        raise ValueError("TODO: unknown cv type!")


