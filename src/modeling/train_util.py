import gc

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoost
from catboost import Pool

import common.com_util as util


def __get_x_y_from_data(
        logger, df, predictors, target):
    """Returns X & Y the DataFrame"""
    if df is not None:
        df_X = df[predictors]
        df_Y = df[target]
    return df_X, df_Y


def __get_x_y_from_training_validation(
        logger, training, validation, predictors, target):
    """Returns X & Y for training & validation data"""
    if training is not None:
        training_X, training_Y = __get_x_y_from_data(logger, training, predictors, target)
    if validation is not None:
        validation_X, validation_Y = __get_x_y_from_data(logger, validation, predictors, target)
    return training_X, training_Y, validation_X, validation_Y


def cat_train_validate_on_holdout(
        logger, training, validation, predictors, target, cat_features,
        params, test_X=None):
    """Train a XGBoost model, validate on holdout data. If `test_X`
       has a valid value, creates a new model with number of best iteration
       found during holdout phase using training as well as validation data.
    """

    logger.info("Training using CatBoost and validating on holdout")
    train_X, train_Y, validation_X, validation_Y = __get_x_y_from_training_validation(
        logger, training, validation, predictors, target)

    logger.info((f"Shape of train_X, train_Y, validation_X, validation_Y: "
                f"{train_X.shape}, {train_Y.shape}, {validation_X.shape}, {validation_Y.shape}"))

    train_pool = Pool(
        data=train_X, label=train_Y, feature_names=predictors,
        cat_features=cat_features)
    valid_pool = Pool(
        data=validation_X, label=validation_Y, feature_names=predictors,
        cat_features=cat_features)

    model = CatBoost(params=params)
    model.fit(X=train_pool, eval_set=[train_pool, valid_pool])

    best_iteration = model.get_best_iteration()

    valid_prediction = model.predict(valid_pool)

    valid_score = np.sqrt(
        metrics.mean_squared_error(validation_Y, valid_prediction))
    logger.info(f'Validation Score {valid_score}')
    logger.info(f'Best Iteration {best_iteration}')

    del train_pool, valid_pool, train_X, train_Y, validation_X, validation_Y
    gc.collect()

    if test_X is not None:
        logger.info("Retraining on the entire data including validation")
        training = pd.concat([training, validation])
        train_X, train_Y = __get_x_y_from_data(logger, training, predictors, target)
        logger.info((f"Shape of train_X, train_Y: "
                    f"{train_X.shape}, {train_Y.shape}"))

        train_pool = Pool(
            data=train_X, label=train_Y, feature_names=predictors,
            cat_features=cat_features)
        test_pool = Pool(
            data=test_X, feature_names=predictors, cat_features=cat_features)

        params.pop("eval_metric")
        params.pop("early_stopping_rounds")
        params.pop("use_best_model")
        params["n_estimators"] = best_iteration

        logger.info(f"Modified parameters for final model training.. {params}")

        model = CatBoost(params=params)
        model.fit(X=train_pool)

        logger.info(f"Predicting on test data: {test_X.shape}")
        prediction = model.predict(test_pool)
        return model, best_iteration, valid_score, prediction
    else:
        return model, valid_score


def xgb_train_validate_on_holdout(
        logger, training, validation, predictors, target,
        params, test_X=None, n_estimators=10000, early_stopping_rounds=100,
        verbose_eval=100):
    """Train a XGBoost model, validate on holdout data. If `test_X`
       has a valid value, creates a new model with number of best iteration
       found during holdout phase using training as well as validation data.

       Args:
            logger: Logger to be used
            training: Training DataFrame
            validation: Validation DataFrame
            predictors: List of names of features
            target: Name of target variable
            params: Parameters for XGBoost
            test_X: Test DataFrame

       Returns:
            bst: XGB Booster object
            valid_score: Best validation score
            best_iteration: Value of best iteration
            prediction: Prediction generated on `test_X`
    """
    logger.info("Training using XGBoost and validating on holdout")
    train_X, train_Y, validation_X, validation_Y = __get_x_y_from_training_validation(
        logger, training, validation, predictors, target)

    logger.info((f"Shape of train_X, train_Y, validation_X, validation_Y: "
                f"{train_X.shape}, {train_Y.shape}, {validation_X.shape}, {validation_Y.shape}"))

    dtrain = xgb.DMatrix(data=train_X, label=train_Y, feature_names=predictors)
    dvalid = xgb.DMatrix(data=validation_X, label=validation_Y, feature_names=predictors)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
    bst = xgb.train(
        dtrain=dtrain, num_boost_round=n_estimators,
        evals=watchlist, early_stopping_rounds=early_stopping_rounds,
        params=params, verbose_eval=verbose_eval)

    valid_prediction = bst.predict(
        xgb.DMatrix(validation_X, feature_names=predictors),
        ntree_limit=bst.best_ntree_limit)

    # Get best iteration
    best_iteration = bst.best_ntree_limit

    valid_score = np.sqrt(
        metrics.mean_squared_error(validation_Y, valid_prediction))
    logger.info(f'Validation Score {valid_score}')
    logger.info(f'Best Iteration {best_iteration}')

    del watchlist, dtrain, dvalid, train_X, train_Y, validation_X, validation_Y
    gc.collect()

    if test_X is not None:
        logger.info("Retraining on the entire data including validation")
        training = pd.concat([training, validation])
        train_X, train_Y = __get_x_y_from_data(logger, training, predictors, target)
        logger.info((f"Shape of train_X, train_Y: "
                    f"{train_X.shape}, {train_Y.shape}"))

        dtrain = xgb.DMatrix(data=train_X, label=train_Y, feature_names=predictors)
        dtest = xgb.DMatrix(data=test_X, feature_names=predictors)

        bst = xgb.train(
            dtrain=dtrain, num_boost_round=best_iteration, params=params)

        logger.info(f"Predicting on test data: {test_X.shape}")
        prediction = bst.predict(dtest, ntree_limit=best_iteration)
        return bst, best_iteration, valid_score, prediction
    else:
        return bst, valid_score


def lgb_train_validate_on_holdout(
        logger, training, validation, predictors, target,
        params, test_X=None, n_estimators=10000, early_stopping_rounds=100):
    """Train a LGB model and validate on holdout data.
    """
    logger.info("Training using LGB and validating on holdout")
    train_X, train_Y, validation_X, validation_Y = __get_x_y_from_training_validation(
        logger, training, validation, predictors, target)

    logger.info((f"Shape of train_X, train_Y, validation_X, validation_Y: "
                f"{train_X.shape}, {train_Y.shape}, {validation_X.shape}, {validation_Y.shape}"))

    dtrain = lgb.Dataset(train_X, label=train_Y)
    dvalid = lgb.Dataset(validation_X, validation_Y)

    bst = lgb.train(
        params, dtrain, valid_sets=[dtrain, dvalid],
        num_boost_round=n_estimators, early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100)

    best_iteration = bst.best_iteration

    valid_prediction = bst.predict(
        validation_X, num_iteration=best_iteration)
    valid_score = np.sqrt(
        metrics.mean_squared_error(validation_Y, valid_prediction))
    logger.info(f'Validation Score {valid_score}')
    logger.info(f'Best Iteration {best_iteration}')

    del dtrain, dvalid, train_X, train_Y, validation_X, validation_Y
    gc.collect()

    if test_X is not None:
        logger.info("Retraining on the entire data including validation")
        training = pd.concat([training, validation])
        train_X, train_Y = __get_x_y_from_data(logger, training, predictors, target)
        logger.info((f"Shape of train_X, train_Y: "
                    f"{train_X.shape}, {train_Y.shape}"))

        dtrain = lgb.Dataset(train_X, label=train_Y)

        bst = lgb.train(
                params, dtrain, num_boost_round=best_iteration,
                verbose_eval=100)

        logger.info(f"Predicting on test data: {test_X.shape}")
        prediction = bst.predict(test_X, num_iteration=best_iteration)

        return bst, best_iteration, valid_score, prediction
    else:
        return bst, valid_score


def make_prediction_classification(logger, run_id, df_train_X, df_train_Y, df_test_X, kf, features=None,
                                   params=None, n_estimators=10000,
                                   early_stopping_rounds=100, model_type='lgb',
                                   is_test=False, seed=42, model=None,
                                   plot_feature_importance=False, cat_features=None):
    """
    Make prediction for classification use case only

    model : Needed only for model_type=='sklearn'
    plot_feature_importance : Only needed for LGBM (in case feature importnace is needed)
    n_estimators : For XGB should be explicitly passed through this method
    early_stopping_rounds : For XGB should be explicitly passed through this method.
                            For LGB can be passed through params as well

    cat_features : Only needed for CatBoost
    params: For SKLearn based models parameters should be passed while creating the Model itself

    """
    yoof = np.zeros(len(df_train_X))
    yhat = np.zeros(len(df_test_X))
    cv_scores = []
    result_dict = {}
    feature_importance = pd.DataFrame()
    best_iterations = []

    # kf = KFold(n_splits=n_splits, random_state=SEED, shuffle=False)
    # kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    fold = 0
    for in_index, oof_index in kf.split(df_train_X[features], df_train_Y):
        # Start a counter describing number of folds
        fold += 1
        # Number of splits defined as a part of KFold/StratifiedKFold
        n_splits = kf.get_n_splits()
        logger.info(f'fold {fold} of {n_splits}')
        X_in, X_oof = df_train_X.iloc[in_index].values, df_train_X.iloc[oof_index].values
        y_in, y_oof = df_train_Y.iloc[in_index].values, df_train_Y.iloc[oof_index].values

        if model_type == 'lgb':
            lgb_train = lgb.Dataset(X_in, y_in)
            lgb_eval = lgb.Dataset(X_oof, y_oof, reference=lgb_train)

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                verbose_eval=50,
                early_stopping_rounds=early_stopping_rounds,
                feature_name=features,
                categorical_feature=cat_features
            )

            del lgb_train, lgb_eval, in_index, X_in, y_in
            gc.collect()

            yoof[oof_index] = model.predict(X_oof, num_iteration=model.best_iteration)
            if is_test is False:
                yhat += model.predict(df_test_X.values, num_iteration=model.best_iteration)

            logger.info(f'Best number of iterations for fold {fold} is: {model.best_iteration}')
            best_iteration = model.best_iteration

        elif model_type == 'xgb':
            xgb_train = xgb.DMatrix(data=X_in, label=y_in, feature_names=features)
            xgb_eval = xgb.DMatrix(data=X_oof, label=y_oof, feature_names=features)

            watchlist = [(xgb_train, 'train'), (xgb_eval, 'valid_data')]
            model = xgb.train(dtrain=xgb_train,
                              num_boost_round=n_estimators,
                              evals=watchlist,
                              early_stopping_rounds=early_stopping_rounds,
                              params=params,
                              verbose_eval=50)

            del xgb_train, xgb_eval, in_index, X_in, y_in
            gc.collect()

            yoof[oof_index] = model.predict(xgb.DMatrix(X_oof, feature_names=features), ntree_limit=model.best_ntree_limit)
            if is_test is False:
                yhat += model.predict(xgb.DMatrix(
                    df_test_X.values, feature_names=features),
                                      ntree_limit=model.best_ntree_limit)

            logger.info(f'Best number of iterations for fold {fold} is: {model.best_ntree_limit}')
            best_iteration = model.best_ntree_limit

        elif model_type == 'cat':
            # feature_names accepts only list
            cat_train = Pool(data=X_in, label=y_in, feature_names=features.tolist(), cat_features=cat_features)
            cat_eval = Pool(data=X_oof, label=y_oof, feature_names=features.tolist(), cat_features=cat_features)
            cat_test = Pool(data=df_test_X, feature_names=features.tolist(), cat_features=cat_features)

            model = CatBoost(params=params)
            model.fit(cat_train, eval_set=cat_eval, use_best_model=True)

            del in_index, X_in, y_in, cat_train
            gc.collect()

            yoof[oof_index] = model.predict(cat_eval)
            if is_test is False:
                # yhat += model.predict(df_test_X.values)
                yhat += model.predict(cat_test)

            del cat_eval, cat_test
            best_iteration = model.best_iteration_
            logger.info(f'Best number of iterations for fold {fold} is: {best_iteration}')

        elif model_type == 'sklearn':
            model = model
            model.fit(X_in, y_in)

            yoof[oof_index] = model.predict_proba(X_oof)[:, 1]
            if is_test is False:
                yhat += model.predict_proba(df_test_X.values)[:, 1]

        # Calculate feature importance per fold
        # TODO : Bolier plate code
        if model_type == 'lgb':
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = features
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            feature_importance.sort_values(by=['importance'], inplace=True)
        elif model_type == 'xgb':
            # Calculate feature importance per fold
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = model.get_score().keys()
            fold_importance["importance"] = model.get_score().values()
            fold_importance["fold"] = fold
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            feature_importance.sort_values(by=['importance'], inplace=True)
        elif model_type == 'cat':
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = model.feature_names_
            fold_importance["importance"] = model.get_feature_importance()
            fold_importance["fold"] = fold
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
            feature_importance.sort_values(by=['importance'], inplace=True)

        cv_oof_score = roc_auc_score(y_oof, yoof[oof_index])
        logger.info(f'CV OOF Score for fold {fold} is {cv_oof_score}')
        cv_scores.append(cv_oof_score)
        best_iterations.append(best_iteration)

        del oof_index, X_oof, y_oof
        gc.collect()

        util.update_tracking(run_id, "metric_fold_{}".format(fold), cv_oof_score, is_integer=False)

    yhat /= n_splits

    oof_score = round(roc_auc_score(df_train_Y, yoof), 5)
    avg_cv_scores = round(sum(cv_scores)/len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)

    logger.info(f'Combined OOF score : {oof_score}')
    logger.info(f'Average of {fold} folds OOF score {avg_cv_scores}')
    logger.info(f'std of {fold} folds OOF score {std_cv_scores}')

    result_dict['yoof'] = yoof
    result_dict['prediction'] = yhat
    result_dict['oof_score'] = oof_score
    result_dict['cv_scores'] = cv_scores
    result_dict['avg_cv_scores'] = avg_cv_scores
    result_dict['std_cv_scores'] = std_cv_scores

    util.update_tracking(run_id, "oof_score", oof_score, is_integer=False)
    util.update_tracking(run_id, "cv_avg_score", avg_cv_scores, is_integer=False)
    util.update_tracking(run_id, "cv_std_score", std_cv_scores, is_integer=False)
    # Best Iteration
    util.update_tracking(run_id, 'avg_best_iteration', np.mean(best_iterations), is_integer=False)
    util.update_tracking(run_id, 'std_best_iteration', np.std(best_iterations), is_integer=False)

    del yoof, yhat
    gc.collect()

    # Plot feature importance
    if (model_type == 'lgb') | (model_type == 'xgb') | (model_type == 'cat'):
        # Not sure why it was necessary. Hence commenting
        # feature_importance["importance"] /= n_splits
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
            by="importance", ascending=False)[:50].index

        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

        result_dict['feature_importance'] = feature_importance
        result_dict['best_features'] = best_features

    logger.info('Training/Prediction completed!')
    return result_dict


def make_sklearn_prediction_classification(logger, run_id, df_train_X,
                                           df_train_Y, df_test_X, kf,
                                           features=None, params=None,
                                           model_type=None, is_test=False,
                                           seed=42, model=None):
    """
    Make prediction for classification use case only

    """
    yoof = np.zeros(len(df_train_X))
    yhat = np.zeros(len(df_test_X))
    cv_scores = []
    result_dict = {}

    fold = 0
    for in_index, oof_index in kf.split(df_train_X[features], df_train_Y):
        # Start a counter describing number of folds
        fold += 1
        # Number of splits defined as a part of KFold/StratifiedKFold
        n_splits = kf.get_n_splits()
        logger.info(f'fold {fold} of {n_splits}')
        X_in, X_oof = df_train_X.iloc[in_index].values, df_train_X.iloc[oof_index].values
        y_in, y_oof = df_train_Y.iloc[in_index].values, df_train_Y.iloc[oof_index].values

        model = model
        model.fit(X_in, y_in)

        yoof[oof_index] = model.predict_proba(X_oof)[:, 1]
        if is_test is False:
            yhat += model.predict_proba(df_test_X.values)[:, 1]

        cv_oof_score = roc_auc_score(y_oof, yoof[oof_index])
        logger.info(f'CV OOF Score for fold {fold} is {cv_oof_score}')
        cv_scores.append(cv_oof_score)

        del oof_index, X_oof, y_oof
        gc.collect()

        util.update_tracking(run_id, "metric_fold_{}".format(fold), cv_oof_score, is_integer=False)

    yhat /= n_splits

    oof_score = round(roc_auc_score(df_train_Y, yoof), 5)
    avg_cv_scores = round(sum(cv_scores)/len(cv_scores), 5)
    std_cv_scores = round(np.array(cv_scores).std(), 5)

    logger.info(f'Combined OOF score : {oof_score}')
    logger.info(f'Average of {fold} folds OOF score {avg_cv_scores}')
    logger.info(f'std of {fold} folds OOF score {std_cv_scores}')

    result_dict['yoof'] = yoof
    result_dict['prediction'] = yhat
    result_dict['oof_score'] = oof_score
    result_dict['cv_scores'] = cv_scores
    result_dict['avg_cv_scores'] = avg_cv_scores
    result_dict['std_cv_scores'] = std_cv_scores

    util.update_tracking(run_id, "oof_score", oof_score, is_integer=False)
    util.update_tracking(run_id, "cv_avg_score", avg_cv_scores, is_integer=False)
    util.update_tracking(run_id, "cv_std_score", std_cv_scores, is_integer=False)

    del yoof, yhat
    gc.collect()

    logger.info('Training/Prediction completed!')
    return result_dict
