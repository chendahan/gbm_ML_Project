
from pandas.api.types import is_string_dtype
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint
from sklearn import metrics
from time import time
import os
from sklearn.preprocessing import LabelEncoder

# TODO : change path according to the location of the datasets files
dirname = 'C:\\Users\\royk\\Downloads\\classification_datasets-20200531T065549Z-001\\classification_datasets'

# model params- used for hyperparameter tuning
model_params = {
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(2, 6),
    'n_estimators': randint(10, 200),
    'subsample': uniform(0.6, 0.4)
}


def multiClassStat(gbm, X_test, y_test, y_pred):
    """
    calculate stats for multiclass classification
    :param gbm: gbm classifier
    :param X_test: test data
    :param y_test: test classification
    :param y_pred: predicted classification using gbm
    :return: accuracy, TPR, FPR, precision, roc_auc, PR_curve
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    y_prob = gbm.predict_proba(X_test)
    try:
        roc_auc = metrics.roc_auc_score(y_test, y_prob, average='macro', multi_class='ovr')
    except ValueError as inst:
        print(inst)
    print('roc_auc ', roc_auc)
    # calculate TPR & FPR using the multilabel_confusion_matrix
    conf_mat = metrics.multilabel_confusion_matrix(y_test, y_pred)
    TPR = 0
    FPR = 0
    for i in conf_mat:
        TN = i[0][0]
        FP = i[0][1]
        FN = i[1][0]
        TP = i[1][1]
        TPR += 0 if (TP + FN) == 0 else TP / (TP + FN)
        FPR += 0 if (FP + TN) == 0 else FP / (FP + TN)
    TPR /= len(conf_mat)
    FPR /= len(conf_mat)
    PR_curve = 0
    # PR_curve - calculate for each class and use average
    for i,cls in zip(range(len(gbm.classes_)),gbm.classes_):
        y_test_ = list(map(int, [num == cls for num in y_test]))
        precision_, recall_, thresholds = metrics.precision_recall_curve(y_test_, y_prob[:, i])
        PR_curve += metrics.auc(recall_, precision_)
    PR_curve /= len(y_prob[0])
    return accuracy, TPR, FPR, precision, roc_auc, PR_curve


def binaryStat(gbm, X_test, y_test, y_pred):
    """
    calculate stats for binary classification
    :param gbm: gbm classifier
    :param X_test: test data
    :param y_test: test classification
    :param y_pred: predicted classification using auggbmboost_m
    :return: accuracy, TPR, FPR, precision, roc_auc, PR_curve
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    y_prob = gbm.predict_proba(X_test)
    try:
        roc_auc = metrics.roc_auc_score(y_test, y_prob[:, 1])
    except ValueError as inst:
        print(inst)
        roc_auc = None
    print('roc_auc:', roc_auc)
    # calculate TPR & FPR using the confusion_matrix
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    TPR = 0 if (TP + FN) == 0 else TP / (TP + FN)
    FPR = 0 if (FP + TN) == 0 else FP / (FP + TN)
    _precision, _recall, thresholds = metrics.precision_recall_curve(y_test, y_prob[:,1], )
    PR_curve = metrics.auc(_recall, _precision)

    return accuracy, TPR, FPR, precision, roc_auc, PR_curve

def sanity_check(data, folds):
    """
    check if num samples from class is less than num of folds
    :param folds: k, num of folds in k-fold
    :param data: DataFrame, last column is target
    :return: dropped- dropped class , drop_file - need to drop file, cls_count
    """
    data_count_target = data[data.columns[-1]].value_counts()
    # remove classes with less than 10 lines
    drop_file = False
    cls_count = 0
    dropped = False
    for cls, cnt in data_count_target.iteritems():
        cls_count += 1
        # if num samples from class is less than num of folds, we drop this class
        if cnt < folds:
            data.drop(data[data[data.columns[-1]] == cls].index, inplace=True)
            cls_count -= 1
            dropped = True
    if cls_count < 2:
        drop_file = True
    return dropped, drop_file, cls_count


def pre_process(cls_count, dropped, data):
    """
    :param cls_count: cls_count - class count as calculated in sanity_check
    :param dropped: dropped from sanity_check
    :param data: DataFrame, last column is target
    """
    # convert to 0 1 labels
    if cls_count == 2 and dropped:
        max_val = data[data.columns[-1]].value_counts().index[0]
        data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: 0 if x == max_val else 1)
    # strings - convert using LabelEncoder
    for i in data.columns:
        if is_string_dtype(data[i]):
            enc = LabelEncoder()
            data[i] = enc.fit_transform(data[i].astype(str))
    # fill nan values
    data.fillna(0, inplace=True)



df_redults = pd.DataFrame(columns=['dataset_name', 'algorithm_name', 'cross_validation', 'hyper_params',
                          'accuracy', 'TPR', 'FPR', 'precision', 'roc_auc', 'PR_curve', 'training_time',
                          'inference_time'])
res_file_name = 'results.csv'
df_redults.to_csv(res_file_name)
num_folds = 10 # split data into 10-fold
for filename in os.listdir(dirname):
    print(filename)
    data = pd.read_csv(dirname+'\\'+filename)
    dropped, drop_file, cls_count = sanity_check(data, num_folds)
    if drop_file:
        print("dropping file: ", filename)
        continue  # continue to next file
    pre_process(cls_count, dropped, data)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    skf = StratifiedKFold(n_splits=num_folds)
    fold=0
    for train_index, test_index in skf.split(X, y):
        fold+=1
        print('fold:', fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # create classifier model
        gbm = GradientBoostingClassifier()
        # hyperparameter tuning using RandomizedSearchCV
        rs_model = RandomizedSearchCV(gbm, model_params, n_iter=50, random_state=0, cv=StratifiedKFold(n_splits=3)).fit(X_train, y_train)
        gbm.set_params(**rs_model.best_params_)
        # fit model again to get training time
        start = time()
        gbm.fit(X_train, y_train)
        training_time = time() - start
        # get inference time
        start = time()
        y_pred = gbm.predict(X_test)
        inference_time = (time() - start) / len(X_test) # single inference time
        inference_time *= 1000 # 1000 lines
        if len(y_test.unique()) == 2:
            accuracy, TPR, FPR, precision, roc_auc, PR_curve = binaryStat(gbm, X_test, y_test, y_pred)
        else:
            accuracy, TPR, FPR, precision, roc_auc, PR_curve = multiClassStat(gbm, X_test, y_test, y_pred)
        df_redults = df_redults.append({'dataset_name': filename , 'algorithm_name':'GradientBoosting',
                                        'cross_validation': fold, 'hyper_params': rs_model.best_params_,
                                        'accuracy': accuracy, 'TPR': TPR, 'FPR': FPR, 'precision': precision,
                                        'roc_auc':roc_auc, 'PR_curve':PR_curve, 'training_time':training_time,
                                        'inference_time':inference_time}, ignore_index=True)
    # write df_results to file
    df_redults.to_csv(res_file_name, mode='a', header=False)