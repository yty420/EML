# 导入操作系统相关库
import os

# 导入数据处理相关库
import pandas as pd
import numpy as np
import csv
import json
from collections import defaultdict
from itertools import chain
import math

# 导入scikit-learn模型选择相关库
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, KFold, learning_curve

# 导入scikit-learn预处理相关库
from sklearn.preprocessing import label_binarize, StandardScaler

# 导入scikit-learn多分类相关库
from sklearn.multiclass import OneVsRestClassifier

# 导入scikit-learn特征选择相关库
from sklearn.feature_selection import RFE, SequentialFeatureSelector, SelectKBest, chi2, mutual_info_classif, VarianceThreshold

# 导入scikit-learn评估指标相关库
from sklearn.metrics import (classification_report, confusion_matrix,
                           accuracy_score, roc_curve, auc, roc_auc_score)

# 导入scikit-learn集成学习相关库
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

# 导入scikit-learn线性模型相关库
from sklearn.linear_model import Ridge, Lasso, LogisticRegression

# 导入scikit-learn支持向量机相关库
from sklearn.svm import SVC

# 导入imbalanced-learn过采样相关库
from imblearn.over_sampling import SMOTE, ADASYN

# 导入xgboost相关库
import xgboost as xgb

# 导入可视化相关库
import matplotlib.pyplot as plt
import seaborn as sns

# 导入joblib和pickle用于加载与保存模型等操作
import joblib
import pickle

# 设置全局随机种子
np.random.seed(11)
os.environ['PYTHONHASHSEED'] = str(11)

# 可视化配置
plt.rcParams['figure.figsize'] = [10, 6]
sns.set_style("whitegrid")





# 应用卡方检验选择特征
def select_features_chi2(X, y, num_features):
    selector = SelectKBest(chi2, k=num_features)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_gene_names = X.columns[selected_indices].tolist()
    return selected_gene_names
# 应用互信息量选择特征
def select_features_mi(X, y, num_features):
    selector = SelectKBest(mutual_info_classif, k=num_features)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    selected_gene_names = X.columns[selected_indices].tolist()
    return selected_gene_names
# 应用方差排序选择特征
def select_features_by_variance(X, num_features):
    variances = np.var(X, axis=0)
    sorted_indices = np.argsort(variances)[::-1]
    selected_indices = sorted_indices[:num_features]
    selected_gene_names = X.columns[selected_indices].tolist()
    return selected_gene_names
# 应用方差阈值
def select_features_variance_bythreshold(X, threshold):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X)
    selected_indices = selector.get_support(indices=True)
    selected_gene_names = X.columns[selected_indices].tolist()
    return selected_gene_names

from sklearn.feature_selection import SequentialFeatureSelector

#前向选择（Forward Selection）
#前向选择是一种迭代方法，在每一步中添加对模型性能改善最大的特征，直到没有剩余特征可以显著改善模型性能
#后向消除（Backward Elimination）
#后向消除与前向选择相反，从所有特征开始，迭代地移除对模型性能改善最小的特征。
def select_features_sfs(X, y, model, n_features_to_select='auto', direction='forward', cv=5):
    sfs = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction, cv=cv,n_jobs=-1)
    sfs.fit(X, y)
    selected_features = sfs.get_support()
    selected_gene_names = X.columns[selected_features].tolist()
    return selected_gene_names


from sklearn.feature_selection import RFE

def select_features_rfe(X, y, model, n_features_to_select=None, step=1, verbose=1):
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select, step=step, verbose=verbose)
    rfe.fit(X, y)
    selected_features = rfe.get_support()
    selected_gene_names = X.columns[selected_features].tolist()
    return selected_gene_names

def select_features_cv_importance_single_model(X, y, model, cv=5, top_n=5, freq=3):
    feature_freq = pd.Series(0, index=X.columns)

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        if hasattr(cloned_model, 'feature_importances_'):
            importances = cloned_model.feature_importances_
        elif hasattr(cloned_model, 'coef_'):
            importances = np.abs(cloned_model.coef_[0])
        else:
            continue  # Skip model if it doesn't have feature_importances_ or coef_

        # Identify top n features for the model
        top_features = pd.Series(importances, index=X.columns).nlargest(top_n).index
        feature_freq[top_features] += 1

    # Select features appearing in top n with frequency >= freq
    selected_features = feature_freq[feature_freq >= freq].index.tolist()

    # Create frequency table
    freq_table = feature_freq.value_counts().sort_index(ascending=False).reset_index()
    freq_table.columns = ['Frequency', 'Number of Features']
    print(freq_table)
    return selected_features

def select_features_cv_importance_multi_model(X, y, model, cv=5, freq_list=[3], top_n_list=[5]):
    feature_rankings = {i: pd.Series(0, index=X.columns) for i in range(cv)}  # 初始化特征排名字典

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_index = 0
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        if hasattr(cloned_model, 'feature_importances_'):
            importances = cloned_model.feature_importances_
        elif hasattr(cloned_model, 'coef_'):
            importances = np.abs(cloned_model.coef_[0])
        else:
            continue  # 如果模型没有 feature_importances_ 或 coef_，则跳过

        # 存储每次模型的特征重要性排名
        feature_rankings[fold_index] = pd.Series(importances, index=X.columns)
        fold_index += 1

    # 用于保存满足不同条件的特征列表
    results = {}

    for top_n in top_n_list:
        for freq in freq_list:
            feature_freq = pd.Series(0, index=X.columns)
            for ranking in feature_rankings.values():
                top_features = ranking.nlargest(top_n).index
                feature_freq[top_features] += 1

            selected_features = feature_freq[feature_freq >= freq].index.tolist()
            results[(top_n, freq)] = selected_features

    return results


def save_results_to_csv(results, filename):
    # 转换字典为适合保存的DataFrame
    data = []
    for key, features in results.items():
        data.append({
            'Top N': key[0],
            'Frequency': key[1],
            'Features': ','.join(features)
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)



def select_features_cv_importance_multi_model2(X, y, model, cv=5, freq_list=[3], top_n_list=[5]):
    feature_rankings = {i: pd.Series(0, index=X.columns) for i in range(cv)}  # 初始化特征排名字典

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    fold_index = 0
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        if hasattr(cloned_model, 'feature_importances_'):
            importances = cloned_model.feature_importances_
        elif hasattr(cloned_model, 'coef_'):
            importances = np.abs(cloned_model.coef_[0])
        else:
            continue  # 如果模型没有 feature_importances_ 或 coef_，则跳过

        # 存储每次模型的特征重要性排名
        feature_rankings[fold_index] = pd.Series(importances, index=X.columns)
        fold_index += 1

    # 用于保存满足不同条件的特征列表
    results = {}

    for top_n in top_n_list:
        for freq in freq_list:
            feature_freq = pd.Series(0, index=X.columns)
            for ranking in feature_rankings.values():
                # 获取当前排名中的前 top_n 个特征及其重要性值
                top_features_with_values = ranking.nlargest(top_n)

                # 过滤掉特征值为 0 的特征
                top_features_non_zero = top_features_with_values[top_features_with_values > 0].index

                # 更新特征频率计数器
                feature_freq[top_features_non_zero] += 1

            # 筛选出符合条件的特征
            selected_features = feature_freq[feature_freq >= freq].index.tolist()

            # 存储结果
            results[(top_n, freq)] = selected_features

    return results



def load_results_from_csv(filename):
    df = pd.read_csv(filename)
    results = {}
    for index, row in df.iterrows():
        key = (row['Top N'], row['Frequency'])
        features = row['Features'].split(',')
        results[key] = features
    return results



def evaluate_feature_lists_with_counts(X_train, X_test, y_train, y_test, feature_lists, model,lp=False,scale=False):
    results = []

    for name, features in feature_lists.items():
        print(f'Evaluating feature list: {name}')
        X_train_sub = X_train[features]
        X_test_sub = X_test[features]
        if scale:
            scaler = StandardScaler()
            X_train_sub = scaler.fit_transform(X_train_sub)
            X_test_sub = scaler.transform(X_test_sub)

        cloned_model = clone(model)
        cloned_model.fit(X_train_sub, y_train)

        #add a if to check whether we need to plot learning curve
        if lp:
            train_sizes, train_scores, test_scores = learning_curve(model, X_train_sub, y_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10),
                                                            cv=5, scoring='accuracy', n_jobs=-1)

            # 计算训练集和验证集的平均和标准差
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            # 绘制学习曲线
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='r', alpha=0.1)
            plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='g', alpha=0.1)

            plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
            plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')

            plt.title('Learning Curve')
            plt.xlabel('Training Size')
            plt.ylabel('Accuracy')
            plt.legend(loc='best')
            plt.show()


        # Training set predictions and metrics
        y_pred_train = cloned_model.predict_proba(X_train_sub)[:, 1]
        y_pred_train_class = cloned_model.predict(X_train_sub)
        auc_train = roc_auc_score(y_train, y_pred_train)
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_pred_train_class).ravel()
        sensitivity_train = tp_train / (tp_train + fn_train)
        specificity_train = tn_train / (tn_train + fp_train)

        # Test set predictions and metrics
        y_pred_test = cloned_model.predict_proba(X_test_sub)[:, 1]
        y_pred_test_class = cloned_model.predict(X_test_sub)
        auc_test = roc_auc_score(y_test, y_pred_test)
        tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_pred_test_class).ravel()
        sensitivity_test = tp_test / (tp_test + fn_test)
        specificity_test = tn_test / (tn_test + fp_test)

        results.append({
            'Feature List': name,
            'Train AUC': auc_train,
            'Train Sensitivity': sensitivity_train,
            'Train Specificity': specificity_train,
            'Train Accuracy': accuracy_score(y_train, y_pred_train_class),
            'Test AUC': auc_test,
            'Test Sensitivity': sensitivity_test,
            'Test Specificity': specificity_test,
            'Test Accuracy': accuracy_score(y_test, y_pred_test_class),
            'TP': tp_test,
            'TN': tn_test,
            'FP': fp_test,
            'FN': fn_test,
            'Training Positive Count': tp_train + fn_train,
            'Training Negative Count': tn_train + fp_train,
            'Training predict Positive Count': tp_train + fp_train,
            'Training predict Negative Count': tn_train + fn_train,
            'Test Positive Count': tp_test + fn_test,
            'Test Negative Count': tn_test + fp_test,
            'Test predict Positive Count': tp_test + fp_test,
            'Test predict Negative Count': tn_test + fn_test

        })

    return pd.DataFrame(results)




def use_select_feature_cv(X,y,use_feature,model,n_splits=10,name="test",random_state=42,output_dir="./",save_model=True):
    X_use=X[use_feature]
    y_use=y
    #kf = KFold(n_splits=10, shuffle=True, random_state=11)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    #model = RandomForestClassifier(random_state=11, max_depth=5, n_estimators=10)
    auc_scores = []
    fold_data = []

    fold_idx = 1
    for train_index, test_index in kf.split(X_use, y_use):
        print(f"Training on fold {fold_idx}...")
        #print(train_index)
        #print(test_index)
        X_train, X_test = X_use.iloc[train_index], X_use.iloc[test_index]
        y_train, y_test = y_use.iloc[train_index], y_use.iloc[test_index]

        # 训练模型
        model.fit(X_train, y_train)
        if save_model:
            # 保存模型 to output_dir pickle
            filename = output_dir + name + ".fold" + str(fold_idx) + ".model.pickle"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

        y_pred = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc_score)

        # 获取当前折的训练集和验证集样本名
        train_names = train_index
        test_names = test_index

        fold_data.append({
            'fold': fold_idx,
            'train_samples': train_names,
            'test_samples': test_names,
            'auc_score': auc_score
        })
        fold_idx += 1

    # 转换成 DataFrame
    df_folds = pd.DataFrame(fold_data)

    # 打印每次交叉验证的AUC分数
    for i, data in df_folds.iterrows():
        print(f"AUC Score for fold {data['fold']}: {data['auc_score']}")

    # 计算平均AUC分数
    average_auc = np.mean(auc_scores)
    print(f"Average AUC Score: {average_auc}")

    # 将交叉验证结果写入DataFrame
    df_auc_scores = pd.DataFrame({'fold': range(1, n_splits+1), 'auc_score': auc_scores})
    # 将DataFrame写入CSV文件
    df_auc_scores.to_csv(output_dir + name + ".cv_auc_scores.csv", index=False)
    # 同样，将训练集和验证集的样本名和AUC分数写入另一个CSV文件
    df_folds.to_csv(output_dir + name + ".cv_folds_detail.csv", index=False)

    return average_auc


def stacking_binary_classification(X_train, y_train, X_test, y_test,estimators,  name="test", output_dir="./", select_features=None,cv_num=5):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Select features if provided
    if select_features is not None:
        X_train = X_train[select_features]
        X_test = X_test[select_features]

    # Define meta-model
    meta_lr = LogisticRegression()

    # Create stacking classifier
    stack_clf = StackingClassifier(estimators=estimators, final_estimator=meta_lr, cv=cv_num)

    # Fit the model on training data
    stack_clf.fit(X_train, y_train)

    # Predict probabilities on test data
    y_pred_proba_test = stack_clf.predict_proba(X_test)[:, 1]

    # Calculate AUC score for test data
    auc_score = roc_auc_score(y_test, y_pred_proba_test)

    # Predict probabilities on train data
    y_pred_proba_train = stack_clf.predict_proba(X_train)[:, 1]

    # Save the trained stacking model
    model_filename = f"{name}_stacking_model.pkl"
    model_filepath = os.path.join(output_dir, model_filename)
    joblib.dump(stack_clf, model_filepath)

    # Prepare DataFrame with true labels and predicted probabilities for test set
    results_df_test = pd.DataFrame({
        'true_label': y_test,
        'predicted_prob': y_pred_proba_test
    })

    # Prepare DataFrame with true labels and predicted probabilities for train set
    results_df_train = pd.DataFrame({
        'true_label': y_train,
        'predicted_prob': y_pred_proba_train
    })

    # Save the DataFrames to CSV files
    csv_filename_test = f"{name}_test_results.csv"
    csv_filepath_test = os.path.join(output_dir, csv_filename_test)
    results_df_test.to_csv(csv_filepath_test, index=False)

    csv_filename_train = f"{name}_train_results.csv"
    csv_filepath_train = os.path.join(output_dir, csv_filename_train)
    results_df_train.to_csv(csv_filepath_train, index=False)

    return auc_score


def use_select_feature_cv2(X, y, use_feature, model, n_splits=10, name="test", random_state=42, output_dir="./", save_model=False):
    X_use = X[use_feature]
    y_use = y
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    fold_data = []
    test_results = []
    train_results = []

    fold_idx = 1
    for train_index, test_index in kf.split(X_use, y_use):
        print(f"Training on fold {fold_idx}...")
        X_train, X_test = X_use.iloc[train_index], X_use.iloc[test_index]
        y_train, y_test = y_use.iloc[train_index], y_use.iloc[test_index]

        # 训练模型
        model.fit(X_train, y_train)
        if save_model:
            filename = f"{output_dir}{name}.fold{fold_idx}.model.pickle"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred_label = model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc_score)

        # 获取当前折的训练集和验证集样本名
        test_names = X.index[test_index].tolist()
        train_names = X.index[train_index].tolist()

        # 存储测试结果
        for i in range(len(test_index)):
            test_results.append({
                'sample_name': test_names[i],
                'predicted_probability': y_pred_prob[i],
                'actual_label': y_test.iloc[i],
                'fold': fold_idx,
                'name': name
            })
                # 存储训练结果
        y_pred_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred_label_train = model.predict(X_train)
        for i in range(len(train_index)):
            train_results.append({
                'sample_name': train_names[i],
                'predicted_probability': y_pred_prob_train[i],
                'actual_label': y_train.iloc[i],
                'fold': fold_idx,
                'name': name
            })

        fold_data.append({
            'fold': fold_idx,
            'train_samples': train_index.tolist(),
            'test_samples': test_index.tolist(),
            'auc_score': auc_score
        })
        fold_idx += 1

    # 转换成 DataFrame
    df_folds = pd.DataFrame(fold_data)
    df_test_results = pd.DataFrame(test_results)
    df_train_results = pd.DataFrame(train_results)

    # 打印每次交叉验证的AUC分数
    for i, data in df_folds.iterrows():
        print(f"AUC Score for fold {data['fold']}: {data['auc_score']}")

    # 计算平均AUC分数
    average_auc = np.mean(auc_scores)
    print(f"Average AUC Score: {average_auc}")

    # 将交叉验证结果写入DataFrame
    df_auc_scores = pd.DataFrame({'fold': range(1, n_splits+1), 'auc_score': auc_scores})
    # 将DataFrame写入CSV文件
    #df_auc_scores.to_csv(f"{output_dir}{name}.cv_auc_scores.csv", index=False)
    # 同样，将训练集和验证集的样本名和AUC分数写入另一个CSV文件
    #df_folds.to_csv(f"{output_dir}{name}.cv_folds_detail.csv", index=False)
    # 将测试结果写入CSV文件
    #df_test_results.to_csv(f"{output_dir}{name}.cv_test_results.csv", index=False)

    return average_auc, df_test_results,df_train_results




from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer, Normalizer

def get_scaler(scaler_type):
    # 创建一个映射简写到相应缩放器类的字典
    scaler_dict = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    return scaler_dict[scaler_type.lower()]()




def use_select_feature_cv3(X, y, use_feature, model, scaler_type,n_splits=10, name="test", random_state=42, output_dir="./", save_model=False):
    X_use = X[use_feature]
    y_use = y
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    fold_data = []
    test_results = []
    train_results = []

    fold_idx = 1
    for train_index, test_index in kf.split(X_use, y_use):
        print(f"Training on fold {fold_idx}...")
        X_train, X_test = X_use.iloc[train_index], X_use.iloc[test_index]
        y_train, y_test = y_use.iloc[train_index], y_use.iloc[test_index]
        scaler = get_scaler(scaler_type)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练模型
        model.fit(X_train, y_train)
        if save_model:
            filename = f"{output_dir}{name}.fold{fold_idx}.model.pickle"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred_label = model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc_score)

        # 获取当前折的训练集和验证集样本名
        test_names = X.index[test_index].tolist()
        train_names = X.index[train_index].tolist()

        # 存储测试结果
        for i in range(len(test_index)):
            test_results.append({
                'sample_name': test_names[i],
                'predicted_probability': y_pred_prob[i],
                'actual_label': y_test.iloc[i],
                'fold': fold_idx,
                'name': name
            })
                # 存储训练结果
        y_pred_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred_label_train = model.predict(X_train)
        for i in range(len(train_index)):
            train_results.append({
                'sample_name': train_names[i],
                'predicted_probability': y_pred_prob_train[i],
                'actual_label': y_train.iloc[i],
                'fold': fold_idx,
                'name': name
            })

        fold_data.append({
            'fold': fold_idx,
            'train_samples': train_index.tolist(),
            'test_samples': test_index.tolist(),
            'auc_score': auc_score
        })
        fold_idx += 1

    # 转换成 DataFrame
    df_folds = pd.DataFrame(fold_data)
    df_test_results = pd.DataFrame(test_results)
    df_train_results = pd.DataFrame(train_results)

    # 打印每次交叉验证的AUC分数
    for i, data in df_folds.iterrows():
        print(f"AUC Score for fold {data['fold']}: {data['auc_score']}")

    # 计算平均AUC分数
    average_auc = np.mean(auc_scores)
    print(f"Average AUC Score: {average_auc}")

    # 将交叉验证结果写入DataFrame
    df_auc_scores = pd.DataFrame({'fold': range(1, n_splits+1), 'auc_score': auc_scores})
    # 将DataFrame写入CSV文件
    #df_auc_scores.to_csv(f"{output_dir}{name}.cv_auc_scores.csv", index=False)
    # 同样，将训练集和验证集的样本名和AUC分数写入另一个CSV文件
    #df_folds.to_csv(f"{output_dir}{name}.cv_folds_detail.csv", index=False)
    # 将测试结果写入CSV文件
    #df_test_results.to_csv(f"{output_dir}{name}.cv_test_results.csv", index=False)

    return average_auc, df_test_results,df_train_results

def use_select_feature_cv4(X, y, use_feature, model, scaler_type, n_splits=10, name="test", random_state=42, output_dir="./",type="test", save_model=False):
    X_use = X[use_feature]
    y_use = y
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc_scores = []
    fold_data = []
    test_results = []
    train_results = []

    all_y_test = []
    all_y_pred_proba = []
    all_y_pred = []

    fold_idx = 1
    for train_index, test_index in kf.split(X_use, y_use):
        print(f"Training on fold {fold_idx}...")
        X_train, X_test = X_use.iloc[train_index], X_use.iloc[test_index]
        y_train, y_test = y_use.iloc[train_index], y_use.iloc[test_index]
        scaler = get_scaler(scaler_type)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练模型
        model.fit(X_train, y_train)
        if save_model:
            filename = f"{output_dir}{name}.fold{fold_idx}.model.pickle"
            with open(filename, 'wb') as file:
                pickle.dump(model, file)

        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred_label = model.predict(X_test)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc_score)

        # 获取当前折的训练集和验证集样本名
        test_names = X.index[test_index].tolist()
        train_names = X.index[train_index].tolist()

        # 存储测试结果
        for i in range(len(test_index)):
            test_results.append({
                'sample_name': test_names[i],
                'predicted_probability': y_pred_prob[i],
                'actual_label': y_test.iloc[i],
                'fold': fold_idx,
                'name': name
            })
        
        # 存储训练结果
        y_pred_prob_train = model.predict_proba(X_train)[:, 1]
        y_pred_label_train = model.predict(X_train)
        for i in range(len(train_index)):
            train_results.append({
                'sample_name': train_names[i],
                'predicted_probability': y_pred_prob_train[i],
                'actual_label': y_train.iloc[i],
                'fold': fold_idx,
                'name': name
            })

        fold_data.append({
            'fold': fold_idx,
            'train_samples': train_index.tolist(),
            'test_samples': test_index.tolist(),
            'auc_score': auc_score
        })
        
        # 收集所有测试集的结果用于计算整体统计指标
        all_y_test.extend(y_test)
        all_y_pred_proba.extend(y_pred_prob)
        all_y_pred.extend(y_pred_label)
        
        fold_idx += 1

    # 转换成 DataFrame
    df_folds = pd.DataFrame(fold_data)
    df_test_results = pd.DataFrame(test_results)
    df_train_results = pd.DataFrame(train_results)

    # 打印每次交叉验证的AUC分数
    for i, data in df_folds.iterrows():
        print(f"AUC Score for fold {data['fold']}: {data['auc_score']}")

    # 计算平均AUC分数
    average_auc = np.mean(auc_scores)
    print(f"Average AUC Score: {average_auc}")

    # 计算整体统计指标
    auc = roc_auc_score(all_y_test, all_y_pred_proba)
    accuracy = accuracy_score(all_y_test, all_y_pred)
    tn, fp, fn, tp = confusion_matrix(all_y_test, all_y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    stat_df = pd.DataFrame({
        'Metric': ['name', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'TN', 'TP', 'FN', 'FP', 'Feature_num'],
        'Value': [name, auc, accuracy, sensitivity, specificity, tn, tp, fn, fp, len(use_feature)]
    })
    
    file_name = name+"_"+type+"_statistics.csv"
    stat_df.to_csv(os.path.join(output_dir,file_name), index=False)

    return average_auc, df_test_results, df_train_results, stat_df

        

def select_features_cv_importance_multi_model3(X, y, scaler_type, model, n_splits=5, freq_list=[3], top_n_list=[5], save_detail=False):
    feature_rankings = {i: pd.Series(0, index=X.columns) for i in range(n_splits)}  # 初始化特征排名字典

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_index = 0
    details = []  # 用于保存详细信息

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = get_scaler(scaler_type)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)

        if hasattr(cloned_model, 'predict_proba'):
            probabilities = cloned_model.predict_proba(X_train)[:, 1]  # 假设二分类问题
        else:
            probabilities = cloned_model.decision_function(X_train)  # 如果模型没有 predict_proba 方法

        predictions = cloned_model.predict(X_train)

        if save_detail:
            for idx, (prob, pred, true_label) in enumerate(zip(probabilities, predictions, y_train)):
                sample_name = X.index[train_index[idx]]
                details.append({
                    'sample_name': sample_name,
                    'predicted_probability': prob,
                    'actual_label': true_label,
                    'fold': fold_index + 1
                })

        if hasattr(cloned_model, 'feature_importances_'):
            importances = cloned_model.feature_importances_
        elif hasattr(cloned_model, 'coef_'):
            importances = np.abs(cloned_model.coef_[0])
        else:
            continue  # 如果模型没有 feature_importances_ 或 coef_，则跳过

        # 存储每次模型的特征重要性排名
        feature_rankings[fold_index] = pd.Series(importances, index=X.columns)
        fold_index += 1

    # 用于保存满足不同条件的特征列表
    results = {}

    for top_n in top_n_list:
        for freq in freq_list:
            feature_freq = pd.Series(0, index=X.columns)
            for ranking in feature_rankings.values():
                # 获取当前排名中的前 top_n 个特征及其重要性值
                top_features_with_values = ranking.nlargest(top_n)

                # 过滤掉特征值为 0 的特征
                top_features_non_zero = top_features_with_values[top_features_with_values > 0].index

                # 更新特征频率计数器
                feature_freq[top_features_non_zero] += 1

            # 筛选出符合条件的特征
            selected_features = feature_freq[feature_freq >= freq].index.tolist()

            # 存储结果
            results[(top_n, freq)] = selected_features

    if save_detail:
        return results, pd.DataFrame(details)
    else:
        return results


def read_csv_to_dict(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        data_dict = {header: [] for header in reader.fieldnames}
        for row in reader:
            for key, value in row.items():
                if value.strip():  # 去除空值
                    data_dict[key].append(value)
    return data_dict


def save_results_to_csv(results, filename):
    data = []
    for key, features in results.items():
        data.append({
            'Top N': key[0],
            'Frequency': key[1],
            'Features': ','.join(features)
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)




def train_and_evaluate_model(X, y, model, name, output_dir):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 训练模型
    model.fit(X, y)

    # 保存训练好的模型
    model_path = os.path.join(output_dir, f"{name}_model.joblib")
    dump(model, model_path)

    # 进行预测
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # 创建包含样本名称、预测值和实际标签的数据框
    re_df = pd.DataFrame({
        'sample_name': X.index,
        'predicted_probability': y_pred_proba,
        'actual_label': y,
        'predicted_label': y_pred
    })
    re_df.to_csv(os.path.join(output_dir, f"{name}_predictions.csv"), index=False)

    # 计算统计指标
    auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    stat_df = pd.DataFrame({
        'Metric': ['name','AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'TN', 'TP', 'FN', 'FP','Feature_num'],
        'Value': [name,auc, accuracy, sensitivity, specificity, tn, tp, fn, fp,X.shape[1]]
    })
    stat_df.to_csv(os.path.join(output_dir, f"{name}_train_statistics.csv"), index=False)

    return model, re_df, stat_df




def evaluate_model(X, y, model, name, output_dir,type):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 进行预测
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # 创建包含样本名称、预测值和实际标签的数据框
    re_df = pd.DataFrame({
        'sample_name': X.index,
        'predicted_probability': y_pred_proba,
        'actual_label': y,
        'predicted_label': y_pred
    })
    re_df_file_name = name+"_"+type+"_predictions.csv"
    re_df.to_csv(os.path.join(output_dir, re_df_file_name), index=False)

    # 计算统计指标
    auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    stat_df = pd.DataFrame({
        'Metric': ['name','AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'TN', 'TP', 'FN', 'FP','Feature_num'],
        'Value': [name,auc, accuracy, sensitivity, specificity, tn, tp, fn, fp,X.shape[1]]
    })
    file_name = name+"_"+type+"_statistics.csv"
    stat_df.to_csv(os.path.join(output_dir, file_name), index=False)

    return re_df, stat_df



def split_and_save_data(X, y, test_size=0.4, random_state=42, shuffle=True, stratify=True, name="test", output_dir="./"):
        # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify
    )

    # 保存分割后的数据集
    X_train.to_csv(os.path.join(output_dir, f"{name}_X_train.csv"))
    X_test.to_csv(os.path.join(output_dir, f"{name}_X_test.csv"))
    y_train.to_csv(os.path.join(output_dir, f"{name}_y_train.csv"))
    y_test.to_csv(os.path.join(output_dir, f"{name}_y_test.csv"))

    return X_train, X_test, y_train, y_test



__all__ = ['select_features_cv_importance_multi_model', 'save_results_to_csv', 'load_results_from_csv',
           'evaluate_feature_lists_with_counts', 'use_select_feature_cv', 'stacking_binary_classification',
           'use_select_feature_cv2', 'use_select_feature_cv3', 'use_select_feature_cv4',
           'select_features_cv_importance_multi_model3', 'read_csv_to_dict', 'save_results_to_csv',
           'train_and_evaluate_model', 'evaluate_model', 'split_and_save_data']



