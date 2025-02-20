# EML
easy machine learning

本包提供集成好的机器学习常用的特征筛选、训练测试模型函数，方便初学者使用

## 功能特性

- 特性1：简单
- 特性2：实用
- ...

## 安装

确保你已经安装了 Python 3.6 或更高版本。推荐使用虚拟环境来管理依赖项。

```bash
pip install git+https://github.com/yty420/EML.git
```



## 概述
EML.py 脚本包含多个用于机器学习任务的函数，包括特征选择、模型评估、交叉验证和结果保存等。以下是每个函数的介绍和使用说明。

## 1. 特征选择函数

### select_features_chi2
- **功能**: 基于卡方检验选择特征。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - num_features: 选择的特征数量。
- **输出**:
  - 选择的特征名称列表。

### select_features_mi
- **功能**: 基于互信息量选择特征。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - num_features: 选择的特征数量。
- **输出**:
  - 选择的特征名称列表。

### select_features_by_variance
- **功能**: 基于方差排序选择特征。
- **输入参数**:
  - X: 特征数据框。
  - num_features: 选择的特征数量。
- **输出**:
  - 选择的特征名称列表。

### select_features_variance_bythreshold
- **功能**: 基于方差阈值选择特征。
- **输入参数**:
  - X: 特征数据框。
  - threshold: 方差阈值。
- **输出**:
  - 选择的特征名称列表。

### select_features_sfs
- **功能**: 基于递归特征消除（RFE）选择特征。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 基础模型。
  - n_features_to_select: 选择的特征数量。
  - direction: 选择方向（forward 或 backward）。
  - cv: 交叉验证的折数。
- **输出**:
  - 选择的特征名称列表。

### select_features_rfe
- **功能**: 基于递归特征消除（RFE）选择特征。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 基础模型。
  - n_features_to_select: 选择的特征数量。
  - step: 每次递归消除的特征比例。
  - verbose: 是否输出详细信息。
- **输出**:
  - 选择的特征名称列表。

### select_features_cv_importance_single_model
- **功能**: 通过交叉验证计算单个模型的特征重要性。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 基础模型。
  - cv: 交叉验证的折数。
  - top_n: 每折选择的特征数量。
  - freq: 特征出现的频率阈值。
- **输出**:
  - 满足频率阈值的特征列表。

### select_features_cv_importance_multi_model
- **功能**: 通过交叉验证计算多个模型的特征重要性。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 基础模型。
  - cv: 交叉验证的折数。
  - freq_list: 特征频率阈值列表。
  - top_n_list: 每折选择的特征数量列表。
- **输出**:
  - 满足不同频率阈值的特征列表。

### select_features_cv_importance_multi_model2
- **功能**: 通过交叉验证计算多个模型的特征重要性。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 基础模型。
  - cv: 交叉验证的折数。
  - freq_list: 特征频率阈值列表。
  - top_n_list: 每折选择的特征数量列表。
- **输出**:
  - 满足不同频率阈值的特征列表。

### select_features_cv_importance_multi_model3
- **功能**: 通过交叉验证计算多个模型的特征重要性。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - scaler_type: 缩放器类型。
  - model: 基础模型。
  - n_splits: 交叉验证的折数。
  - freq_list: 特征频率阈值列表。
  - top_n_list: 每折选择的特征数量列表。
  - save_detail: 是否保存详细信息。
- **输出**:
  - 满足不同频率阈值的特征列表。

## 2. 模型训练和评估函数

### evaluate_feature_lists_with_counts
- **功能**: 评估不同特征列表的性能。
- **输入参数**:
  - X_train: 训练集特征。
  - X_test: 测试集特征。
  - y_train: 训练集标签。
  - y_test: 测试集标签。
  - feature_lists: 特征列表字典。
  - model: 模型。
  - lp: 是否绘制学习曲线。
  - scale: 是否进行特征缩放。
- **输出**:
  - 包含评估结果的 DataFrame。

### train_and_evaluate_model
- **功能**: 训练模型并保存结果。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 模型。
  - name: 模型名称。
  - output_dir: 输出目录。
- **输出文件**:
  - 保存的模型文件。
  - 预测结果的 CSV 文件。
  - 统计指标的 CSV 文件。

### evaluate_model
- **功能**: 评估模型性能。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - model: 训练好的模型。
  - name: 模型名称。
  - output_dir: 输出目录。
  - type: 结果类型。
- **输出文件**:
  - 预测结果的 CSV 文件。
  - 统计指标的 CSV 文件。

### stacking_binary_classification
- **功能**: 基于集成学习中的堆叠（Stacking）方法进行二分类。
- **输入参数**:
  - X_train: 训练集特征。
  - y_train: 训练集标签。
  - X_test: 测试集特征。
  - y_test: 测试集标签。
  - estimators: 基础模型列表。
  - name: 模型名称。
  - output_dir: 输出目录。
  - select_features: 选择的特征。
  - cv_num: 交叉验证的折数。
- **输出文件**:
  - 保存的模型文件。
  - 测试集预测结果的 CSV 文件。
  - 训练集预测结果的 CSV 文件。

## 3. 交叉验证函数

### use_select_feature_cv4
- **功能**: 使用交叉验证评估模型性能。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - use_feature: 选择的特征列表。
  - model: 模型。
  - scaler_type: 缩放器类型。
  - n_splits: 交叉验证的折数。
  - name: 模型名称。
  - random_state: 随机种子。
  - output_dir: 输出目录。
  - type: 结果类型。
  - save_model: 是否保存模型。
- **输出文件**:
  - 预测结果的 CSV 文件。
  - 统计指标的 CSV 文件。

### use_select_feature_cv3
- **功能**: 使用交叉验证评估模型性能，并保存结果。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - use_feature: 选择的特征列表。
  - model: 模型。
  - scaler_type: 缩放器类型。
  - n_splits: 交叉验证的折数。
  - name: 模型名称。
  - random_state: 随机种子。
  - output_dir: 输出目录。
  - save_model: 是否保存模型。
- **输出文件**:
  - 预测结果的 CSV 文件。

## 4. 数据分割函数

### split_and_save_data
- **功能**: 分割数据集并保存。
- **输入参数**:
  - X: 特征数据框。
  - y: 标签数据。
  - test_size: 测试集比例。
  - random_state: 随机种子。
  - shuffle: 是否打乱数据。
  - stratify: 是否分层抽样。
  - name: 数据集名称。
  - output_dir: 输出目录。
- **输出文件**:
  - 分割后的训练集特征、测试集特征、训练集标签、测试集标签 CSV 文件。

## 5. 其他辅助函数

### get_scaler
- **功能**: 获取缩放器。
- **输入参数**:
  - scaler_type: 缩放器类型（如 minmax, standard, robust）。
- **输出**:
  - 初始化的缩放器对象。

### read_csv_to_dict
- **功能**: 读取 CSV 文件并转换为字典。
- **输入参数**:
  - file_path: CSV 文件路径。
- **输出**:
  - 以 CSV 列头为键，列值为列表的字典。

### save_results_to_csv
- **功能**: 将特征选择结果保存为 CSV 文件。
- **输入参数**:
  - results: 特征选择结果字典。
  - filename: 保存文件名。
- **输出文件**:
  - 特征选择结果的 CSV 文件。

### load_results_from_csv
- **功能**: 从 CSV 文件中加载特征选择结果。
- **输入参数**:
  - filename: CSV 文件路径。
- **输出**:
  - 特征选择结果字典。

## 文件结构
脚本中的函数会生成以下类型的文件：
- 模型文件（如 .joblib 或 .pkl）。
- 预测结果的 CSV 文件。
- 统计指标的 CSV 文件。
- 分割后数据集的 CSV 文件。

当你使用这些函数时，可以通过调整输入参数来适应不同的机器学习任务，并根据输出文件评估模型性能和特征选择结果。



