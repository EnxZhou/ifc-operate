import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.decomposition import PCA

label_feature = LabelEncoder()


def PreprocessData():
    xyz_feature = ['solidMass',
                   'centreOfMassX', 'centreOfMassY', 'centreOfMassZ',
                   'surfaceArea', 'maxFaceMass',
                   'maxFaceCentreOfMassX', 'maxFaceCentreOfMassY', 'maxFaceCentreOfMassZ',
                   'maxFaceAxisLocationX', 'maxFaceAxisLocationY', 'maxFaceAxisLocationZ',
                   'maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
                   'maxFacePerimeter',
                   'maxFaceMaxEdgeCentreX', 'maxFaceMaxEdgeCentreY', 'maxFaceMaxEdgeCentreZ',
                   'maxFaceMinEdgeCentreX', 'maxFaceMinEdgeCentreY', 'maxFaceMinEdgeCentreZ',
                   'maxFaceEdgeLengthAverage', 'maxFaceEdgeLengthVariance', 'minFaceMass',
                   'minFaceCentreOfMassX', 'minFaceCentreOfMassY', 'minFaceCentreOfMassZ',
                   'minFaceAxisLocationX', 'minFaceAxisLocationY', 'minFaceAxisLocationZ',
                   'minFaceAxisDirectX', 'minFaceAxisDirectY', 'minFaceAxisDirectZ',
                   'minFacePerimeter',
                   'minFaceMaxEdgeCentreX', 'minFaceMaxEdgeCentreY', 'minFaceMaxEdgeCentreZ',
                   'minFaceMinEdgeCentreX', 'minFaceMinEdgeCentreY', 'minFaceMinEdgeCentreZ',
                   'minFaceEdgeLengthAverage', 'minFaceEdgeLengthVariance',
                   'faceMassAverage', 'faceMassVariance', 'edgeCount', 'edgeLenSum']
    # xyz_feature = ['centreOfMassX', 'centreOfMassY', 'centreOfMassZ', 'maxFaceMass',
    #                'maxFaceCentreOfMassX', 'maxFaceCentreOfMassY', 'maxFaceCentreOfMassZ',
    #                'maxFaceAxisLocationX', 'maxFaceAxisLocationY', 'maxFaceAxisLocationZ',
    #                'maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
    #                'faceMassAverage', 'faceMassVariance']

    direct_feature = ['maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
                      'minFaceAxisDirectX', 'minFaceAxisDirectY', 'minFaceAxisDirectZ']

    label_name = 'class'

    file_path = 'dataset/label-v3.1/xmSS9_extend_sc27train-lm61test/'
    train_data = pd.read_csv(file_path + 'train.csv')
    test_data = pd.read_csv(file_path + 'test.csv')
    train_data['is_train'] = True
    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)

    data[direct_feature] = data[direct_feature].abs()

    data.dropna(inplace=True, subset=label_name)
    # data.fillna(data.median(), inplace=True)
    # data.dropna(inplace=True)

    is_train_label = data['is_train']

    # 归一化操作
    scaler = MinMaxScaler()
    if "segment_no" in data.columns:
        data_groups = data.groupby('segment_no')
        for segment_no, group in data_groups:
            # 对当前分组中需要进行归一化的字段进行归一化操作
            group[xyz_feature] = scaler.fit_transform(group[xyz_feature])
            # 将归一化后的分组数据更新回原始数据中
            data.update(group)
    else:
        data[xyz_feature] = scaler.fit_transform(data[xyz_feature])

    data['is_train'] = is_train_label

    label = label_feature.fit_transform(data[label_name])
    object_feature = list(data.select_dtypes(include=['object']).columns)
    object_feature.remove("is_train")
    data.drop(object_feature, axis=1, inplace=True)
    data[label_name] = label

    train_data = data[data['is_train'].notnull()].reset_index(drop=True)
    train_data.drop('is_train', axis=1, inplace=True)

    # 调用函数找出最没用的特征名称
    least_useful_features = find_least_useful_features(train_data, target_column=label_name)
    train_data.drop(least_useful_features, axis=1, inplace=True)

    test_data = data[data['is_train'].isna()].reset_index(drop=True)
    test_data.drop('is_train', axis=1, inplace=True)
    y_train = train_data[label_name].values
    train_data.drop(label_name, axis=1, inplace=True)
    reduced_train_data = perform_pca(train_data, 3)

    y_test = test_data[label_name].values
    test_data.drop(label_name, axis=1, inplace=True)
    test_data.drop(least_useful_features, axis=1, inplace=True)
    reduced_test_data = perform_pca(test_data, 3)
    # return reduced_train_data, y_train, reduced_test_data, y_test
    return train_data, y_train, test_data, y_test


def show_object_col(data):
    object_feature = list(data.select_dtypes(include=['object']).columns)
    object_column_name = []
    unique_value = []
    for col in object_feature:
        object_column_name.append(col)
        unique_value.append(data[col].nunique())
    # print("object_column", object_column_name)
    # print("numerical_column", numerical_column_name)
    df = pd.DataFrame()
    df['col_name'] = object_column_name
    df['value'] = unique_value
    df = df.sort_values('value', ascending=False)
    print(df)


def show_distplot(serial_df):
    plt.figure(figsize=(10, 5))
    f = sns.FacetGrid(serial_df, col='variable', col_wrap=3, sharex=False, sharey=False)  # 生成画布，最多三列，不共享x、y轴
    f.map(sns.distplot, "value")
    plt.show()


def show_numerical_cor(data):
    cor = data.corr()
    sns.set_theme(style="white")
    plt.figure(figsize=(16, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(cor, cmap=cmap, annot=True, linewidths=0.2,
                cbar_kws={"shrink": 0.5}, linecolor="white", fmt=".1g")
    plt.show()


def show_boxplot(serial_df):
    plt.figure(figsize=(16, 8))
    f_box = sns.FacetGrid(serial_df, col='variable', col_wrap=5, sharex=False, sharey=False)
    f_box.map(sns.boxplot, "value")
    plt.show()


def show_label_plot(data, label_name: str):
    # 使用 value_counts() 方法获取 label 列中每个值的计数
    label_counts = data[label_name].value_counts()

    # 绘制柱状图展示 label 列的分布情况
    label_counts.plot(kind='bar')
    plt.title('Distribution of Label Column')
    plt.xlabel('Label Value')
    plt.ylabel('Count')
    plt.show()


def plot_data_correlation(dataframe):
    """
    绘制Pandas DataFrame的数据关联性热图

    参数：
    dataframe (pd.DataFrame): 包含数据的Pandas DataFrame

    返回：
    None
    """
    # 计算协方差矩阵
    correlation_matrix = dataframe.corr()

    # 使用Seaborn绘制热图来显示协方差矩阵
    plt.figure(figsize=(10, 8))  # 可根据需要调整热图大小
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Data Correlation Heatmap")
    plt.show()


def find_strong_correlations(dataframe, threshold=0.9):
    """
    找出DataFrame中相关系数大于指定阈值的特征对

    参数：
    dataframe (pd.DataFrame): 包含数据的Pandas DataFrame
    threshold (float, optional): 相关系数的阈值，默认为0.7

    返回：
    correlated_features (list): 包含相关性较强的特征对的列表
    """
    # 计算协方差矩阵
    correlation_matrix = dataframe.corr()

    # 找出相关系数大于阈值的特征对
    correlated_features = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                correlated_features.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    return correlated_features


def find_least_useful_features(dataframe, target_column, threshold=0.1):
    """
    提取DataFrame中相关系数绝对值最接近于0的特征名称

    参数：
    dataframe (pd.DataFrame): 包含数据的Pandas DataFrame
    target_column (str): 目标变量列的名称
    threshold (float, optional): 相关系数的阈值，默认为0.1

    返回：
    least_useful_features (list): 包含最没用的特征名称的列表
    """
    # 计算特征与目标变量之间的相关系数
    correlations = dataframe.drop(target_column, axis=1).apply(lambda x: x.corr(dataframe[target_column]))

    # 找出相关系数绝对值最接近于0的特征
    least_useful_features = correlations[abs(correlations) < threshold].index.tolist()

    return least_useful_features


# 主成分分析,将高维度数据处理成需要的维度
def perform_pca(df, n_components=2):
    """
    对 pandas DataFrame 进行 PCA 主成分分析，并返回处理后的 pandas DataFrame。

    参数：
    df (pandas DataFrame): 需要进行 PCA 的数据，DataFrame 格式。
    n_components (int): 降维后的维数，默认为 2。

    返回：
    pandas DataFrame: 处理后的 DataFrame，包含降维后的数据。
    """
    # 将 DataFrame 转换为 NumPy 数组
    data = df.values

    # 创建 PCA 对象并拟合数据
    pca = PCA(n_components=n_components)
    pca.fit(data)

    # 对数据进行降维
    reduced_data = pca.transform(data)

    # 创建降维后的 DataFrame
    columns = [f'PC{i + 1}' for i in range(n_components)]
    df_reduced = pd.DataFrame(reduced_data, columns=columns)

    # 将原 DataFrame 的索引和降维后的数据合并为新的 DataFrame
    # df_reduced = pd.concat([df.reset_index(drop=True), df_reduced], axis=1)

    return df_reduced


def Train1(input_X_train, input_y_train, input_X_test,
           input_y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    # 划分数据集为测试集和训练集
    has_nan = input_X_train.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    # 划分数据集为测试集和训练集
    if input_X_test.empty:
        X_train, X_test, y_train, y_test = \
            train_test_split(input_X_train, input_y_train, train_size=0.7)
    else:
        X_train, X_test, y_train, y_test = \
            input_X_train, input_X_test, input_y_train, input_y_test

    SVM_param = {
        'kernel': 'rbf',  # 核函数，可以选择 'linear', 'poly', 'rbf', 'sigmoid'等
        'C': 1.0,  # 惩罚系数，用于控制错误分类的惩罚程度
        'gamma': 'scale',  # 核系数，影响样本点的映射范围，可以是 'scale'或者'auto'，也可以是一个具体的值
        'random_state': 42  # 随机种子，用于复现结果
    }

    SVM_clf = SVC(**SVM_param)

    Ada_param = {
        'base_estimator': DecisionTreeClassifier(max_depth=1),  # 弱分类器，默认是决策树
        'n_estimators': 50,  # 弱分类器的数量
        'learning_rate': 1.0,  # 学习率，用于控制每个弱分类器的权重
        'algorithm': 'SAMME.R',  # 计算权重的算法
        'random_state': 42  # 随机种子，用于复现结果
    }

    Ada_clf = AdaBoostClassifier(**Ada_param)

    # 集合算法树模型
    GBDT_param = {
        'loss': 'log_loss',
        'learning_rate': 0.1,
        'n_estimators': 30,
        'max_depth': 3,
        'min_samples_split': 300
    }
    GBDT_clf = GradientBoostingClassifier(**GBDT_param)  # GBDT模型

    tree_param = {
        'criterion': 'gini',
        'max_depth': 30,
        'min_impurity_decrease': 0.1,
        'min_samples_leaf': 2

    }
    Tree_clf = DecisionTreeClassifier(**tree_param)  # 决策树模型

    xgboost_param = {
        'learning_rate': 0.01,
        'reg_alpha': 0.,
        'max_depth': 3,
        'gamma': 0,
        'min_child_weight': 1

    }
    xgboost_clf = xgboost.XGBClassifier(**xgboost_param)  # xgboost模型

    rfc_params = {
        'n_estimators': 120,
        'min_samples_split': 4,
        'min_samples_leaf': 1,
        'max_features': 10,
        'max_depth': 10,
        'criterion': 'entropy',
        'bootstrap': True
    }

    RFC_clf = RandomForestClassifier(**rfc_params)

    LGBM_clf = lgb.LGBMClassifier(
        num_leaves=2 ** 5 - 1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
        max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
        n_estimators=2000, subsample=1, colsample_bytree=1,
    )

    SVM_clf.fit(X_train, y_train)
    Ada_clf.fit(X_train, y_train)
    xgboost_clf.fit(X_train, y_train)
    GBDT_clf.fit(X_train, y_train)
    Tree_clf.fit(X_train, y_train)
    RFC_clf.fit(X_train, y_train)
    LGBM_clf.fit(X_train, y_train)

    # K折交叉检验
    K_model_list = [SVM_clf, Ada_clf, xgboost_clf, Tree_clf, GBDT_clf, RFC_clf, LGBM_clf]
    kFold = KFold(n_splits=6, shuffle=True, random_state=42)
    model_results = []
    for i, val in enumerate(K_model_list):
        score = cross_validate(val, X_train, y_train, cv=kFold, scoring='accuracy',
                               error_score='raise')
        y_pred = val.predict(X_test)
        # 计算正确率
        accuracy = accuracy_score(y_test, y_pred)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)

        # 计算精确率（Precision）
        precision = precision_score(y_test, y_pred, average='macro')

        # 计算召回率（Recall）
        recall = recall_score(y_test, y_pred, average='macro')

        # 计算 F1 分数
        f1 = f1_score(y_test, y_pred, average='macro')

        model_results.append({
            'model_name': val.__class__.__name__,
            'cross_accuracy': score['test_score'].mean(),
            'cross_stand': score['test_score'].std(),
            'confusion_matrix': conf_matrix,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        })

    # Create a DataFrame from the results_list
    results_df = pd.DataFrame(model_results)

    # Save the DataFrame to a CSV file
    results_df.to_csv('model_evaluation_results.csv', index=False)


def Train2(input_X_train, input_y_train, input_X_test,
           input_y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    has_nan = input_X_train.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    # 划分数据集为测试集和训练集
    if input_X_test.empty:
        X_train, X_test, y_train, y_test = \
            train_test_split(input_X_train, input_y_train, train_size=0.7)
    else:
        X_train, X_test, y_train, y_test = \
            input_X_train, input_X_test, input_y_train, input_y_test

    # 根据超参数调试，得到的最优参数
    best_params = {
        'n_estimators': 150,
        'min_samples_split': 5,
        'min_samples_leaf': 5,
        'max_features': 'auto',
        'max_depth': None,
        'criterion': 'gini',
        'bootstrap': False
    }

    this_clf = RandomForestClassifier(**best_params)

    this_clf.fit(X_train, y_train)

    # K折交叉检验
    kFold = KFold(n_splits=6, shuffle=True, random_state=42)
    score = cross_validate(this_clf, X_train, y_train, cv=kFold, scoring='accuracy',
                           error_score='raise')
    y_pred = this_clf.predict(X_test)
    df_y_pred = pd.DataFrame(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print('the predict accuracy is: ', accuracy)
    y_pred_feature = label_feature.inverse_transform(y_pred)
    df_y_pred_feature = pd.DataFrame(y_pred_feature)
    prob = this_clf.predict_proba(X_test)
    df_prob = pd.DataFrame(prob)
    df_result = pd.concat([df_y_pred, df_y_pred_feature, df_prob], axis=1)
    df_result.to_csv('predict_result.csv', index=False)


# 为了进行超参数调试
def Train3(input_X_train, input_y_train):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    has_nan = input_X_train.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    # 定义 RandomForestClassifier 的超参数空间
    param_dist = {
        'n_estimators': np.arange(50, 201, 10),
        'criterion': ['gini', 'entropy'],
        'max_depth': [None] + list(np.arange(5, 26, 5)),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    # param_dist = {
    #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    #     'C': np.logspace(-3, 3, 7),  # C values in the range [0.001, 1000]
    #     'gamma': ['scale', 'auto'] + list(np.logspace(-3, 2, 6))  # Gamma values for rbf kernel
    # }

    # 划分数据集为测试集和训练集
    X_train, X_test, y_train, y_test = \
        train_test_split(input_X_train, input_y_train, train_size=0.7)

    this_clf = RandomForestClassifier()

    random_search = RandomizedSearchCV(this_clf, param_distributions=param_dist)

    random_search.fit(X_train, y_train)

    # 输出最优超参数组合和对应的得分
    print("Best Parameters:", random_search.best_params_)
    print("Best Score:", random_search.best_score_)


import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def Train4(X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # Check for NaN values in the training data
    has_nan = X_train.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    # PyTorch Neural Network Model
    input_size = X_train.shape[1]
    hidden_size = 64  # You can adjust this as needed
    output_size = 1  # For binary classification, change this to the number of classes for multi-class

    model = NeuralNetwork(input_size, hidden_size, output_size)
    # Calculate the positive weight based on the class imbalance
    num_pos = y_train.sum()
    num_neg = len(y_train) - num_pos
    pos_weight = num_neg / num_pos

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))  # Specify pos_weight
    optimizer = optim.Adam(model.parameters())

    # Convert data to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test.values)

    # Training loop for the neural network
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate the neural network model
    model.eval()
    with torch.no_grad():
        y_pred_nn = model(X_test_tensor)
        y_pred_nn = (y_pred_nn > 0.5).float()  # Binary threshold for binary classification

        # 计算正确率
        acc_nn = accuracy_score(y_test, y_pred_nn)

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred_nn)

        # 计算精确率（Precision）
        precision_nn = precision_score(y_test, y_pred_nn, average='macro')

        # 计算召回率（Recall）
        recall_nn = recall_score(y_test, y_pred_nn, average='macro')

        # 计算 F1 分数
        f1_nn = f1_score(y_test, y_pred_nn, average='macro')

    print("Neural Network Classifier Metrics:")
    print(f"Accuracy: {acc_nn:.4f}")
    print(f"ConfMatrix: {conf_matrix}")
    print(f"Precision: {precision_nn:.4f}")
    print(f"Recall: {recall_nn:.4f}")
    print(f"F1-score: {f1_nn:.4f}")


def main():
    X_train, y_train, X_test, y_test = PreprocessData()
    print(label_feature.classes_)
    # Train1(X_train, y_train, X_test, y_test)
    # Train2(X_train, y_train, X_test, y_test)
    Train4(X_train, y_train, X_test, y_test)
    # merge_X = pd.concat([X_train, X_test], ignore_index=True)
    # merge_y = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], ignore_index=True)
    # Train3(merge_X, merge_y)


if __name__ == '__main__':
    main()
