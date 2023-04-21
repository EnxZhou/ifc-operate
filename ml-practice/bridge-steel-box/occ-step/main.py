import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, KFold

label_feature = LabelEncoder()

def PreprocessDataForAP():
    xyz_feature = [ 'centreOfMassX', 'centreOfMassY', 'centreOfMassZ', 'maxFaceMass'
                   ]
    # xyz_feature = [ 'centreOfMassX', 'centreOfMassY', 'centreOfMassZ']

    label_name = 'class'

    file_path = 'dataset/label-v3/12train-3test/'
    train_data = pd.read_csv(file_path + 'AP-train-scw_C3-JD-27.csv')
    data = train_data
    data.reset_index(drop=True, inplace=True)

    data.dropna(inplace=True, subset=label_name)


    scaler = MinMaxScaler()
    # 对当前需要进行归一化的字段进行归一化操作
    data[xyz_feature] = scaler.fit_transform(data[xyz_feature])

    object_feature = list(data.select_dtypes(include=['object']).columns)
    data.drop(object_feature, axis=1, inplace=True)

    return data

def PreprocessData():
    # xyz_feature = ['solidMass',
    #                'centreOfMassX', 'centreOfMassY', 'centreOfMassZ',
    #                'surfaceArea', 'maxFaceMass',
    #                'maxFaceCentreOfMassX', 'maxFaceCentreOfMassY', 'maxFaceCentreOfMassZ',
    #                'maxFaceAxisLocationX', 'maxFaceAxisLocationY', 'maxFaceAxisLocationZ',
    #                'maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
    #                'maxFacePerimeter',
    #                'maxFaceMaxEdgeCentreX', 'maxFaceMaxEdgeCentreY', 'maxFaceMaxEdgeCentreZ',
    #                'maxFaceMinEdgeCentreX', 'maxFaceMinEdgeCentreY', 'maxFaceMinEdgeCentreZ',
    #                'maxFaceEdgeLengthAverage', 'maxFaceEdgeLengthVariance', 'minFaceMass',
    #                'minFaceCentreOfMassX', 'minFaceCentreOfMassY', 'minFaceCentreOfMassZ',
    #                'minFaceAxisLocationX', 'minFaceAxisLocationY', 'minFaceAxisLocationZ',
    #                'minFaceAxisDirectX', 'minFaceAxisDirectY', 'minFaceAxisDirectZ',
    #                'minFacePerimeter',
    #                'minFaceMaxEdgeCentreX', 'minFaceMaxEdgeCentreY', 'minFaceMaxEdgeCentreZ',
    #                'minFaceMinEdgeCentreX', 'minFaceMinEdgeCentreY', 'minFaceMinEdgeCentreZ',
    #                'minFaceEdgeLengthAverage', 'minFaceEdgeLengthVariance',
    #                'faceMassAverage', 'faceMassVariance', 'edgeLenSum']
    xyz_feature = ['centreOfMassX', 'centreOfMassY', 'centreOfMassZ', 'maxFaceMass',
                   'maxFaceCentreOfMassX', 'maxFaceCentreOfMassY', 'maxFaceCentreOfMassZ',
                   'maxFaceAxisLocationX', 'maxFaceAxisLocationY', 'maxFaceAxisLocationZ',
                   'maxFaceAxisDirectX', 'maxFaceAxisDirectY', 'maxFaceAxisDirectZ',
                   'faceMassAverage', 'faceMassVariance']

    label_name = 'class'

    file_path = 'dataset/label-v3/12train-3test/'
    train_data = pd.read_csv(file_path + 'AP-train2.csv')
    test_data = pd.read_csv(file_path + 'test.csv')
    train_data['is_train'] = True
    data = pd.concat([train_data, test_data])
    data.reset_index(drop=True, inplace=True)

    data.dropna(inplace=True, subset=label_name)
    # data.fillna(data.median(), inplace=True)
    # data.dropna(inplace=True)

    is_train_label = data['is_train']

    data_groups = data.groupby('segment_no')

    scaler = MinMaxScaler()
    for segment_no, group in data_groups:
        # 对当前分组中需要进行归一化的字段进行归一化操作
        group[xyz_feature] = scaler.fit_transform(group[xyz_feature])
        # 将归一化后的分组数据更新回原始数据中
        data.update(group)

    data['is_train'] = is_train_label

    # le = LabelEncoder()
    # label = le.fit_transform(data[label_name])
    label = label_feature.fit_transform(data[label_name])
    object_feature = list(data.select_dtypes(include=['object']).columns)
    object_feature.remove("is_train")
    data.drop(object_feature, axis=1, inplace=True)
    data[label_name] = label

    train_data = data[data['is_train'].notnull()].reset_index(drop=True)
    train_data.drop('is_train', axis=1, inplace=True)
    test_data = data[data['is_train'].isna()].reset_index(drop=True)
    test_data.drop('is_train', axis=1, inplace=True)

    y_train = train_data[label_name].values
    train_data.drop(label_name, axis=1, inplace=True)
    y_test = test_data[label_name].values
    test_data.drop(label_name, axis=1, inplace=True)
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


def Train1(input_X_train, input_y_train, input_X_test,
           input_y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    import xgboost
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
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

    # 集合算法树模型
    GBDT_param = {
        'loss': 'log_loss',
        'learning_rate': 0.1,
        'n_estimators': 30,
        'max_depth': 3,
        'min_samples_split': 300
    }
    GBDT_clf = GradientBoostingClassifier()  # GBDT模型

    tree_param = {
        'criterion': 'gini',
        'max_depth': 30,
        'min_impurity_decrease': 0.1,
        'min_samples_leaf': 2

    }
    Tree_clf = DecisionTreeClassifier(**tree_param)  # 决策树模型

    # xgboost_param = {
    #     'learning_rate': 0.01,
    #     'reg_alpha': 0.,
    #     'max_depth': 3,
    #     'gamma': 0,
    #     'min_child_weight': 1
    #
    # }
    # xgboost_clf = xgboost.XGBClassifier(**xgboost_param)  # xgboost模型

    RFC_clf = RandomForestClassifier()

    model_lgb = lgb.LGBMClassifier(
        num_leaves=2 ** 5 - 1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
        max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
        n_estimators=2000, subsample=1, colsample_bytree=1,
    )

    # xgboost_clf.fit(X_train, y_train)
    GBDT_clf.fit(X_train, y_train)
    Tree_clf.fit(X_train, y_train)
    RFC_clf.fit(X_train, y_train)
    model_lgb.fit(X_train, y_train)

    # K折交叉检验
    K_model_list = [Tree_clf, GBDT_clf, RFC_clf, model_lgb]
    kFold = KFold(n_splits=6, shuffle=True, random_state=2)
    K_result = pd.DataFrame()
    for i, val in enumerate(K_model_list):
        score = cross_validate(val, X_train, y_train, cv=kFold, scoring='accuracy',
                               error_score='raise')
        K_result.loc[i, 'cross_accuracy'] = score['test_score'].mean()
        K_result.loc[i, 'cross_stand'] = score['test_score'].std()

        y_pred = val.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        K_result.loc[i, 'predict_accuracy'] = accuracy
        # K_result.loc[i, 'report'] = classification_report(y_test,y_pred)
    K_result.index = pd.Series(['Tree', 'GBDT', 'RFC', 'lgb'])
    print(K_result)


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

    RFC_clf = RandomForestClassifier()

    RFC_clf.fit(X_train, y_train)

    # K折交叉检验
    kFold = KFold(n_splits=6, shuffle=True, random_state=42)
    score = cross_validate(RFC_clf, X_train, y_train, cv=kFold, scoring='accuracy',
                           error_score='raise')
    y_pred = RFC_clf.predict(X_test)
    df_y_pred = pd.DataFrame(y_pred)
    y_pred_feature = label_feature.inverse_transform(y_pred)
    df_y_pred_feature = pd.DataFrame(y_pred_feature)
    prob = RFC_clf.predict_proba(X_test)
    df_prob = pd.DataFrame(prob)
    df_result = pd.concat([df_y_pred, df_y_pred_feature, df_prob], axis=1)
    df_result.to_csv('predict_result.csv', index=False)


def AP_train(data):
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import AgglomerativeClustering
    has_nan = data.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    for preference in np.linspace(-50,-30,21):
        af_clf = AffinityPropagation(preference=preference)
        af_clf.fit(data)
        cluster_centers_indices = af_clf.cluster_centers_indices_
        labels = af_clf.labels_
        n_clusters = len(cluster_centers_indices)
        print("preference={}, n_clusters={}".format(preference,n_clusters))



    # agg_clf = AgglomerativeClustering(n_clusters=10, linkage='ward')
    # agg_clf.fit(data)
    # labels = agg_clf.labels_

    # af_clf = AffinityPropagation(preference=-50)
    # af_clf.fit(data)
    # cluster_centers_indices = af_clf.cluster_centers_indices_
    # labels = af_clf.labels_
    # n_clusters = len(cluster_centers_indices)
    # print("n_clusters={}".format(n_clusters))
    #
    # df_labels = pd.DataFrame(labels)
    # df_labels.to_csv('ap_result.csv', index=False)


def main():
    # X_train, y_train, X_test, y_test = PreprocessData()
    # print(label_feature.classes_)
    # Train2(X_train, y_train, X_test, y_test)
    data = PreprocessDataForAP()
    AP_train(data)


if __name__ == '__main__':
    main()
