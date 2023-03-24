import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ProcessData():
    data = pd.read_csv('step-C3-JD-24_26-name_to_class.csv')
    numerical_feature = list(data.select_dtypes(exclude=['object']).columns)  # 数值型变量
    object_feature = list(data.select_dtypes(include=['object']).columns)

    # show_object_col(data)
    # 连续型变量
    serial_feature = []
    # 离散型变量
    discrete_feature = []
    # 单值变量
    unique_feature = []
    for feature in numerical_feature:
        temp = data[feature].nunique()  # 返回数据去重后的个数
        if temp == 1:
            unique_feature.append(feature)
        elif 1 < temp <= 10:
            discrete_feature.append(feature)
        else:
            serial_feature.append(feature)

    serial_df = pd.melt(data,
                        value_vars=serial_feature)  # 将连续型变量融合在一个dataframe中
    # show_distplot(serial_df)
    # show_numerical_cor(data[numerical_feature])
    # show_boxplot(serial_df)

    data.drop(['index'],axis=1,inplace=True)
    nan_rows = data[data.isna().any(axis=1)]
    data.dropna(inplace=True)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    label = le.fit_transform(data['class'])
    data.drop(object_feature, axis=1, inplace=True)
    data["class"] = label

    train_data = data[data['class'].notnull()]
    test_data = data[data['class'].isnull()]
    X_train = train_data.copy()
    X_train.drop(['class'], axis=1, inplace=True)
    y_label = train_data['class']
    return X_train, y_label, test_data


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


def Train1(mean_X_train, y_label):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    # 划分数据集为测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(mean_X_train, y_label, train_size=0.7)
    # print(X_train.shape)
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

    xgboost_param = {
        'learning_rate': 0.01,
        'reg_alpha': 0.,
        'max_depth': 3,
        'gamma': 0,
        'min_child_weight': 1

    }
    xgboost_clf = xgboost.XGBClassifier(**xgboost_param)  # xgboost模型

    RFC_clf = RandomForestClassifier()

    model_lgb = lgb.LGBMClassifier(
        num_leaves=2 ** 5 - 1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
        max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
        n_estimators=2000, subsample=1, colsample_bytree=1,
    )

    X_train = mean_X_train
    y_train = y_label
    xgboost_clf.fit(X_train, y_train)
    GBDT_clf.fit(X_train, y_train)
    Tree_clf.fit(X_train, y_train)
    RFC_clf.fit(X_train, y_train)
    model_lgb.fit(X_train, y_label)

    # K折交叉检验
    K_model_list = [Tree_clf, GBDT_clf, xgboost_clf, RFC_clf, model_lgb]
    K_result = pd.DataFrame()
    for i, val in enumerate(K_model_list):
        score = cross_validate(val, X_train, y_train, cv=6, scoring='accuracy')
        K_result.loc[i, 'accuracy'] = score['test_score'].mean()
    K_result.index = pd.Series(['Tree', 'GBDT', 'XGBoost', 'RFC', 'lgb'])
    print(K_result)


def main():
    X_train, y_label, test_data = ProcessData()
    Train1(X_train, y_label)


if __name__ == '__main__':
    main()
