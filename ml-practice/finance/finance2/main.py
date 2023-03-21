import pandas as pd
import numpy as np


def Finance2():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    data = pd.concat([train, test], axis=0)
    data.index = range(len(data))
    # data.replace('?', np.nan, regex=False, inplace=True)

    numerical_feature = list(data.select_dtypes(exclude=['object']).columns)  # 数值型变量
    object_feature = list(data.select_dtypes(include=['object']).columns)

    column_name = []
    unique_value = []
    for col in object_feature:
        column_name.append(col)
        unique_value.append(data[col].nunique())

    # df = pd.DataFrame()
    # df['col_name']=column_name
    # df['value'] = unique_value
    # df = df.sort_values('value',ascending=False)

    # 单独看某个字段
    # print(data['property_damage'].value_counts())
    data['property_damage'] = data['property_damage'].map({'NO': 0, 'YES': 1, '?': 2})
    data['police_report_available'] = data['police_report_available'].map({'NO': 0, 'YES': 1, '?': 2})

    # 将时间object转换为datetime64
    data['policy_bind_date'] = pd.to_datetime(data['policy_bind_date'])
    data['incident_date'] = pd.to_datetime(data['incident_date'])

    base_date = data['policy_bind_date'].min()
    data['policy_bind_date_diff'] = (data['policy_bind_date'] - base_date).dt.days
    data['incident_date_diff'] = (data['incident_date'] - base_date).dt.days
    data['incident_date_policy_bind_date_diff'] = data['incident_date_diff'] - data['policy_bind_date_diff']
    data.drop(['policy_bind_date', 'incident_date'], axis=1, inplace=True)
    data.drop(['policy_id'], axis=1, inplace=True)
    object_feature = list(data.select_dtypes(include=['object']).columns)

    # 标签编码
    from sklearn.preprocessing import LabelEncoder
    for col in object_feature:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # 数据集切分
    train_data = data[data['fraud'].notnull()]
    test_data = data[data['fraud'].isnull()]
    X_train = train_data.copy()
    X_train.drop(['fraud'], axis=1, inplace=True)
    y_label = train_data['fraud']
    # X_train.to_csv("tmp_train.csv", index=False, sep=',')
    return X_train, y_label, test_data


def Train(mean_X_train, y_label):
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
        score = cross_validate(val, X_train, y_train, cv=3, scoring='accuracy')
        K_result.loc[i, 'accuracy'] = score['test_score'].mean()
    K_result.index = pd.Series(['Tree', 'GBDT', 'XGBoost', 'RFC', 'lgb'])
    print(K_result)


def Train2(X_train, y_label, test_data):
    import lightgbm as lgb
    from sklearn.model_selection import cross_validate
    model_lgb = lgb.LGBMClassifier(
        num_leaves=2 ** 5 - 1, reg_alpha=0.25, reg_lambda=0.25, objective='binary',
        max_depth=-1, learning_rate=0.005, min_child_samples=3, random_state=2022,
        n_estimators=2000, subsample=1, colsample_bytree=1,
    )
    # 模型训练
    model_lgb.fit(X_train, y_label)

    score = cross_validate(model_lgb, X_train, y_label, cv=6, scoring='accuracy')
    print(score['test_score'].mean())

    # AUC评测： 以proba进行提交，结果会更好
    # y_pred = model_lgb.predict_proba(test_data.drop(['fraud'], axis=1))
    # import time
    # nowTime = time.strftime('%Y%m%d-%H%M%S')
    # sub_df = pd.read_csv('./submission.csv')
    # sub_df['fraud'] = y_pred[:, 1]
    # sub_df.to_csv('./result-' + nowTime + '.csv', index=False)


def main():
    X_train, y_lable, test_data = Finance2()
    # Train(X_train, y_lable)
    Train2(X_train,y_lable,test_data)


if __name__ == '__main__':
    main()
