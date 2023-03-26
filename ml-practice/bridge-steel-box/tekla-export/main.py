import pandas as pd
import time
from decorators import timer
import numpy as np
import preprocess.show as show
from sklearn.preprocessing import LabelEncoder


@timer
def Preprocess1():
    data = pd.read_csv('tekla-C3-JD-24_26-name_to_class.csv')
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
    # show.distplot(serial_df)
    # show.numerical_cor(data[numerical_feature])
    # show.boxplot(serial_df)
    # nan_rows = data[data.isna().any(axis=1)]

    data.dropna(inplace=True)

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


@timer
def Train1(mean_X_train, y_label):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.tree import DecisionTreeClassifier
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
    # y_train = y_label
    # xgboost_clf.fit(X_train, y_train)
    GBDT_clf.fit(X_train, y_train)
    # Tree_clf.fit(X_train, y_train)
    # RFC_clf.fit(X_train, y_train)
    # model_lgb.fit(X_train, y_train)

    # K折交叉检验
    # K_model_list = [Tree_clf, GBDT_clf, xgboost_clf, RFC_clf, model_lgb]
    K_model_list = [GBDT_clf]
    K_result = pd.DataFrame()
    for i, val in enumerate(K_model_list):
        score = cross_validate(val, X_train, y_train, cv=6, scoring='accuracy')
        K_result.loc[i, 'accuracy'] = score['test_score'].mean()
    # K_result.index = pd.Series(['Tree', 'GBDT', 'XGBoost', 'RFC', 'lgb'])
    K_result.index = pd.Series(['GBDT'])
    print(K_result)


@timer
def Train2(mean_X_train, y_label):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_validate
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # 划分数据集为测试集和训练集
    X_train, X_test, y_train, y_test = train_test_split(mean_X_train, y_label, train_size=0.7)

    GBDT_param = {
        'loss': 'log_loss',
        'learning_rate': 0.1,
        'n_estimators': 30,
        'max_depth': 3,
        'min_samples_split': 300
    }
    GBDT_clf = GradientBoostingClassifier()
    GBDT_clf.fit(X_train, y_train)

    # 使用测试集来评估模型性能
    y_pred = GBDT_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")


from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import f_classif
import pickle


class GBDTModel:
    def __init__(self, n_features=10, feature_names=None):
        self.scaler = StandardScaler()
        self.selector = SelectKBest(k=n_features)
        self.model = GradientBoostingClassifier()
        self.feature_names = feature_names

    def score_features(self, X, y):
        scores = f_classif(X, y)[0]
        if self.feature_names:
            feature_dict = dict(zip(self.feature_names,scores))
            for name, score in feature_dict.items():
                print(f"{name}: {score}")
        return scores

    def train(self, X_train, y_train):
        X_train = self.scaler.fit_transform(X_train)
        X_train = self.selector.fit_transform(X_train, y_train)
        self.model.fit(X_train, y_train)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.scaler, self.selector, self.model), f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.scaler, self.selector, self.model = pickle.load(f)

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        X_test = self.selector.transform(X_test)
        return self.model.predict(X_test)


@timer
def PreprocessData():
    label_name = 'class'
    train_data = pd.read_csv('tekla-C3-JD-24_26-name_to_class.csv')
    train_data.dropna(inplace=True)
    test_data = pd.read_csv('tekla-C3-JD-27-name_to_class-test.csv')
    test_data.dropna(inplace=True)
    # remove class in test_data, to predict class
    train_object_feature = list(train_data.select_dtypes(include=['object']).columns)
    test_object_feature = list(test_data.select_dtypes(include=['object']).columns)

    le = LabelEncoder()
    train_label = le.fit_transform(train_data[label_name])
    train_data.drop(train_object_feature, axis=1, inplace=True)

    test_label = le.fit_transform(test_data[label_name])
    test_data.drop(test_object_feature, axis=1, inplace=True)

    X_train = train_data
    y_train = train_label
    X_test = test_data
    y_test = test_label
    # 数据集切分
    # X_train, y_train, X_test = split_train_test(data, label_name)
    return X_train, y_train, X_test, y_test




def split_train_test(data, label_name):
    train_data = data[data[label_name].notna()]
    test_data = data[data[label_name].isna()]
    X_train = train_data.copy()
    X_train.drop([label_name], axis=1, inplace=True)
    y_label = train_data[label_name]
    return X_train, y_label, test_data


def main():
    X_train, y_train, X_test, y_test = PreprocessData()
    feature_names = X_train.columns.tolist()
    # clf = GBDTModel(n_features=17)
    # clf.train(X_train, y_train)
    # clf.save('GBDT_model.pkl')

    clf = GBDTModel(n_features=17,feature_names=feature_names)
    clf.load('GBDT_model.pkl')
    scores = clf.score_features(X_test,y_test)
    print(scores)


    # y_pred = clf.predict(test_data)
    # nowTime = time.strftime('%Y%m%d-%H%M%S')
    # sub_df = pd.read_csv('./tekla-C3-JD-27-name_to_class.csv')
    # sub_df['result'] = y_pred[:]
    # sub_df.to_csv('./result-' + nowTime + '.csv', index=False)

    # Train2(X_train, y_label)


if __name__ == '__main__':
    main()
