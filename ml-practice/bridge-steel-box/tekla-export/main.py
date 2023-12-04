import pandas as pd
import time
from decorators import timer
import numpy as np
import preprocess.show as show
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@timer
def Train1(input_X_train, input_y_train, input_X_test,
           input_y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_validate, KFold
    from sklearn.tree import DecisionTreeClassifier
    import xgboost
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
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
        'max_depth': 30,
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
        'max_depth': 30,
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

    xgboost_clf.fit(X_train, y_train)
    GBDT_clf.fit(X_train, y_train)
    Tree_clf.fit(X_train, y_train)
    RFC_clf.fit(X_train, y_train)
    model_lgb.fit(X_train, y_train)

    # K折交叉检验
    K_model_list = [Tree_clf, GBDT_clf, xgboost_clf, RFC_clf, model_lgb]
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
    K_result.index = pd.Series(['Tree', 'GBDT', 'XGBoost', 'RFC', 'lgb'])
    print(K_result)


@timer
def Train3(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix
    # 划分数据集为测试集和训练集
    has_nan = X_train.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return
    feature_names = X_test.columns.tolist()
    GBDT_param = {
        'loss': 'log_loss',
        'learning_rate': 0.1,
        'n_estimators': 30,
        'max_depth': 3,
        'min_samples_split': 300
    }
    RFC_clf = RandomForestClassifier()
    RFC_clf.fit(X_train, y_train)
    # GBDT_clf = GradientBoostingClassifier()
    # GBDT_clf.fit(X_train, y_train)

    # 使用测试集来评估模型性能
    y_pred = RFC_clf.predict(X_test)
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
            feature_dict = dict(zip(self.feature_names, scores))
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
    xyz_feature = ['cog_x', 'cog_y', 'cog_z', 'box_minx', 'box_miny', 'box_minz', 'box_maxx', 'box_maxy', 'box_maxz']
    label_name = 'new-class'

    file_path = 'dataset/label-v2/12train-3test/with-xyz/'
    train_data = pd.read_csv(file_path + 'train.csv')
    test_data = pd.read_csv(file_path + 'train1.csv')
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    data = pd.concat([train_data, test_data])
    # data.dropna(inplace=True, subset=label_name)
    data.dropna(inplace=True)

    data_groups = data.groupby('segment_no')

    scaler = MinMaxScaler()
    for segment_no, group in data_groups:
        # 对当前分组中需要进行归一化的字段进行归一化操作
        group[xyz_feature] = scaler.fit_transform(group[xyz_feature])
        # 将归一化后的分组数据更新回原始数据中
        data.update(group)

    le = LabelEncoder()
    label = le.fit_transform(data[label_name])
    object_feature = list(data.select_dtypes(include=['object']).columns)
    data.drop(object_feature, axis=1, inplace=True)
    data[label_name] = label

    train_data = data[data['is_train'] == 1].reset_index(drop=True)
    train_data.drop('is_train', axis=1, inplace=True)
    test_data = data[data['is_train'] == 0].reset_index(drop=True)
    test_data.drop('is_train', axis=1, inplace=True)

    y_train = train_data[label_name].values
    train_data.drop(label_name, axis=1, inplace=True)
    y_test = test_data[label_name].values
    test_data.drop(label_name, axis=1, inplace=True)
    return train_data, y_train, test_data, y_test


def split_train_test(data, label_name):
    train_data = data[data[label_name].notna()]
    test_data = data[data[label_name].isna()]
    X_train = train_data.copy()
    X_train.drop([label_name], axis=1, inplace=True)
    y_label = train_data[label_name]
    return X_train, y_label, test_data


def main():
    # Preprocess1()
    X_train, y_train, X_test, y_test = PreprocessData()

    # 就利用GBDT模型进行训练，并取出特征名称
    feature_names = X_train.columns.tolist()
    # clf = GBDTModel(n_features=8, feature_names=feature_names)
    # clf.train(X_train, y_train)

    # 将训练出的模型，保存在本地
    # clf.save('GBDT_model.pkl')
    # clf = GBDTModel(n_features=17,feature_names=feature_names)
    # clf.load('GBDT_model.pkl')

    # 输出每项特征所占权重
    # scores = clf.score_features(X_test,y_test)

    # y_pred = clf.predict(test_data)
    # nowTime = time.strftime('%Y%m%d-%H%M%S')
    # sub_df = pd.read_csv('./tekla-C3-JD-27-name_to_class.csv')
    # sub_df['result'] = y_pred[:]
    # sub_df.to_csv('./result-' + nowTime + '.csv', index=False)

    Train3(X_train, y_train,X_test,y_test)


def testModel():
    X_train, y_train, X_test, y_test = PreprocessData()
    Train1(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
