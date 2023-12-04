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

    file_path = 'dataset/label-v2/xmSS9train-lm61test/'
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

def AP_train(data):
    from sklearn.cluster import AffinityPropagation
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.cluster import KMeans
    has_nan = data.isna().any().any()
    if has_nan:
        print('数据存在NaN')
        return

    # for preference in np.linspace(-50,-30,21):
    #     af_clf = AffinityPropagation(preference=preference)
    #     af_clf.fit(data)
    #     cluster_centers_indices = af_clf.cluster_centers_indices_
    #     labels = af_clf.labels_
    #     n_clusters = len(cluster_centers_indices)
    #     print("preference={}, n_clusters={}".format(preference,n_clusters))



    clf = KMeans(n_clusters=10, random_state=42)
    clf.fit(data)
    labels = clf.labels_

    df_labels = pd.DataFrame(labels)
    df_labels.to_csv('ap_result.csv', index=False)


def main():
    # X_train, y_train, X_test, y_test = PreprocessData()
    # print(label_feature.classes_)
    # Train2(X_train, y_train, X_test, y_test)
    data = PreprocessDataForAP()
    AP_train(data)


if __name__ == '__main__':
    main()