from sklearn.model_selection import StratifiedKFold
import numpy as np
# np.seterr(divide='ignore')
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,auc
from sklearn import metrics
import xlrd
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import warnings
def getTrainFeature(path,i):
    datafeature = xlrd.open_workbook(path)
    trainFeature = []
    test1Feature = []
    test2Feature = []
    # print(i,j)
    for w in range(1,11):
        if w==i:
            path2 = path.replace('1fold', str(w) + "fold")
            datafeature = xlrd.open_workbook(path2)
            tablesfeature = datafeature.sheets()[0]
            for t in range(tablesfeature.nrows):
                test2Feature.append(tablesfeature.row_values(t))
        else:
            path1 = path.replace('1fold',str(w)+"fold")
            datafeature = xlrd.open_workbook(path1)
            tablesfeature = datafeature.sheets()[0]
            for t in range(tablesfeature.nrows):
                trainFeature.append(tablesfeature.row_values(t))

    test2Feature = np.array(test2Feature)
    test2Label = test2Feature[:, 1]
    test2Feature = np.float64(test2Feature[:, 3:])

    trainFeature = np.array(trainFeature)
    trainLabel = trainFeature[:, 1]
    trainFeature = np.float64(trainFeature[:,3:])
    # print(testFeature)
    return trainFeature,test2Feature,trainLabel,test2Label
def svmSingle(pathfeature):
    net_name = pathfeature.split('_')[1]  # netnam
    kf = StratifiedKFold(n_splits=10, shuffle=False,random_state=0)
    Xfeature, valFeature, Xlabel,val_labels  = getTrainFeature(pathfeature, 1)
    c_value_list = [0.0001, 0.001, 0.01, 0.1, 1, 10,100, 500, 600]
    gamma_value_list = [0.001, 0.01, 0.1, 1, 10, 100, 500]
    for feature_num in range(1, 129, 1):  # 0-129 以1为步长
        max_ACC = 0
        max_c_value = 0
        max_SPE = 0
        max_TPR = 0
        max_F1 = 0
        max_gamma_value = 0
        max_auc = 0
        for c_value in c_value_list:
            for gamma_value in gamma_value_list:
                ACC = 0
                TPR = 0
                TNR = 0
                F1_score = 0
                roc_auc=0
                for train_index, test_index in kf.split(Xfeature, Xlabel):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    train_X, train_y = Xfeature[train_index], Xlabel[train_index]
                    test_X, test_y = Xfeature[test_index], Xlabel[test_index]
                    train_features = train_X[:, :feature_num]
                    test1_features = test_X[:, :feature_num]
                    clf = SVC(gamma=gamma_value, C=c_value, probability=True,random_state=0)
                    clf.fit(train_features,train_y)
                    prediction = clf.predict(test1_features)
                    y_true = test_y
                    y_prediction = prediction

                    cnf_matrix = confusion_matrix(y_true, y_prediction)
                    # print(cnf_matrix)
                    fpr, tpr, thresholds = metrics.roc_curve(y_true.ravel(), y_prediction.ravel())
                    roc_auc += auc(fpr, tpr)
                    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
                    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
                    TP = np.diag(cnf_matrix)
                    TN = cnf_matrix.sum() - (FP + FN + TP)
                    FP = FP.astype(float)
                    FN = FN.astype(float)
                    TP = TP.astype(float)
                    TN = TN.astype(float)

                    ACC = ACC + accuracy_score(y_true, y_prediction)

                    # Sensitivity, hit rate, recall, or true positive rate
                    TPR = TPR + metrics.recall_score(y_true, y_prediction, average='macro')
                    # Specificity or true negative rate
                    TNR = TNR + np.mean(TN / (TN + FP))
                    # Precision or positive predictive value
                    F1_score = F1_score + metrics.f1_score(y_true, y_prediction, average='macro')

                ACC /= 10
                TPR /= 10
                F1_score /= 10
                TNR /= 10
                roc_auc/=10
                if ACC > max_ACC:
                    max_ACC = ACC
                    max_c_value = c_value
                    max_gamma_value = gamma_value
                    max_SPE = TNR
                    max_TPR = TPR
                    max_F1 = F1_score
                    max_auc = roc_auc
                print('net_name:', net_name, 'feature_num:', str(feature_num) + ' c_value:', c_value, 'gamma', gamma_value,
                          'ACC:', ACC,
                          'TPR:', TPR, 'SPE', TNR, 'F1_score:', F1_score,'AUC:',roc_auc)
        print("max------------------------------------------------------------------------------------------------")
        print('net_name:', net_name, 'feature_num:', str(feature_num) + ' c_value:', max_c_value, 'gamma',
              max_gamma_value,
              'ACC:', max_ACC, 'TPR:', max_TPR, 'SPE', max_SPE, 'F1_score:', max_F1,'AUC:',max_auc)
        list_2 = [net_name, feature_num, max_c_value, max_gamma_value, max_ACC, max_TPR, max_SPE, max_F1,max_auc]

        df_to_save = pd.DataFrame(np.array(list_2).reshape(1, 9))
        df_to_save.to_csv('../final_val_result_try_traditional1.csv', mode='a+', header=False)
