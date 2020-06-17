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
def getArray(list):
    # print(list)
    x = [[0] * 9] * len(list)
    y = np.array(x)
    for i in range(len(list)):
        for j in range(len(list[i])):
            y[i][int(list[i][j])]=1
    return y
def getPrediction(probability,T):
    prediction = []
    for i in range(len(probability)):
        label = []
        max_probability = np.max(probability[i])
        # label.append(np.argmax(probability[i]))
        for j in range(len(probability[i])):
            if (max_probability-probability[i][j])<T:
                label.append(j)
        prediction.append(label)
    return prediction
def getLabel(sLabel):
    labels = []
    for i in range(len(sLabel)):
        label = []
        for j in range(len(sLabel[i])):
            label.append((sLabel[i])[j:j+1])
        labels.append(label)
    return labels
def Accuracy(y_true,y_pred,classes):
    result = 0.0
    for i in range(len(y_true)):
        point = 0.0
        up = 0.0
        down = 0.0
        for j in range(classes):
            if y_true[i][j]==1 and y_pred[i][j]==1:
                up = up+1

            if y_true[i][j] == 1 or y_pred[i][j] == 1:
                down = down+1

        point = up/down
        result  = result + point
    return result/len(y_true)
    # print(result/len(y_true))
def Average_Label_Accuracy(y_true,y_pred,classes):
    Average_Label_Accuracy = 0.0
    for i in range(classes):
        Label_Accuracy = 0.0
        for j in range(len(y_true)):
            if y_true[j][i] == y_pred[j][i]:
                Label_Accuracy = Label_Accuracy+1
        # print(Label_Accuracy / (len(y_true)))
        # print("----")
        Average_Label_Accuracy = Average_Label_Accuracy+(Label_Accuracy/len(y_true))
    return Average_Label_Accuracy/classes
    # print(Average_Label_Accuracy/classes)
def svmMulti(pathfeature):
    net_name = pathfeature.split('_')[1]  # netnam
    kf = StratifiedKFold(n_splits=10, shuffle=False, random_state=0)
    Xfeature, valFeature, Xlabel, val_labels = getTrainFeature(pathfeature, 1)
    # c_value_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30, 50, 100, 150, 200, 300, 400, 500, 600]
    # c_value_list = [0.01, 0.1, 1, 10, 20, 30, 50, 100, 150, 200, 300, 400, 500, 600]
    c_value_list = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 600]
    gamma_value_list = [0.001, 0.01, 0.1, 1, 10, 100, 500]
    for feature_num in range(1,129, 1):  # 0-129 以1为步长
        max_ACC = 0
        max_c_value = 0
        max_SPE = 0
        max_TPR = 0
        max_F1 = 0
        max_T = 0
        max_gamma_value = 0
        max_subAcc = 0
        max_PRE = 0
        max_labelAcc = 0
        max_auc = 0
        for c_value in c_value_list:
            for gamma_value in gamma_value_list:
                for T in range(10,100, 5):
                    SubACC = 0
                    ACC = 0
                    LabelAcc = 0
                    PRE = 0
                    TPR = 0
                    TNR = 0
                    F1_score = 0
                    roc_auc = 0
                    # print(float(T) / 100.0)
                    # print("***********************************T*******************************************")
                    for train_index, test_index in kf.split(Xfeature, Xlabel):
                        # print("TRAIN:", train_index, "TEST:", test_index)
                        train_X, train_y = Xfeature[train_index], Xlabel[train_index]
                        test_X, test_y = Xfeature[test_index], Xlabel[test_index]
                        train_features = train_X[:, :feature_num]
                        test1_features = test_X[:, :feature_num]
                        clf =  OneVsRestClassifier(SVC(gamma=gamma_value, C=c_value, probability=True, random_state=0))
                        train_yy = MultiLabelBinarizer().fit_transform(getLabel(train_y))
                        # labelx = csr_matrix(train_yy)
                        # featurex = csr_matrix(train_features)
                        # print(train_yy)
                        clf.fit(train_features, train_yy)
                        probability = clf.predict_proba(test1_features)
                        prediction = getPrediction(probability, float(T) / 100.0)

                        y_true = np.ndarray(shape=(len(getLabel(test_y)), 9), dtype=int,
                                            buffer=np.array(getArray(getLabel(test_y))),
                                            offset=0, order="C")
                        y_prediction = np.ndarray(shape=(len(prediction), 9), dtype=int,
                                                  buffer=np.array(getArray(prediction)),
                                                  offset=0, order="C")
                        # result = np.c_[np.c_[test_y, prediction], probability]
                        # df_to_save = pd.DataFrame(result)
                        # df_to_save.to_csv('E:/hpaData/ProcessMultiLabel/feature/probability1.csv', mode='a+',
                        #                   header=False)
                        cnf_matrix = multilabel_confusion_matrix(y_true, y_prediction)
                        # print(cnf_matrix)
                        fpr, tpr, thresholds = metrics.roc_curve(y_true.ravel(), y_prediction.ravel())
                        roc_auc += auc(fpr, tpr)
                        TN = cnf_matrix[:, 0, 0]
                        TP = cnf_matrix[:, 1, 1]
                        FN = cnf_matrix[:, 1, 0]
                        FP = cnf_matrix[:, 0, 1]
                        FP = FP.astype(float)
                        FN = FN.astype(float)
                        TP = TP.astype(float)
                        TN = TN.astype(float)
                        #Subset Accuracy
                        SubACC = SubACC + accuracy_score(y_true, y_prediction)
                        # Sensitivity, hit rate, recall, or true positive rate
                        TPR = TPR + metrics.recall_score(y_true, y_prediction, average='macro')
                        # Specificity or true negative rate
                        TNR = TNR + np.mean(TN / (TN + FP))
                        # Precision or positive predictive value
                        PRE = PRE+precision_score(y_true, y_prediction, average='macro')
                        #F1_score
                        F1_score = F1_score + metrics.f1_score(y_true, y_prediction, average='macro')
                        #Accuracy
                        ACC = ACC+Accuracy(y_true, y_prediction, 9)
                        #Average_Label_Accuracy
                        LabelAcc = LabelAcc+Average_Label_Accuracy(y_true, y_prediction, 9)
                    ACC /= 10
                    TPR /= 10
                    F1_score /= 10
                    TNR /= 10
                    SubACC/=10
                    LabelAcc/=10
                    PRE/=10
                    roc_auc/=10
                    if SubACC > max_subAcc:
                        max_ACC = ACC
                        max_c_value = c_value
                        max_gamma_value = gamma_value
                        max_SPE = TNR
                        max_TPR = TPR
                        max_F1 = F1_score
                        max_T = T
                        max_subAcc = SubACC
                        max_labelAcc = LabelAcc
                        max_PRE = PRE
                        max_auc = roc_auc
                    print('net_name:', net_name, 'T',T,'feature_num:', str(feature_num) + ' c_value:', c_value, 'gamma',
                              gamma_value,'SUB_ACC',SubACC,'ACC:', ACC,'Label_ACC',LabelAcc,'PRE',PRE,
                              'TPR:', TPR, 'SPE', TNR, 'F1_score:', F1_score,'auc:',roc_auc)
        print("max------------------------------------------------------------------------------------------------")
        print('net_name:', net_name,'T',max_T, 'feature_num:', str(feature_num) + ' c_value:', max_c_value, 'gamma',
              max_gamma_value, 'SUB_ACC',max_subAcc,'ACC:', max_ACC,'Label_ACC',max_labelAcc,'PRE',max_PRE,
             'TPR:', max_TPR, 'SPE', max_SPE, 'F1_score:', max_F1,'auc',max_auc)
        list_2 = [net_name, feature_num,max_T, max_c_value, max_gamma_value, max_ACC,max_subAcc,max_labelAcc,max_PRE, max_TPR, max_SPE, max_F1,max_auc]

        df_to_save = pd.DataFrame(np.array(list_2).reshape(1, 13))
        df_to_save.to_csv('../xception/result.csv', mode='a+',
                          header=False)

