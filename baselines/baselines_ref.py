import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import pandas as pd
import random
from sklearn import model_selection, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier

balanced = False
bootstrap = False

#few shot learning(??)

file_path = '..\\data\\tox21.csv'
df1 = pd.read_csv(file_path)
targets = df1.columns[0:12]


for idx, elem in enumerate(targets):
    df = df1
    target_name = targets[idx]
    print(idx, elem, target_name)

    df = df[[target_name, 'smiles']].dropna()


    """     # balance the dataset
    if bootstrap:
        pos_rows = df[df[target_name] == 1]
        concat_amount = 1500 - len(pos_rows)
        pos_rows_concat = pos_rows.sample(n=concat_amount, replace=True)
        df = pd.concat([df, pos_rows_concat], ignore_index=True)
    print(len(df))
    
    pos_num = df[df[target_name] == 1].shape[0]
    neg_num = df[df[target_name] == 0].shape[0]
    print(pos_num, neg_num)

    if balanced:
        neg_rows = df[df[target_name] == 0]
        drop_rows = neg_rows.sample(n=neg_num-pos_num).index
        df = df.drop(drop_rows) """

    

    num_bits = 1024
    num_rows = len(df)

    fp_array = np.zeros((num_rows, num_bits))
    target_array = np.zeros((num_rows, 1))
    i = 0
    morgan_fp_gen = GetMorganGenerator(radius=2,fpSize=num_bits)

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])

        if mol is not None:
            morgan_fp = morgan_fp_gen.GetFingerprint(mol)
            fp_array[i] = np.array(morgan_fp)
            target_array[i] = row[target_name]
            i += 1
    target_array = target_array.ravel()
    print("fingerprints:", fp_array.shape)
    print("targets", target_array.shape)

    #Metrics ---- roc  acc f1  prs sns sps
    svm_metrics = [[], [], [], [], [], []]
    rf_metrics  = [[], [], [], [], [], []]
    knn_metrics = [[], [], [], [], [], []]
    xgb_metrics = [[], [], [], [], [], []]
    mlp_metrics = [[], [], [], [], [], []]

    knn_param_dist = {
            'n_neighbors': range(1, 20),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    knn_best_params = []
    
    print("Iterations:")
    for i in range(2, 32):
        print(str(i - 1) + "/30")
        seed = i - 1
        random.seed(seed)
        
        Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(fp_array,target_array, test_size=0.3, shuffle=True, random_state=seed)

        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Train_Y), y=Train_Y)
        #class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
        class_weights_dict = "balanced"


        #################### SVM ####################

        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight=class_weights_dict, random_state=seed)

        SVM.fit(Train_X,Train_Y)

        predictions_SVM = SVM.predict(Test_X)

        svm_metrics[0].append(roc_auc_score(Test_Y, predictions_SVM))
        svm_metrics[1].append(accuracy_score(Test_Y, predictions_SVM))
        svm_metrics[2].append(f1_score(Test_Y, predictions_SVM))
        svm_metrics[3].append(precision_score(Test_Y, predictions_SVM))
        svm_metrics[4].append(recall_score(Test_Y, predictions_SVM))
        tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_SVM).ravel()
        sp = tn/(tn+fp)
        svm_metrics[5].append(sp)

        #################### RF ####################

        RF = RandomForestClassifier(max_depth=10, n_estimators=100, class_weight=class_weights_dict, random_state=seed)

        RF.fit(Train_X,Train_Y)

        predictions_RF = RF.predict(Test_X)

        rf_metrics[0].append(roc_auc_score(Test_Y, predictions_RF))
        rf_metrics[1].append(accuracy_score(Test_Y, predictions_RF))
        rf_metrics[2].append(f1_score(Test_Y, predictions_RF))
        rf_metrics[3].append(precision_score(Test_Y, predictions_RF))
        rf_metrics[4].append(recall_score(Test_Y, predictions_RF))
        tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_RF).ravel()
        sp = tn/(tn+fp)
        rf_metrics[5].append(sp)

        #################### XGB ####################
        #pos_weight = sum(Train_Y == 0) / sum(Train_Y == 1)
        pos_weight=1
        XGB = XGBClassifier(objective="binary:logistic",learning_rate=0.1,max_depth=6,n_estimators=100,scale_pos_weight=pos_weight)

        XGB.fit(Train_X,Train_Y)

        predictions_XGB = XGB.predict(Test_X)

        xgb_metrics[0].append(roc_auc_score(Test_Y, predictions_XGB))
        xgb_metrics[1].append(accuracy_score(Test_Y, predictions_XGB))
        xgb_metrics[2].append(f1_score(Test_Y, predictions_XGB))
        xgb_metrics[3].append(precision_score(Test_Y, predictions_XGB))
        xgb_metrics[4].append(recall_score(Test_Y, predictions_XGB))
        tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_XGB).ravel()
        sp = tn/(tn+fp)
        xgb_metrics[5].append(sp)

        #################### KNN ####################

        # Randomized search for knn
        
        if i == 2:
            KNN = KNeighborsClassifier()

            #Randomized Search
            random_search = RandomizedSearchCV(KNN, knn_param_dist, n_iter=20, cv=5, scoring='roc_auc', random_state=42)
            random_search.fit(Train_X, Train_Y)

            print("KNN Best Parameters:", random_search.best_params_)
            print("KNN Best Score:", random_search.best_score_)

            knn_best_params = random_search.best_params_

        else:
            KNN = KNeighborsClassifier(
                n_neighbors=knn_best_params['n_neighbors'],
                weights=knn_best_params['weights'],
                metric=knn_best_params['metric'])


        KNN.fit(Train_X,Train_Y)

        predictions_KNN = KNN.predict(Test_X)

        knn_metrics[0].append(roc_auc_score(Test_Y, predictions_KNN))
        knn_metrics[1].append(accuracy_score(Test_Y, predictions_KNN))
        knn_metrics[2].append(f1_score(Test_Y, predictions_KNN))
        knn_metrics[3].append(precision_score(Test_Y, predictions_KNN))
        knn_metrics[4].append(recall_score(Test_Y, predictions_KNN))
        tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_KNN).ravel()
        sp = tn/(tn+fp)
        knn_metrics[5].append(sp)

        #################### MLP ####################

        #sample_weight = np.array([class_weights[cls] for cls in Train_Y])
        #sample_weight = None
        MLP =  MLPClassifier(hidden_layer_sizes=(num_bits), activation='relu', solver='adam', max_iter=200)
        MLP.fit(Train_X, Train_Y)
        predictions_MLP = MLP.predict(Test_X)

        mlp_metrics[0].append(roc_auc_score(Test_Y, predictions_MLP))
        mlp_metrics[1].append(accuracy_score(Test_Y, predictions_MLP))
        mlp_metrics[2].append(f1_score(Test_Y, predictions_MLP))
        mlp_metrics[3].append(precision_score(Test_Y, predictions_MLP))
        mlp_metrics[4].append(recall_score(Test_Y, predictions_MLP))
        tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_MLP).ravel()
        sp = tn/(tn+fp)
        mlp_metrics[5].append(sp)

    metrics = [svm_metrics, rf_metrics, xgb_metrics, knn_metrics, mlp_metrics]
    metrics_np = np.zeros((len(metrics), 12))
    
    for i, clf in enumerate(metrics):
        metrics_np[i, 0::2] = np.round([np.mean(metric) for metric in clf], 3)
        metrics_np[i, 1::2] = np.round([np.std(metric) for metric in clf], 3)    

    
    metric_names = ['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Sensitivity', 'Specificity']

    columns = []
    clfs = ["SVM", "RF","XGB", "KNN", "MLP"]
    for name in metric_names:
        columns.extend([f'Mean {name}', f'Std {name}'])

    df_clfs = pd.DataFrame(clfs, columns=["Classifier"])
    df_metrics = pd.DataFrame(metrics_np, columns=columns)
    df = pd.concat([df_clfs, df_metrics], axis=1)

    filename = f"results\\updated\\unweighted\\ml_baselines_tox21_morgan_{num_bits}_{elem}.csv"
    df.to_csv(filename, index=False)
    print(filename)


#ADD MLP - multilayer perceptron