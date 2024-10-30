import os
import sys
import numpy as np
import random
from rdkit import Chem
from sklearn import model_selection, svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()   

def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
    
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res

def baselines(data):

    root_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(root_dir)
    file_path = os.path.join(root_dir, 'data', '{}.csv'.format(data))
    print(file_path)
    file = open(file_path, 'r').readlines()[1:]
    #np.random.shuffle(file)
    
    T = {}
  
    s = 0
           
    if data == "tox21":
        #maximum common substructure??
        mcss_SMILES = []
        mcss_label = []
        for _,j in enumerate(file):
          sample = j.split(",")
          sample = [-1 if item.strip() == '' else item for item in sample]
          #print(sample)
          smile = sample[13]
          
          m = Chem.MolFromSmiles(smile)
          if m is None:
            print('invalid')
            #print(sample)

          for i in range(11):
                if i not in T and sample[i] != -1:
                    T[i] = [[],[]]
                if sample[i] == '0':
                    T[i][0].append(smile)
                elif sample[i] == '1':
                    T[i][1].append(smile)

                if (sample[i] == '0' or sample[i] == '1') and i == 0:
                    mcss_SMILES.append(smile)   
                    mcss_label.append(sample[i])
        
               
        smiles_data_neg =  T[0][0]
        smiles_data_pos =  T[0][1]        
        smiles_neg = []
        smiles_neg_label = []
        smiles_pos=[]
        smiles_pos_label = []
        
        
        #descriptors_list = [x[0] for x in Descriptors._descList]
        #print(descriptors_list)
        
        for i in smiles_data_neg:
            
            #Molecular Graph
            mol = Chem.MolFromSmiles(i)
           
            #smiles_neg.append(np.array(RDKFingerprint(mol))) #RDKit Fingerprints
            smiles_neg.append((np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)))) #Morgan Fingerprints
            smiles_neg_label.append(0)
        for i in smiles_data_pos:
            #Molecular Graph
            mol = Chem.MolFromSmiles(i)
            #smiles_pos.append(np.array(RDKFingerprint(mol))) #RDKit Fingerprints
            smiles_pos.append((np.array(rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2)))) #Morgan Fingerprints
            smiles_pos_label.append(1)
                
        all_smiles = smiles_neg + smiles_pos
        all_labels = smiles_neg_label + smiles_pos_label
        
        #print(all_smiles)
        #print(all_labels)
        roc_SVM = []
        acc_SVM = []
        f1s_SVM = []
        prs_SVM = []
        sns_SVM = []
        sps_SVM = []
        
        roc_RF = []
        acc_RF = []
        f1s_RF = []
        prs_RF = []
        sns_RF = []
        sps_RF = []
        
        roc_KNN = []
        acc_KNN = []
        f1s_KNN = []
        prs_KNN = []
        sns_KNN = []
        sps_KNN = []
        
        roc_GP = []
        acc_GP = []
        f1s_GP = []
        prs_GP = []
        sns_GP = []
        sps_GP = []
        
        for i in range(2, 32):
            print("ITERATION:", i)
            seed = i
            random.seed(seed)
           
            Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(all_smiles,all_labels, test_size=0.3, shuffle=True, random_state=seed)
            #################### SVM ####################
                        
            SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', class_weight= 'balanced', random_state=seed)
            
            SVM.fit(Train_X,Train_Y)
            
            predictions_SVM = SVM.predict(Test_X)
            
            roc_SVM.append(roc_auc_score(Test_Y, predictions_SVM))
            acc_SVM.append(accuracy_score(Test_Y, predictions_SVM))
            f1s_SVM.append(f1_score(Test_Y, predictions_SVM))
            prs_SVM.append(precision_score(Test_Y, predictions_SVM))
            sns_SVM.append(recall_score(Test_Y, predictions_SVM))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_SVM).ravel()
            sp = tn/(tn+fp)
            sps_SVM.append(sp)
            
            #################### RF ####################
            
            RF = RandomForestClassifier(max_depth=10, n_estimators=100, class_weight='balanced', random_state=seed)
            
            RF.fit(Train_X,Train_Y)
            
            predictions_RF = RF.predict(Test_X)
            
            roc_RF.append(roc_auc_score(Test_Y, predictions_RF))
            acc_RF.append(accuracy_score(Test_Y, predictions_RF))
            f1s_RF.append(f1_score(Test_Y, predictions_RF))
            prs_RF.append(precision_score(Test_Y, predictions_RF))
            sns_RF.append(recall_score(Test_Y, predictions_RF))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_RF).ravel()
            sp = tn/(tn+fp)
            sps_RF.append(sp)
            
            #################### GP ####################
            
            GP = GaussianProcessClassifier(kernel = 2.0 * RBF(1.0), max_iter_predict = 100, random_state=seed)
            
            GP.fit(Train_X,Train_Y)
            
            predictions_GP = GP.predict(Test_X)
            
            roc_GP.append(roc_auc_score(Test_Y, predictions_GP))
            acc_GP.append(accuracy_score(Test_Y, predictions_GP))
            f1s_GP.append(f1_score(Test_Y, predictions_GP))
            prs_GP.append(precision_score(Test_Y, predictions_GP))
            sns_GP.append(recall_score(Test_Y, predictions_GP))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_GP).ravel()
            sp = tn/(tn+fp)
            sps_GP.append(sp)
                        
            #################### KNN ####################
            
            KNN = KNeighborsClassifier(n_neighbors=3)
            
            KNN.fit(Train_X,Train_Y)
            
            predictions_KNN = KNN.predict(Test_X)
            
            roc_KNN.append(roc_auc_score(Test_Y, predictions_KNN))
            acc_KNN.append(accuracy_score(Test_Y, predictions_KNN))
            f1s_KNN.append(f1_score(Test_Y, predictions_KNN))
            prs_KNN.append(precision_score(Test_Y, predictions_KNN))
            sns_KNN.append(recall_score(Test_Y, predictions_KNN))
            tn, fp, fn, tp = confusion_matrix(Test_Y, predictions_KNN).ravel()
            sp = tn/(tn+fp)
            sps_KNN.append(sp)
            
        filename = "results-exp/ml-baselines_muta_morgan.txt"
        file = open(filename, "a")
                                   
        file.write("\nMean SVM Accuracy Score -> " + str(np.mean(acc_SVM)) + "+-"+ str(np.std(acc_SVM)))
        file.write("\nMean SVM ROC-AUC Score -> "+ str(np.mean(roc_SVM)) + "+-"+ str(np.std(roc_SVM)))
        file.write("\nMean SVM F1-Score -> "+ str(np.mean(f1s_SVM))+ "+-" + str(np.std(f1s_SVM)))
        file.write("\nMean SVM Precision Score -> "+ str(np.mean(prs_SVM))+ "+-"+ str(np.std(prs_SVM)))
        file.write("\nMean SVM Sensitivity Score -> "+ str(np.mean(sns_SVM))+ "+-"+ str(np.std(sns_SVM)))
        file.write("\nMean SVM Specificity Score -> "+ str(np.mean(sps_SVM))+ "+-"+ str(np.std(sps_SVM)))
        file.write("\n")
        file.write("\nMean RF Accuracy Score -> "+ str(np.mean(acc_RF))+ "+-"+ str(np.std(acc_RF)))
        file.write("\nMean RF ROC-AUC Score -> "+ str(np.mean(roc_RF))+ "+-"+ str(np.std(roc_RF)))
        file.write("\nMean RF F1-Score -> "+ str(np.mean(f1s_RF))+ "+-"+ str(np.std(f1s_RF)))
        file.write("\nMean RF Precision Score -> "+ str(np.mean(prs_RF))+ "+-"+ str(np.std(prs_RF)))
        file.write("\nMean RF Sensitivity Score -> "+ str(np.mean(sns_RF))+ "+-"+ str(np.std(sns_RF)))
        file.write("\nMean RF Specificity Score -> "+ str(np.mean(sps_RF))+ "+-"+ str(np.std(sps_RF)))
        file.write("\n")
        file.write("\nMean GP Accuracy Score -> "+ str(np.mean(acc_GP))+ "+-"+ str(np.std(acc_GP)))
        file.write("\nMean GP ROC-AUC Score -> "+ str(np.mean(roc_GP))+ "+-"+ str(np.std(roc_GP)))
        file.write("\nMean GP F1-Score -> "+ str(np.mean(f1s_GP))+ "+-"+ str(np.std(f1s_GP)))
        file.write("\nMean GP Precision Score -> "+ str(np.mean(prs_GP))+ "+-"+ str(np.std(prs_GP)))
        file.write("\nMean GP Sensitivity Score -> "+ str(np.mean(sns_GP))+ "+-"+ str(np.std(sns_GP)))
        file.write("\nMean GP Specificity Score -> "+ str(np.mean(sps_GP))+ "+-"+ str(np.std(sps_GP)))
        file.write("\n")
        file.write("\nMean KNN Accuracy Score -> "+ str(np.mean(acc_KNN))+ "+-"+ str(np.std(acc_KNN)))
        file.write("\nMean KNN ROC-AUC Score -> "+ str(np.mean(roc_KNN))+ "+-"+ str(np.std(roc_KNN)))
        file.write("\nMean KNN F1-Score -> "+ str(np.mean(f1s_KNN))+ "+-"+ str(np.std(f1s_KNN)))
        file.write("\nMean KNN Precision Score -> "+ str(np.mean(prs_KNN))+ "+-"+ str(np.std(prs_KNN)))
        file.write("\nMean KNN Sensitivity Score -> "+ str(np.mean(sns_KNN))+ "+-"+ str(np.std(sns_KNN)))
        file.write("\nMean KNN Specificity Score -> "+ str(np.mean(sps_KNN))+ "+-"+ str(np.std(sps_KNN)))
        
        
if __name__ == "__main__":
    # compute baseline results - SVM, RF, GP, KNN
    baselines("tox21")
