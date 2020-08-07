import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from itertools import combinations, product

# Task
task_name  = 'ComParE2020_Mask'  # os.getcwd().split('/')[-2]
classes    = ['clear', 'mask']

# Enter your team name HERE
team_name = 'baseline'

# Enter your submission number HERE
submission_index = 1

# Option
show_confusion = True   # Display confusion matrix on devel

# Configuration
# feature_set = 'ComParE'  # For all available options, see the dictionary feat_conf
complexities = [1e-5,1e-4,1e-3,1e-2,1e-1,1e0]  # SVM complexities (linear kernel)


# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf_c = {'ComParE':      (6373, 1, ';', 'infer')}
feat_conf_d = {'DeepSpectrum_resnet50': (2048, 1, ',', 'infer')}

feat_conf_b = {'BoAW-125':     ( 250, 1, ';',  None),
               'BoAW-250':     ( 500, 1, ';',  None),
               'BoAW-500':     (1000, 1, ';',  None),
               'BoAW-1000':    (2000, 1, ';',  None),
               'BoAW-2000':    (4000, 1, ';',  None)}

feat_conf_a ={'auDeep-30':    (1024, 2, ',', 'infer'),
              'auDeep-45':    (1024, 2, ',', 'infer'),
              'auDeep-60':    (1024, 2, ',', 'infer'),
              'auDeep-75':    (1024, 2, ',', 'infer'),
              'auDeep-fused': (4096, 2, ',', 'infer')}
fisher_path = './Fisher_vector/'
fisher_option= [32,64,128]

feat_conf_f = {f'fisher_vector_{fisher_option[0]}': (),
               f'fisher_vector_{fisher_option[1]}': (),
               f'fisher_vector_{fisher_option[2]}': ()}
feat_conf = {**feat_conf_a,**feat_conf_b,**feat_conf_c,**feat_conf_d,**feat_conf_f}

# Path of the features and labels
features_path = '../features/'
label_file    = '../lab/labels.csv'
df_labels = pd.read_csv(label_file)
y_train = df_labels['label'][df_labels['file_name'].str.startswith('train')].values
y_devel = df_labels['label'][df_labels['file_name'].str.startswith('devel')].values

# feat_fusion_5 = list(product(feat_conf_a, feat_conf_b, feat_conf_c, feat_conf_d, feat_conf_f))
feat_fusion_4 = list(product(feat_conf_a, feat_conf_b, feat_conf_c, {**feat_conf_d,**feat_conf_f}))+list(product(feat_conf_c,feat_conf_d,feat_conf_f,{**feat_conf_a,**feat_conf_b}))+list(product(feat_conf_a,feat_conf_c,feat_conf_d,feat_conf_f))
# feat_fusion_3 = list(product({**feat_conf_c,**feat_conf_d,**feat_conf_f}, feat_conf_a,feat_conf_b))+list(product({**feat_conf_a,**feat_conf_b,**feat_conf_f}, feat_conf_c, feat_conf_d))+list(product(feat_conf_b,feat_conf_c,feat_conf_f))+list(product(feat_conf_a,feat_conf_d,{**feat_conf_c,**feat_conf_f}))
# feat_fusion_2 = list(product(feat_conf_c,{**feat_conf_a,**feat_conf_b,**feat_conf_d}))

X_train_fusion = []
X_devel_fusion = []

#     X_test       = scaler.transform(X_test)

for x in feat_fusion_4:
    X_train_fused = pd.DataFrame()
    X_devel_fused = pd.DataFrame()
    current_feature = x
    print(f'Preparing for {x}')
#         X_test_fused  = []
    for i in x:
        if 'fisher_vector_' in i:
            option=x.split("fisher_vector_",2)[1]
            X_train = pd.read_csv(fisher_path+f'{x}/train.csv',header=None)
            X_devel = pd.read_csv(fisher_path+f'{x}/devel.csv',header=None)
#             X_test = pd.read_csv(fisher_path+f'{x}/test.csv',header=None)

        else:
            num_feat = feat_conf[i][0]
            ind_off  = feat_conf[i][1]
            sep      = feat_conf[i][2]
            header   = feat_conf[i][3]        
            # Load features and labels
            X_train = pd.read_csv(features_path + task_name + '.' + i + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32)
            X_devel = pd.read_csv(features_path + task_name + '.' + i + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32)
#                 X_test  = pd.read_csv(features_path + task_name + '.' + x + '.test.csv',  sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
        X_train_fused = pd.concat((X_train_fused,X_train),axis=1)
        X_devel_fused = pd.concat((X_devel_fused,X_devel),axis=1)


    # Feature normalisation
    scaler       = MinMaxScaler()
    X_train      = scaler.fit_transform(X_train_fused)
    X_devel      = scaler.transform(X_devel_fused)
    # Train SVM model with different complexities and evaluate
    uar_scores = []
    print(f'current features set is: {feat_fusion_4[i]}')
    for comp in complexities:
        print('\nComplexity {0:.6f}'.format(comp))
        clf = svm.LinearSVC(C=comp, random_state=0, max_iter=100000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_devel)
        uar_scores.append( recall_score(y_devel, y_pred, labels=classes, average='macro') )
        print('UAR on Devel {0:.1f}'.format(uar_scores[-1]*100))
        if show_confusion:
            print('Confusion matrix (Devel):')
            print(classes)
            print(confusion_matrix(y_devel, y_pred, labels=classes))

    # Train SVM model on the whole training data with optimum complexity and get predictions on test data
    optimum_complexity = complexities[np.argmax(uar_scores)]
    print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(optimum_complexity, np.max(uar_scores)*100))
    UAR.extend(np.max(uar_scores)*100)
    

print(UAR)
print(sorted(zip(UAR,feat_fusion_4),reverse=True))