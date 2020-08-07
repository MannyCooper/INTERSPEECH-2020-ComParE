#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
import os.path
import sys

classes = ['L', 'M', 'H']

# Task
task_name = 'ComParE2020_USOMS-e'
# Enter your team name HERE
team_name = 'baseline'
# Enter your submission number HERE
submission_index = 1
# Option
show_confusion = True  # Display confusion matrix on devel
majority_vote_story_id = True
# Configuration
feature_set_V = ['BoW','TF_IDF','frozen-bert-gmax','frozen-bert-rnnatt','frozen-bert-pos-fuse-rnnatt','fused','PMI_base_1500','NGD_base_1800','PMI_base_BoW_1500','PMI_base_TF_IDF_1500','NGD_base_BoW_1800','NGD_base_TF_IDF_800']  # For all available options, see the dictionary feat_conf
feature_set_A = ['PMI_base_800','NGD_base_800','TF_IDF','BoW_3','frozen-bert-gmax','frozen-bert-rnnatt','frozen-bert-pos-fuse-rnnatt','fused','PMI_base_BoW_800','PMI_base_TF_IDF_800','NGD_base_BoW_800','NGD_base_TF_IDF_800']
label_options = ['V_cat','A_cat']

complexities = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
list_V= []
list_A= []
list_V_value= []
list_A_value= []
# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'sf_PMI_base_TF_IDF_800' :(15,1,',','infer'),
             'mean_PMI_base_BoW_800' :(3,1,',','infer'),
            'frozen-bert-gmax': (512, 1, ',', 'infer'),
             'frozen-bert-rnnatt': (512, 1, ',', 'infer'),
             'frozen-bert-pos-fuse-rnnatt': (512, 1, ',', 'infer'),
             'fused': (1536, 1, ',', 'infer'),
             'PMI_L': (2838, 1, ',', 'infer'),
             'PMI_M': (2838, 1, ',', 'infer'),
             'PMI_H': (2838, 1, ',', 'infer'),
             'PMI_Merge': (2838*3, 1, ',', 'infer'),
             'PMI_Real_BoW': (2838, 1, ',', 'infer'),
             'PMI_Real_TF_IDF': (2838, 1, ',', 'infer'),
             'PMI_600': (1800, 1, ',', 'infer'),
             'PMI_800': (2400, 1, ',', 'infer'),
             'PMI_1000': (3000, 1, ',', 'infer'),
             'PMI_1200': (3600, 1, ',', 'infer'),
             'PMI_1500': (4500, 1, ',', 'infer'),
             'PMI_1800': (5400, 1, ',', 'infer'),
             'PMI_2000': (6000, 1, ',', 'infer'),
             'PMI_2200': (6600, 1, ',', 'infer'),
             'PMI_2500': (7500, 1, ',', 'infer'),
             'PMI_TF_IDF1500': (2838, 1, ',', 'infer'),
             'PMI_TF_IDF_1500': (4500, 1, ',', 'infer'),
             'NGD_TF_IDF_1500': (4500, 1, ',', 'infer'),
             'NGD_without_log_TF_IDF_1500': (4500, 1, ',', 'infer'),
             'NGD_with_log_TF_IDF_1200': (3600, 1, ',', 'infer'),
             'PMI_without_log_method_600': (1800, 1, ',', 'infer'),
             'PMI_without_log_method_800': (2400, 1, ',', 'infer'),
             'PMI_without_log_method_1000': (3000, 1, ',', 'infer'),
             'PMI_without_log_method_1200': (3600, 1, ',', 'infer'),
             'PMI_without_log_method_1500': (4500, 1, ',', 'infer'),
             'PMI_without_log_method_1800': (5400, 1, ',', 'infer'),
             'PMI_without_log_method_2000': (6000, 1, ',', 'infer'),
             'PMI_with_log_method1500': (4500, 1, ',', 'infer'),
             'NGD_with_log_method_1200': (3600, 1, ',', 'infer'),
             'NGD_with_log_method_1000': (3000, 1, ',', 'infer'),
             'NGD_with_log_method_800': (2400, 1, ',', 'infer'),
             'BoW_PCA' : (192,0,',',None),
             'BoW' : (2836,0,',','infer'),
             'BoW_3' : (2836,0,',','infer'),
             'TF_IDF' : (2836,0,',','infer'),
             'PMI_base_1800' :(5400,0,',','infer'),
             'PMI_base_1500' :(4500,0,',','infer'),
             'PMI_base_1200' :(3600,0,',','infer'),
             'PMI_base_1000' :(3000,0,',','infer'),
             'PMI_base_800' :(2400,0,',','infer'),
             'PMI_base_600' :(1200,0,',','infer'),
             'PMI_base_TF_IDF_1800' :(5400,0,',','infer'),
             'PMI_base_TF_IDF_1500' :(4500,0,',','infer'),
             'PMI_base_TF_IDF_1200' :(3600,0,',','infer'),
             'PMI_base_TF_IDF_1000' :(3000,0,',','infer'),
             'PMI_base_TF_IDF_800' :(2400,0,',','infer'),
             'PMI_base_TF_IDF_600' :(1200,0,',','infer'),
             'PMI_base_BoW_1800' :(5400,0,',','infer'),
             'PMI_base_BoW_1500' :(4500,0,',','infer'),
             'PMI_base_BoW_1200' :(3600,0,',','infer'),
             'PMI_base_BoW_1000' :(3000,0,',','infer'),
             'PMI_base_BoW_800' :(2400,0,',','infer'),
             'PMI_base_BoW_600' :(1200,0,',','infer'),
             'NGD_base_1800' :(5400,0,',','infer'),
             'NGD_base_1500' :(4500,0,',','infer'),
             'NGD_base_1200' :(3600,0,',','infer'),
             'NGD_base_1000' :(3000,0,',','infer'),
             'NGD_base_800' :(2400,0,',','infer'),
             'NGD_base_600' :(1200,0,',','infer'),
             'NGD_base_TF_IDF_1800' :(5400,0,',','infer'),
             'NGD_base_TF_IDF_1500' :(4500,0,',','infer'),
             'NGD_base_TF_IDF_1200' :(3600,0,',','infer'),
             'NGD_base_TF_IDF_1000' :(3000,0,',','infer'),
             'NGD_base_TF_IDF_800' :(2400,0,',','infer'),
             'NGD_base_TF_IDF_600' :(1200,0,',','infer'),
             'NGD_base_BoW_1800' :(5400,0,',','infer'),
             'NGD_base_BoW_1500' :(4500,0,',','infer'),
             'NGD_base_BoW_1200' :(3600,0,',','infer'),
             'NGD_base_BoW_1000' :(3000,0,',','infer'),
             'NGD_base_BoW_800' :(2400,0,',','infer'),
             'NGD_base_BoW_600' :(1200,0,',','infer'),
             'PMI_his' : (114,0,',','infer'),
             'NGD_his' : (114,0,',','infer')
             }


# Path of the features and labels
features_path = '../features/'
label_file = '../lab/labels.csv'

# Start
temp = np.arange(0).reshape(87,0)
from itertools import combinations

for current_label in label_options:
    if current_label == 'V_cat':
        feature_set = feature_set_V
    else:
        feature_set = feature_set_A
    from itertools import combinations
    nums = []
    for lens in range(0, len(feature_set)):
        nums.extend(list(combinations(feature_set, lens)))
    nums = nums[1:]
    print(len(nums))
    for feature_set in nums:
        feature_set_list  = '+'.join(feature_set)
        # print(feature_set_list)
        print('\nRunning ' + task_name + ' ' + feature_set_list + ' ' +
              current_label + ' baseline ... (this might take a while) \n')
        # Load features and labels
        print('\nLoading Features and Labels')
        for features in feature_set:
            num_feat = feat_conf[features][0]
            ind_off = feat_conf[features][1]
            sep = feat_conf[features][2]
            header = feat_conf[features][3]

            X_train_temp = pd.read_csv(features_path + task_name + '.' + features + '.' + current_label + '_no.train.csv', sep=sep, header=header,
                                  usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
            X_devel = pd.read_csv(features_path + task_name + '.' + features + '.' + current_label + '_no.devel.csv', sep=sep, header=header,
                                  usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
            X_test = pd.read_csv(features_path + task_name + '.' + features + '.' + current_label + '_no.test.csv', sep=sep, header=header,
                                 usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
            X_train = np.hstack((temp,X_train_temp))
            print(len(X_train[0]))

        # X_train = pd.read_csv(features_path + task_name + '.' + feature_set +'.train.csv',
        #                       sep=sep, header=header,
        #                       usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
        # X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv',
        #                       sep=sep, header=header,
        #                       usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values
        # X_test = pd.read_csv(features_path + task_name + '.' + feature_set + '.test.csv', sep=sep,
        #                      he ader=header,
        #                      usecols=range(ind_off, num_feat + ind_off), dtype=np.float32).values

        df_labels = pd.read_csv(label_file)
        df_agg = df_labels.groupby(['filename_text', 'partition'])[current_label].agg(
            lambda x: x.value_counts().index[0]).reset_index()

        print('currently running on dev for ' + current_label)
        y_train = df_agg[current_label][df_agg['partition'] == 'train'].values
        y_devel = df_agg[current_label][df_agg['partition'] == 'devel'].values

        # Concatenate training and development for final training
        X_traindevel = np.concatenate((X_train, X_devel))
        y_traindevel = np.concatenate((y_train, y_devel))

        # Upsampling / Balancing
        print('Upsampling ... ')
        num_samples_train = []
        num_samples_traindevel = []
        for label in classes:
            num_samples_train.append(len(y_train[y_train == label]))
            num_samples_traindevel.append(len(y_traindevel[y_traindevel == label]))

        for label, ns_tr, ns_trd in zip(classes, num_samples_train, num_samples_traindevel):
            factor_tr = np.max(num_samples_train) // ns_tr
            X_train = np.concatenate(
                (X_train, np.tile(X_train[y_train == label], (factor_tr - 1, 1))))
            y_train = np.concatenate(
                (y_train, np.tile(y_train[y_train == label], (factor_tr - 1))))
            factor_trd = np.max(num_samples_traindevel) // ns_trd
            X_traindevel = np.concatenate((X_traindevel, np.tile(
                X_traindevel[y_traindevel == label], (factor_trd - 1, 1))))
            y_traindevel = np.concatenate((y_traindevel, np.tile(
                y_traindevel[y_traindevel == label], (factor_trd - 1))))

        # Feature normalisation
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_devel = scaler.transform(X_devel)
        X_traindevel = scaler.fit_transform(X_traindevel)
        X_test = scaler.transform(X_test)

        # Train SVM model with different complexities and evaluate
        uar_scores = []
        for comp in complexities:
            print('\nComplexity {0:.6f}'.format(comp))
            clf = svm.LinearSVC(C=comp, random_state=0, max_iter=10000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_devel)
            uar_scores.append(recall_score(
                y_devel, y_pred, labels=classes, average='macro'))
            print('UAR on Devel {0:.1f}'.format(uar_scores[-1] * 100))
            if show_confusion:
                print('Confusion matrix (Devel):')
                print(classes)
                print(confusion_matrix(y_devel, y_pred, labels=classes))

        # Train SVM model on the whole training data with optimum complexity and get predictions on test data
        optimum_complexity = complexities[np.argmax(uar_scores)]
        print('\nOptimum complexity: {0:.6f}, maximum UAR on Devel {1:.1f}\n'.format(
            optimum_complexity, np.max(uar_scores) * 100))
        if current_label == 'V_cat':
            list_V_value.append(np.max(uar_scores) * 100)
            list_V.append(feature_set_list)
        else:
            list_A_value.append(np.max(uar_scores) * 100)
            list_A.append(feature_set_list)

        clf = svm.LinearSVC(C=optimum_complexity, random_state=0, max_iter=10000)
        clf.fit(X_train, y_train)
        y_pred_label = clf.predict(X_devel)
        y_pred = clf._predict_proba_lr(X_devel)

        y_pred_L = [i[0] for i in y_pred]
        y_pred_M = [i[1] for i in y_pred]
        y_pred_H = [i[2] for i in y_pred]

        pred_file_name = task_name + '_' + feature_set_list + '_' + current_label + \
                '.devel.' + team_name + '_' + str(submission_index) + '.csv'

        print('Writing file ' + pred_file_name + '\n')

        # df = pd.DataFrame(data={'filename': df_agg['filename_text'][df_agg['partition'] == 'devel'].values,
        #                         'prediction': y_pred.flatten()},
        #                   columns=['filename', 'prediction'])

        df = pd.DataFrame(data={'filename': df_agg['filename_text'][df_agg['partition'] == 'devel'].values,
                                'perc_L': y_pred_M, 'perc_M': y_pred_H,'perc_H':y_pred_L,'predict': y_pred_label.flatten()},
                          columns=['filename', 'perc_L', 'perc_M', 'perc_H' , 'predict'])
        # print(df)
        df.to_csv('perc_'+ pred_file_name, index=False)

        print('Done.\n')

        # clf.fit(X_traindevel, y_traindevel)
        # y_pred = clf.predict(X_test)
        #
        # # Write out predictions to csv file (official submission format)
        # pred_file_name = task_name + '_' + feature_set + '_' + current_label + \
        #     '.test.' + team_name + '_' + str(submission_index) + '.csv'
        #
        # print('Writing file ' + pred_file_name + '\n')
        #
        # df = pd.DataFrame(data={'filename': df_agg['filename_text'][df_agg['partition'] == 'test'].values,
        #                         'prediction': y_pred.flatten()},
        #                   columns=['filename', 'prediction'])
        #
        # df.to_csv(pred_file_name, index=False)
    #
    # print('Done.\n')
print('V '+str(max(list_V_value)))
print(list_V[list_V_value.index(max(list_V_value))])
print('A '+str(max(list_A_value)))
print(list_A[list_A_value.index(max(list_A_value))])
