import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from scipy import stats

path=''

classes = ['L', 'M', 'H']
label_options = ['V_cat', 'A_cat']
df_labels = pd.read_csv(path+'labels.csv')
feature_sets=['ComParE','BoAW','auDeep','DeepSpectrum_resnet50','fisher_vector']
tuned_parameters = {'gamma': [1000,100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9],
                    'C': [0.001,0.01,0.1,1, 10, 100, 1000,10000,1e5,1e6,1e7,1e8,1e9,1e10]}
# stat_list=['argmax','argmin','max','min','mean','std','kurtosis','meanstd_err','skew','range','intercept','slope']
def getStats(e):
    slope, intercept, r_value, p_value, std_err = stats.linregress([i for i in range(len(e))],e)
    return [np.argmax(e),np.argmin(e),np.max(e),np.min(e),np.mean(e),
            np.std(e),stats.kurtosis(e),stats.sem(e),stats.skew(e),(np.max(e)-np.min(e)),slope,intercept]

def readcsv(feature,cat):
    filepath=path+'before agg best on test/Devel/'
    files=os.listdir(filepath)
    for file in files:
        if cat in file and feature in file:
            result=[]
            result.append(pd.read_csv(filepath+file)['L'].values)
            result.append(pd.read_csv(filepath+file)['M'].values)
            result.append(pd.read_csv(filepath+file)['H'].values)
            return  result
          

def agg(feature_sets,current_label):

    preds=[]
    for feature_set in feature_sets:
        #[[l],[m],[h]]
        preds.append(readcsv(feature_set,current_label))
        
    y_devel = df_labels[current_label][df_labels['filename_audio'].str.startswith('devel')].values
    
    stories_devel = pd.DataFrame(
                      data={'filename_audio': df_labels['filename_audio'][df_labels['filename_audio'].str.startswith('devel')].values,
                      'filename_text': df_labels['filename_text'][df_labels['filename_audio'].str.startswith('devel')].values,  # filename_text == ID_Story
                      'true': y_devel},columns=['filename_audio', 'filename_text','true'])
    stories=stories_devel.groupby(['filename_text']).agg(
                lambda x: x.value_counts().sort_index().sort_values(ascending=False, kind='mergesort').index[0])
    X_train=[]
    for curr_story in stories.index:
        st=stories_devel['filename_text'][stories_devel['filename_text'] == curr_story]
        temp=[]
        for j in range(len(feature_sets)):
            for i in range(3):
                temp.extend(getStats(preds[j][i][st.index[0]:st.index[-1]+1]))
        X_train.append(temp)
        
    y=stories['true'].values
    
    uar_scores = []
    
    print('gamma:\t',end='')
    for g in tuned_parameters['gamma']:
        print(g,'\t',end='')
    print('\nC:')
    for comp in tuned_parameters['C']:
        print(comp,':\t\t',end='')
        for g in tuned_parameters['gamma']:
            clf = svm.SVC(C=comp,gamma=g)  

            #cross validation
            kf = KFold(n_splits=5)
            
            y_pred=[]
            y_proba=[]
            
            for train_index, test_index in kf.split(stories.index):
#                 print(test_index)
                scaler = MinMaxScaler()
                X_traint = scaler.fit_transform([X_train[i] for i in train_index])
                X_develt = scaler.transform([X_train[i] for i in test_index])
                clf.fit(X_traint, [y[i] for i in train_index])
                y_pred.extend(clf.predict(X_develt))
                y_proba.extend(clf.decision_function(X_develt))
                
            uar_maj = recall_score(y, y_pred, labels=classes,average='macro')
            uar_scores.append(uar_maj)
            print('{0:.2f}'.format(uar_scores[-1]*100),'\t',end='')
            if uar_scores[-1]==max(uar_scores):
                
                best=(g,comp)
                best_pred=y_pred
                best_prob=y_proba
        
        print()
    print(max(uar_scores))
    return (max(uar_scores),best,best_pred,best_prob)


def write_result(pred_file_name,pred,proba):        
    if len(proba[0])==3:
        temp=[]
        for i in range(3):
            temp.append([j[i] for j in proba])     
        proba=temp

    print('Writing file ' + pred_file_name + '\n')
    
    stories_devel = pd.DataFrame(
          data={'filename_audio': df_labels['filename_audio'][df_labels['filename_audio'].str.startswith('devel')].values,
          'filename_text': df_labels['filename_text'][df_labels['filename_audio'].str.startswith('devel')].values # filename_text == ID_Story
               },columns=['filename_audio', 'filename_text'])

    stories=stories_devel.groupby(['filename_text']).agg(
                lambda x: x.value_counts().sort_index().sort_values(ascending=False, kind='mergesort').index[0])

    df = pd.DataFrame(data={'filename': stories.index.values,
                            'prediction': pred,
                            'L':proba[1],
                            'M':proba[2],
                            'H':proba[0]},
                      columns=['filename', 'prediction','L','M','H'])
    print(pred_file_name)
    print(df)
    df.to_csv(pred_file_name, index=False)
    print('Done.\n')  

if not os.path.exists(path+'after agg best on test/devel'):
    os.mkdir(path+'after agg best on test/devel')

for feature in feature_sets:
    for cat in label_options:
        curr=agg([feature],cat)

        out_path=path+'after agg best on test/devel/'+feature
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        write_result('%s/%s.csv'%(out_path,cat),curr[2],curr[3])