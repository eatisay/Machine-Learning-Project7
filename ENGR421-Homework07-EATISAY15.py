import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
from sklearn.metrics import roc_auc_score
pd.options.mode.use_inf_as_na = True
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#------------------------------------------------------PART 1--------------------------------------------------------------

#Pre-Proccess Part with training Data

raw_training = pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target1_training_data.csv')
raw_label= pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target1_training_label.csv')

temp_df=pd.DataFrame(raw_label['TARGET'])
concat_x_y=pd.concat([raw_training,temp_df],axis=1)


for (columnName, columnData) in concat_x_y.iteritems():
    nanct=0;
    if float(columnData.isna().sum())/len(columnData) > 0.5:
        concat_x_y=concat_x_y.drop([columnName], axis=1)

end_point=len(concat_x_y.columns)-1

x = concat_x_y.iloc[:, 0:end_point].values

#print(x)
#print x_format.describe()
y = concat_x_y.iloc[:, end_point].values

x[:, 45] = labelencoder.fit_transform(x[:, 45])
x[:, 43] = labelencoder.fit_transform(x[:, 43])
x[:, 68] = labelencoder.fit_transform(x[:, 68])

x_format=pd.DataFrame(x)
#x_format.fillna(x_format.mean(), inplace=True)

for (columnName, columnData) in x_format.iteritems():
    temp=columnName

    x_format.iloc[:, temp] = x_format.iloc[:, temp].fillna(x_format.iloc[:, temp].mean(), inplace=False)


#x_format.iloc[:,3]=x_format.iloc[:,3].fillna(x_format.iloc[:,3].mean(),inplace=False)

x=x_format.values

kf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)
for train_index, test_index in kf.split(x):
      x_training, x_testing = x[train_index], x[test_index]
      y_training, y_testing = y[train_index], y[test_index]

#x_training, x_testing, y_training, y_testing =train_test_split(x, y, test_size=0.8, random_state=42)

mein_classifier = GradientBoostingClassifier(n_estimators=400, random_state=0,learning_rate=0.09,warm_start=True)
mein_classifier.fit(x_training,y_training)

#print(mein_classifier.feature_importances_)

prediction=mein_classifier.predict(x_testing)

print(round(roc_auc_score(y_testing, prediction),5))

#Evaluation Part with test Data

final_testing = pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target1_test_data.csv')

for (columnName, columnData) in final_testing.iteritems():
    if float(columnData.isna().sum())/len(columnData) > 0.5:
        final_testing=final_testing.drop([columnName], axis=1)

final_length=len(final_testing.columns)-1

final_x = final_testing.iloc[:, 0:final_length].values


final_x[:, 45] = labelencoder.fit_transform(final_x[:, 45])
final_x[:, 43] = labelencoder.fit_transform(final_x[:, 43])
final_x[:, 69] = labelencoder.fit_transform(final_x[:, 69])

final_x_format=pd.DataFrame(final_x)

for (columnName, columnData) in final_x_format.iteritems():
    temp=columnName
    final_x_format.iloc[:, temp] = final_x_format.iloc[:, temp].fillna(final_x_format.iloc[:, temp].mean(), inplace=False)

result_x=final_x_format.values

final_prediction=mein_classifier.predict(result_x)

t1_probability = pd.DataFrame(mein_classifier.predict_proba(result_x))

t1_result = t1_probability.iloc[:,1]

print (t1_result)

pd.DataFrame(t1_result).to_csv("hw07_target1_test_predictions.csv")

#------------------------------------------------------PART 2--------------------------------------------------------------

t2_raw_training = pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target2_training_data.csv')
t2_raw_label= pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target2_training_label.csv')
t2_final_testing = pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target2_test_data.csv')

t2_temp_df=pd.DataFrame(t2_raw_label['TARGET'])
t2_concat_x_y=pd.concat([t2_raw_training,t2_temp_df],axis=1)

for (columnName, columnData) in t2_concat_x_y.iteritems():
    if float(columnData.isna().sum())/len(columnData) > 0.4:
        t2_concat_x_y=t2_concat_x_y.drop([columnName], axis=1)
        t2_final_testing=t2_final_testing.drop([columnName],axis=1)

t2_end_point=len(t2_concat_x_y.columns)-1

t2_x = t2_concat_x_y.iloc[:, 0:t2_end_point].values
t2_y = t2_concat_x_y.iloc[:, t2_end_point].values

t2_x[:, 32] = labelencoder.fit_transform(t2_x[:, 32])
t2_x[:, 64] = labelencoder.fit_transform(t2_x[:, 64])
t2_x[:, 181] = labelencoder.fit_transform(t2_x[:, 181])

t2_x_format=pd.DataFrame(t2_x)


for (columnName, columnData) in t2_x_format.iteritems():
    temp=columnName
    t2_x_format.iloc[:, temp] = t2_x_format.iloc[:, temp].fillna(t2_x_format.iloc[:, temp].mean(), inplace=False)

t2_x_format=t2_x_format.drop([62], axis=1)
t2_x_format=t2_x_format.drop([75], axis=1)
t2_x_format=t2_x_format.drop([167], axis=1)
t2_x_format=t2_x_format.drop([63], axis=1)
t2_x_format=t2_x_format.drop([42], axis=1)


t2_x=t2_x_format.values

kf = RepeatedKFold(n_splits=4, n_repeats=20, random_state=0)
for train_index, test_index in kf.split(t2_x):
      t2_x_training, t2_x_testing = t2_x[train_index], t2_x[test_index]
      t2_y_training, t2_y_testing = t2_y[train_index], t2_y[test_index]

t2_mein_classifier = GradientBoostingClassifier(n_estimators=400, random_state=0,learning_rate=0.09,warm_start=True)
t2_mein_classifier.fit(t2_x_training,t2_y_training)

array=t2_mein_classifier.feature_importances_.tolist()
dropto=array.index(min(array))
#print(dropto)
#print(t2_mein_classifier.feature_importances_)

prediction=t2_mein_classifier.predict(t2_x_testing)

print(round(roc_auc_score(t2_y_testing, prediction),5))

#Evaluation Part with test Data


t2_final_length=len(t2_final_testing.columns)

t2_final_x = t2_final_testing.iloc[:, 0:t2_final_length].values


t2_final_x[:, 32] = labelencoder.fit_transform(t2_final_x[:, 32])
t2_final_x[:, 64] = labelencoder.fit_transform(t2_final_x[:, 64])
t2_final_x[:, 181] = labelencoder.fit_transform(t2_final_x[:, 181])


t2_final_x_format=pd.DataFrame(t2_final_x)

for (columnName, columnData) in t2_final_x_format.iteritems():
    temp=columnName
    t2_final_x_format.iloc[:, temp] = t2_final_x_format.iloc[:, temp].fillna(t2_final_x_format.iloc[:, temp].mean(), inplace=False)

t2_final_x_format=t2_final_x_format.drop([62], axis=1)
t2_final_x_format=t2_final_x_format.drop([75], axis=1)
t2_final_x_format=t2_final_x_format.drop([167], axis=1)
t2_final_x_format=t2_final_x_format.drop([63], axis=1)
t2_final_x_format=t2_final_x_format.drop([42], axis=1)

t2_result_x=t2_final_x_format.values

#t2_final_prediction=t2_mein_classifier.predict(t2_result_x)

t2_probability = pd.DataFrame(t2_mein_classifier.predict_proba(t2_result_x))

t2_result = t2_probability.iloc[:,1]

print(t2_result)

pd.DataFrame(t2_result).to_csv("hw07_target2_test_predictions.csv")

#--------------------------------------------------PART 3-----------------------------------------------------------------

t3_raw_training = pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target3_training_data.csv')
t3_raw_label= pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target3_training_label.csv')

t3_temp_df=pd.DataFrame(t3_raw_label['TARGET'])
t3_concat_x_y=pd.concat([t3_raw_training,t3_temp_df],axis=1)

for (columnName, columnData) in t3_concat_x_y.iteritems():
    if float(columnData.isna().sum())/len(columnData) > 0.4:
        t3_concat_x_y=t3_concat_x_y.drop([columnName], axis=1)

t3_end_point=len(t3_concat_x_y.columns)-1

t3_x = t3_concat_x_y.iloc[:, 0:t3_end_point].values
t3_x[:, 36] = labelencoder.fit_transform(t3_x[:, 36])
t3_x[:, 145] = labelencoder.fit_transform(t3_x[:, 145])


t3_x_format=pd.DataFrame(t3_x)
for (columnName, columnData) in t3_x_format.iteritems():
    temp=columnName
    t3_x_format.iloc[:, temp] = t3_x_format.iloc[:, temp].fillna(t3_x_format.iloc[:, temp].mean(), inplace=False)

t3_x_format=t3_x_format.drop([1], axis=1)

t3_x=t3_x_format.values

t3_x = StandardScaler().fit_transform(t3_x)

t3_y = t3_concat_x_y.iloc[:, t3_end_point].values

pca = PCA()
t3_x_pca = pca.fit_transform(t3_x)

t3_pca_format=pd.DataFrame(t3_x_pca)

for index in range(80,184):
    t3_pca_format=t3_pca_format.drop([index], axis=1)

t3_x_pca=t3_pca_format.values


kf = RepeatedKFold(n_splits=4, n_repeats=20, random_state=0)
for train_index, test_index in kf.split(t3_x_pca):
      t3_x_training, t3_x_testing = t3_x_pca[train_index], t3_x_pca[test_index]
      t3_y_training, t3_y_testing = t3_y[train_index], t3_y[test_index]


t3_mein_classifier = GradientBoostingClassifier(n_estimators=400, random_state=0,learning_rate=0.09,warm_start=True)
t3_mein_classifier.fit(t3_x_training,t3_y_training)

t3_prediction=t3_mein_classifier.predict(t3_x_testing)

print(round(roc_auc_score(t3_y_testing, t3_prediction),5))

#-----------------------------------------Evaluation Part with test Data----------------------------------------------------------

t3_final_testing = pd.read_csv(r'C:\Users\egebe\Desktop\project7\hw07_target3_test_data.csv')

for (columnName, columnData) in t3_final_testing.iteritems():
    if float(columnData.isna().sum())/len(columnData) > 0.4:
        t3_final_testing=t3_final_testing.drop([columnName], axis=1)

final_t3_end_point=len(t3_final_testing.columns)-1

final_t3_x = t3_final_testing.iloc[:, 0:final_t3_end_point].values
final_t3_x[:, 36] = labelencoder.fit_transform(final_t3_x[:, 36])
final_t3_x[:, 145] = labelencoder.fit_transform(final_t3_x[:, 145])

final_t3_x_format=pd.DataFrame(final_t3_x)
for (columnName, columnData) in final_t3_x_format.iteritems():
    temp=columnName
    final_t3_x_format.iloc[:, temp] = final_t3_x_format.iloc[:, temp].fillna(final_t3_x_format.iloc[:, temp].mean(), inplace=False)

final_t3_x=final_t3_x_format.values

final_t3_x = StandardScaler().fit_transform(final_t3_x)
pca = PCA()
final_t3_x_pca = pca.fit_transform(final_t3_x)

final_t3_x_pca_format=pd.DataFrame(final_t3_x_pca)

for index in range(80,184):
    final_t3_x_pca_format=final_t3_x_pca_format.drop([index], axis=1)

final_t3_x_pca=final_t3_x_pca_format.values

t3_final_prediction=t3_mein_classifier.predict(final_t3_x_pca)

#print(np.sum(t3_final_prediction)/float(len(t3_final_prediction)))
t3_probability = pd.DataFrame(t3_mein_classifier.predict_proba(final_t3_x_pca))

t3_result = t3_probability.iloc[:,1]

print (t3_result)

pd.DataFrame(t3_result).to_csv("hw07_target3_test_predictions.csv")












