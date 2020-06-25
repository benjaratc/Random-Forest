#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# 1.โหลด csv เข้าไปใน Python Pandas

# In[2]:


df = pd.read_csv('../Desktop/DataCamp/german_credit_data 2.csv')
df.drop('Unnamed: 0', axis = 1, inplace = True)
df


# 2. เขียนโค้ดแสดง หัว10แถว ท้าย10แถว และสุ่ม10แถว

# In[3]:


df.head(10)


# In[4]:


df.sample(10)


# In[5]:


df.tail(10)


# 3. เช็คว่ามีข้อมูลที่หายไปไหม สามารถจัดการได้ตามความเหมาะสม

# In[6]:


df.isnull().sum()


# In[7]:


df['Saving accounts'].value_counts()


# In[8]:


df['Saving accounts'].fillna(value = 'little', inplace = True)


# In[9]:


df['Saving accounts'].value_counts()


# In[10]:


df['Checking account'].value_counts()


# In[11]:


df['Checking account'].fillna(value = 'little', inplace = True)


# In[12]:


df['Checking account'].value_counts()


# In[13]:


df.isnull().sum()


# In[14]:


df


# 4. ใช้ info และ describe อธิบายข้อมูลเบื้องต้น

# In[15]:


df.info()


# In[16]:


df.describe()


# 5. ใช้ pairplot ดูความสัมพันธ์เบื้องต้นของ features ที่สนใจ

# In[17]:


sns.pairplot(df)


# 6. ใช้ displot เพื่อดูการกระจายของแต่ละคอลัมน์

# In[18]:


sns.distplot(df['Duration'])


# In[19]:


sns.distplot(df['Credit amount'])


# 7. ใช้ heatmap ดูความสัมพันธ์ของคอลัมน์ที่สนใจ

# In[20]:


sns.heatmap(df.corr(), annot = df.corr())


# In[21]:


df['Risk'].replace('good',1,inplace = True)


# In[22]:


df['Risk'].replace('bad',0,inplace = True)


# In[23]:


df


# In[24]:


df1 = pd.get_dummies(df, drop_first = True)
df1


# In[25]:


plt.figure(figsize = (15,8))
sns.heatmap(df1.corr(), annot = df1.corr())


# 8. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation สูงสุด

# In[26]:


fig = plt.figure(figsize = (10,8))
sns.scatterplot(data = df1, x = 'Credit amount', y ='Duration')


# 9. สร้าง scatter plot ของความสัมพันธ์ที่มี Correlation ต่ำสุด

# In[27]:


fig = plt.figure(figsize = (10,8))
sns.scatterplot(data = df1, x = 'Housing_rent', y ='Housing_own')


# 10. สร้าง histogram ของ feature ที่สนใจ

# In[28]:


plt.hist(df['Credit amount'])


# In[29]:


plt.hist(df['Duration'])


# 11. สร้าง box plot ของ features ที่สนใจ

# In[30]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'Risk', y = 'Credit amount', orient = 'v')


# In[31]:


fig = plt.figure(figsize = (10,8))
sns.boxplot(data = df, x = 'Risk', y = 'Age', orient = 'v')


# 13. ทำ Data Visualization อื่นๆ (แล้วแต่เลือก)

# In[32]:


sns.countplot(data = df, x = 'Risk')


# In[33]:


sns.countplot(data = df, x = 'Job')


# 14. พิจารณาว่าควรทำ Normalization หรือ Standardization หรือไม่ควรทั้งสองอย่าง พร้อมให้เหตุผล 

# ควรทำ Normalization เพราะ x ไม่เป็น normal distribution

# # Default

# In[111]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score,accuracy_score


# In[112]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[113]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[117]:


rf =  RandomForestClassifier()
rf.fit(X_train,y_train)


# In[118]:


predicted = rf.predict(X_test)
predicted 


# In[119]:


confusion_matrix(y_test,predicted)


# In[120]:


print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted))
print('recall_score',recall_score(y_test,predicted))
print('f1 score',f1_score(y_test,predicted))


# # Normalization

# In[121]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[122]:


min_max_scaler = MinMaxScaler()


# In[123]:


X_minmax = min_max_scaler.fit_transform(X)
X_minmax


# In[124]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax,y, test_size = 0.2, random_state = 100)


# In[125]:


rf2 =  RandomForestClassifier()
rf2.fit(X_train,y_train)


# In[126]:


predicted2 = rf2.predict(X_test)
predicted2 


# In[127]:


confusion_matrix(y_test,predicted2)


# In[128]:


print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# # Standardization

# In[129]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[130]:


sc_X = StandardScaler()
X1 = sc_X.fit_transform(X)


# In[131]:


X_train,X_test,y_train,y_test = train_test_split(X1,y, test_size = 0.2, random_state = 100)


# In[132]:


rf3 =  RandomForestClassifier()
rf3.fit(X_train,y_train)


# In[133]:


predicted3 = rf3.predict(X_test)
predicted3 


# In[134]:


confusion_matrix(y_test,predicted3)


# In[135]:


print('accuracy score',accuracy_score(y_test,predicted3))
print('precision score',precision_score(y_test,predicted3))
print('recall_score',recall_score(y_test,predicted3))
print('f1 score',f1_score(y_test,predicted3))


# 15. เลือกช้อยที่ดีที่สุดจากข้อ 14 (หรือจะทำทุกอันแล้วนำมาเปรียบเทียบก็ได้)

# ผลของ Normalization ดีกว่า Standardization แต่ Default ดีที่สุด

# 16. วัดผลโมเดล โดยใช้ confusion matrix และ ประเมินผลด้วยคะแนน Accuracy, 
# F1 score, Recall, Precision

# In[136]:


#Standardization
print('accuracy score',accuracy_score(y_test,predicted3))
print('precision score',precision_score(y_test,predicted3))
print('recall_score',recall_score(y_test,predicted3))
print('f1 score',f1_score(y_test,predicted3))


# In[137]:


#Normalization
print('accuracy score',accuracy_score(y_test,predicted2))
print('precision score',precision_score(y_test,predicted2))
print('recall_score',recall_score(y_test,predicted2))
print('f1 score',f1_score(y_test,predicted2))


# In[138]:


#Default
print('accuracy score',accuracy_score(y_test,predicted))
print('precision score',precision_score(y_test,predicted))
print('recall_score',recall_score(y_test,predicted))
print('f1 score',f1_score(y_test,predicted))


# 17. หาค่า parameter combination ที่ดีที่สุด สำหรับ Dataset นี้ โดยใช้ GridSearch (Hyperparameter Tuning)

# In[146]:


from sklearn.model_selection import GridSearchCV


# In[147]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16], 'n_estimators': [10,20,50]}


# In[148]:


grid_search = GridSearchCV(RandomForestClassifier(), param_combination,verbose = 3)


# In[149]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[150]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 100)


# In[151]:


grid_search.fit(X_train,y_train)


# In[152]:


grid_search.best_params_


# In[153]:


grid_search.best_estimator_


# In[154]:


grid_predicted = grid_search.predict(X_test)
grid_predicted 


# In[155]:


print('accuracy score',accuracy_score(y_test,grid_predicted))
print('precision score',precision_score(y_test,grid_predicted))
print('recall_score',recall_score(y_test,grid_predicted))
print('f1 score',f1_score(y_test,grid_predicted))


# 18. เลือกเฉพาะ features ที่สนใจมาเทรนโมเดล และวัดผลเปรียบเทียบกับแบบ all-features

# In[156]:


X = df1[['Credit amount','Duration','Checking account_moderate']]
y = df1['Risk']


# In[157]:


min_max_scaler2 = MinMaxScaler()


# In[158]:


X_minmax2 = min_max_scaler2.fit_transform(X)
X_minmax2


# In[159]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax2,y, test_size = 0.2, random_state = 100)


# In[160]:


rf4 =  RandomForestClassifier()
rf4.fit(X_train,y_train)


# In[161]:


predicted4 = rf4.predict(X_test)
predicted4 


# In[162]:


confusion_matrix(y_test,predicted4)


# In[163]:


print('accuracy score',accuracy_score(y_test,predicted4))
print('precision score',precision_score(y_test,predicted4))
print('recall_score',recall_score(y_test,predicted4))
print('f1 score',f1_score(y_test,predicted4))


# 19. ทำ Visualization ของค่า F1 Score ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[167]:


data = {'Default' : f1_score(y_test,predicted) , 'Grid Search': f1_score(y_test,grid_predicted),
        'Normalization' : f1_score(y_test,predicted2)}
data


# In[168]:


Series1 = pd.Series(data = data)
Series1


# In[169]:


df2 = pd.DataFrame(Series1)
df2


# In[170]:


sns.barplot(data = df2, x = df2.index, y = df2[0])
plt.ylabel('f1 score')


# 20. ทำ Visualization ของค่า Recall ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[171]:


data = {'Default' : recall_score(y_test,predicted) , 'Grid Search': recall_score(y_test,grid_predicted),
        'Normalization' : recall_score(y_test,predicted2)}
data


# In[172]:


Series2 = pd.Series(data = data)
Series2


# In[173]:


df3 = pd.DataFrame(Series2)
df3


# In[174]:


sns.barplot(data = df3, x = df3.index, y = df3[0])
plt.ylabel('recall score')


# 21. ทำ Visualization ของค่า Accuracy ระหว่าง ผลลัพธ์ที่ได้จากค่า Default, ผลลัพธ์ที่ได้จากการใช้ Grid Search และ ผลลัพธ์ของ Normalization

# In[175]:


data = {'Default' : accuracy_score(y_test,predicted) , 'Grid Search': accuracy_score(y_test,grid_predicted),
        'Normalization' : accuracy_score(y_test,predicted2)}
data


# In[176]:


Series3 = pd.Series(data = data)
Series3


# In[177]:


df4 = pd.DataFrame(Series3)
df4


# In[178]:


sns.barplot(data = df4, x = df4.index, y = df4[0])
plt.ylabel('accuracy score')


# 22. สามารถใช้เทคนิคใดก็ได้ตามที่สอนมา ใช้ Decision Tree Algorithm แล้วให้ผลลัพธ์ที่ดีที่สุดที่เป็นไปได้ (อาจจะรวม Grid Search กับ Normalization/Standardization ?)

# # Grid Search Standardization

# In[179]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[180]:


sc_X2 =  StandardScaler()
X2 = sc_X2.fit_transform(X)


# In[181]:


X_train,X_test,y_train,y_test = train_test_split(X2,y,test_size =0.3, random_state = 20)


# In[183]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16], 'n_estimators':[10,20,50,100]}


# In[185]:


grid_search1 = GridSearchCV(RandomForestClassifier(), param_combination,verbose = 3)


# In[187]:


grid_search1.fit(X_train,y_train)


# In[188]:


grid_search1.best_params_


# In[189]:


grid_predicted1 = grid_search1.predict(X_test)
grid_predicted1


# In[190]:


confusion_matrix(y_test,grid_predicted1)


# In[191]:


print('accuracy score',accuracy_score(y_test,grid_predicted1))
print('precision score',precision_score(y_test,grid_predicted1))
print('recall_score',recall_score(y_test,grid_predicted1))
print('f1 score',f1_score(y_test,grid_predicted1))


# # Grid Search Normalization

# In[206]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[207]:


min_max_scaler3 = MinMaxScaler()


# In[208]:


X_minmax3 = min_max_scaler3.fit_transform(X)
X_minmax3


# In[209]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax3,y, test_size = 0.2, random_state = 100)


# In[210]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16],'n_estimators':[10,20,50,100]}


# In[211]:


grid_search2 = GridSearchCV(RandomForestClassifier(), param_combination,verbose = 3)


# In[212]:


grid_search2.fit(X_train,y_train)


# In[213]:


grid_search2.best_params_


# In[214]:


grid_predicted2 = grid_search2.predict(X_test)
grid_predicted2


# In[215]:


confusion_matrix(y_test,grid_predicted2)


# In[216]:


print('accuracy score',accuracy_score(y_test,grid_predicted2))
print('precision score',precision_score(y_test,grid_predicted2))
print('recall_score',recall_score(y_test,grid_predicted2))
print('f1 score',f1_score(y_test,grid_predicted2))


# # Best Decision Tree Model

# In[218]:


X = df1.drop('Risk', axis = 1)
y = df1['Risk']


# In[219]:


min_max_scaler3 = MinMaxScaler()


# In[220]:


X_minmax3 = min_max_scaler3.fit_transform(X)
X_minmax3


# In[221]:


X_train,X_test,y_train,y_test = train_test_split(X_minmax3,y, test_size = 0.2, random_state = 100)


# In[222]:


param_combination = {'max_depth': [4,8,16,32,64], 'min_samples_leaf':[1,2,4,8,16]}


# In[228]:


grid_search3 = GridSearchCV(DecisionTreeClassifier(), param_combination,verbose = 3)


# In[229]:


grid_search3.fit(X_train,y_train)


# In[230]:


grid_search3.best_params_


# In[231]:


grid_predicted3 = grid_search3.predict(X_test)
grid_predicted3


# In[232]:


confusion_matrix(y_test,grid_predicted3)


# In[233]:


print('accuracy score',accuracy_score(y_test,grid_predicted3))
print('precision score',precision_score(y_test,grid_predicted3))
print('recall_score',recall_score(y_test,grid_predicted3))
print('f1 score',f1_score(y_test,grid_predicted3))


# 23. สร้าง bar chart เปรียบเทียบค่า Accuracy, F1 score, Recall, Precision ของ Decision Tree Model ที่ดีที่สุด กับ Random Forest Model ที่ดีที่สุด

# In[311]:


data = np.array([[accuracy_score(y_test,grid_predicted3), f1_score(y_test,grid_predicted3),
                             recall_score(y_test,grid_predicted3),precision_score(y_test,grid_predicted3)],
                             [accuracy_score(y_test,predicted), f1_score(y_test,predicted),
                             recall_score(y_test,predicted),precision_score(y_test,predicted)]]).reshape(2,4)
data


# In[312]:


df2 = pd.DataFrame(data,
                   columns =['Accuracy', 'F1', 'Recall','Precision'],
                   index = ['Decision Tree','Random Forest'])
df2


# In[313]:


df2.columns


# In[314]:


df2.index


# In[322]:


fig = plt.figure(figsize = (7,6))
sns.barplot(y = 'Accuracy', x = df2.index, data = df2)


# In[323]:


fig = plt.figure(figsize = (7,6))
sns.barplot(y = 'F1', x = df2.index, data = df2)


# In[325]:


fig = plt.figure(figsize = (7,6))
sns.barplot(y = 'Recall', x = df2.index, data = df2)


# In[324]:


fig = plt.figure(figsize = (7,6))
sns.barplot(y = 'Precision', x = df2.index, data = df2)

