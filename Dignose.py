import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

Categories = ['fungal', 'normal', 'pimple', 'psoriasis']

flat_data_arr=[]                                         
target_arr=[]                                            
datadir='./static/images'   
print('3')
for i in Categories:                                     
  path=os.path.join(datadir,i)
  for img in os.listdir(path):                           
    img_array=imread(os.path.join(path,img))
    img_resized=resize(img_array,(100,100,2))             
    flat_data_arr.append(img_resized.flatten())          
    target_arr.append(Categories.index(i))    
print('2')                     
flat_data=np.array(flat_data_arr)                       
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target
x=df.iloc[:,:-1]
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0,stratify=y)
print('1')

param_grid={'C':[1,1000000],'gamma':[0.0001,1],'kernel':['rbf','poly']}                                    
svc=svm.SVC(probability=True)                                                                                           
model=GridSearchCV(svc,param_grid)                                                                                     
model.fit(x_train,y_train)
model.best_params_
y_pred=model.predict(x_test)
np.array(y_test)
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

url=input('Enter URL or name of Image ')
img=imread(url)
img_resize=resize(img,(100,100,2))
l=[img_resize.flatten()]
probability=model.predict_proba(l)
for ind,val in enumerate(Categories):
  print(f'{val} = {probability[0][ind]*100}%')
predicted_class = Categories[np.argmax(probability)]
print(f"The predicted image is: {predicted_class} with {probability[0][np.argmax(probability)] * 100:.2f}%.")
print(f'Is the image a {predicted_class} ?(y/n)')
while(True):
    b=input()
    print("please enter either y or n")
    if(b=="y"):
        pickle.dump(model,open('model.pkl','wb'))
        with open('model.pkl', 'wb') as new_model_file:
            pickle.dump(model, new_model_file)
        break
    
    if(b=='n'):
        print("What is the image?")
        for i in range(len(Categories)):
            print(f"Enter {i} for {Categories[i]}")
        k=int(input())
        while(k<0 or k>=len(Categories)):
            print(f"Please enter a valid number between 0-{len(Categories)-1}")
            k=int(input())
        flat_arr=flat_data_arr.copy()
        tar_arr=target_arr.copy()
        tar_arr.append(k)
        flat_arr.extend(l)
        tar_arr=np.array(tar_arr)
        flat_df=np.array(flat_arr)
        df1=pd.DataFrame(flat_df)
        df1['Target']=tar_arr
        model1=GridSearchCV(svc,param_grid)
        x1=df1.iloc[:,:-1]
        y1=df1.iloc[:,-1]
        x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.20,random_state=77,stratify=y1)
        d={}
        for i in model.best_params_:
            d[i]=[model.best_params_[i]]
        model1=GridSearchCV(svc,d)
        model1.fit(x_train1,y_train1)
        y_pred1=model.predict(x_test1)
        print(f"The model is now {accuracy_score(y_pred1,y_test1)*100}% accurate")
        print("Thank you for your feedback")
        pickle.dump(model1,open('model.pkl','wb'))
        with open('model.pkl', 'wb') as new_model_file:
            pickle.dump(model1, new_model_file)
        break
