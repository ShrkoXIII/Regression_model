import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
st.title("price prediction using linear regression")
st.sidebar.header("Configurations")
## file uploader
uploaded_file=st.sidebar.file_uploader("please upload your file",type=['csv'])
if st.sidebar.button("read"):
    if uploaded_file is not None:
        st.session_state['data']=pd.read_csv(uploaded_file)
        
    else:
        st.sidebar.error('please upload your file')            
if 'data' in st.session_state:
    data=st.session_state['data']
    st.write("uploaded data",data)
    try :    
        features=st.sidebar.multiselect('select features',data.columns)
        
        model_feature=data[features]
    
        columns2_ignore=data.drop(features,axis=1 )
        target=st.sidebar.multiselect('select target',columns2_ignore.columns)     
        model_target=data[target]      
        Xtrain,Xtest,Ytrain,Ytest=train_test_split(model_feature,model_target,random_state=0,test_size=.2)
        model=LinearRegression()
        model.fit(Xtrain,Ytrain)
        y_pred=model.predict(Xtest)
        st.subheader("residual check")
        st.write('mean_squared_error',mean_squared_error(y_pred,Ytest))
        st.write('mean_absolute_error',mean_absolute_error(y_pred,Ytest))
        st.write('r2_score',r2_score(y_pred,Ytest))
        fig,ax=plt.subplots()
        ax.scatter(y_pred,Ytest)
        ax.set_xlabel('measure')
        ax.set_ylabel('predictions')
        st.pyplot(fig)   
        st.write('now u could try it by your self')
        input_data={}
        for feature in features:
            input_data[feature]=st.sidebar.number_input(f"input{feature}",value=float(data[feature].mean()))
        user_input_df=pd.DataFrame([input_data])
        st.write("your input is ",user_input_df)
        if st.button("predict"):
           predicted= model.predict(user_input_df)
           st.write("the predict is ",predicted)    
    except:
        st.sidebar.write('u should select feature&target')
         