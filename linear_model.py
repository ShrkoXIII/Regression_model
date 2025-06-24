import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
st.title("price prediction using linear regression")
st.header('User Input:')
## file uploader
uploaded_file=st.file_uploader("please upload your file",type=['csv'])
# if st.button("read"):
if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    st.write("uploaded data",data)
        # st.dataframe()
    feature=st.multiselect('select features',data.columns)
    model_feature=data[feature]
    columns2_ignore=data.drop(feature,axis=1 )
    target=st.multiselect('select target',columns2_ignore.columns)
    model_target=data[target]
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(model_feature,model_target,random_state=0,test_size=.2)
    model=LinearRegression()
    model.fit(Xtrain,Ytrain)
    y_pred=model.predict(Xtest)
    st.subheader("residual check")
    fig,ax=plt.subplots()
    ax.scatter(y_pred,Ytest)
    ax.set_xlabel('measure')
    ax.set_ylabel('predictions')
    st.pyplot(fig)
else:
    st.error('please upload your file')    

         