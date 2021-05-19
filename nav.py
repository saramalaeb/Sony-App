import numpy as np
import pandas as pd
import streamlit as st
import chart_studio.plotly as py
import matplotlib.pyplot as plt
import plotly
import math 

import plotly.express as px
from PIL import Image
import plotly.graph_objs as go
import statsmodels.api as sm
import seaborn as sns
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title = 'Streamlit individual project',
    page_icon = '✅',
    layout = 'wide'
)

st.title("MSBA 370 -Streamlit Individual Project" )
st.title("Sony Dataset Dashboard & Sales Predictor")
Sony_Data = pd.read_csv("Sony.csv")
from matplotlib.figure import Figure

nav = st.sidebar.radio("Navigation",["Intro","Dashboard","Prediction"])
if nav == "Intro":
    st.write("Hey there!Welcome to My Streamlit Indiviudal Project.")
 
    st.write(" The App introduces you to the Sony Music Entertainment Dataset and predicts the Sales providing insights and sales decisions.")

    
    st.subheader("**To begin, Let's Introduce You to Sony Music Entertainemt Data ** 👇")
    st.subheader("Sony Music Entertainment is an American global music company.")
    if st.checkbox("Show Data"):
        st.table(Sony_Data)
    if st.checkbox("Describe Data"):
       show= Sony_Data.describe()
       show
    if st.checkbox("Show Size"):
       show= Sony_Data.shape
       show



    if st.checkbox("Show Data Columns"):
        columns= sorted(Sony_Data)
        st.table(columns)
    if st.checkbox("Select Columns"):
     column= st.selectbox("Please Select Column",
['Adverts ', 'Age', 'Airplay', 'Album No.', 'Attract', 'Gender', 'Genre', 'Language',  'StreamingServices',
])
     if column == 'Adverts ':
        data=Sony_Data['Adverts ']
        data
     if column == 'Age':
        data=Sony_Data['Age']
        data
    
     if column == 'Airplay':
        data=Sony_Data['Airplay']
        data
     if column == 'Album No.':
        data=Sony_Data['Album No.']
        data
     if column == 'Gender':
        data=Sony_Data['Gender']
        data
     if column == 'Airplay':
        data=Sony_Data['Airplay']
        data
     if column=="Genre":
        data=Sony_Data['Genre']
        data
     if column == 'Language':
        data=Sony_Data['Language']
        data
     if column == 'StreamingServices':
        data=Sony_Data['StreamingServices']
        data
    
    






if nav == "Dashboard":
    st.subheader("Sony Dataset Plots")
    
    
    if st.checkbox("Show Table"):
        st.table(Sony_Data)
    
    chart1, chart2 = st.beta_columns(2)

    with chart1:
        st.subheader('Albums Released in a Year')
        year_df = pd.DataFrame(
        Sony_Data["Year Released"].dropna().value_counts()).reset_index()
        year_df = year_df.sort_values(by='index')
        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=year_df['index'],
        y=year_df['Year Released'], color='goldenrod', ax=ax)
        ax.set_xlabel('Year')
        ax.set_ylabel('Albums')
        st.pyplot(fig)
    with chart2:
        st.subheader("Sales & Piracy")
        Sales_Piracy = px.scatter(Sony_Data, x='Piracy', y="Sales ",trendline="lowess")
        st.plotly_chart(Sales_Piracy)
    

    st.subheader("Age")
    age_df = Sony_Data['Age'].value_counts().to_frame()
    age_df = age_df.reset_index()
    age_df.columns = ['Age Range','Counts']
    p01 = px.bar(age_df,x='Age Range',y='Counts')
    st.plotly_chart(p01,use_container_width=True)

    #New
    

    
    st.subheader("Gender vs Genre")
    GenderGenre_df = Sony_Data.groupby(['Gender','Genre']).size().to_frame().reset_index()
    GenderGenre_df.rename(columns={0:'Counts'},inplace=True)
    po2 = px.bar(GenderGenre_df,x='Genre',y='Counts',color='Gender')
    st.plotly_chart(po2)
        #plot 3
    chart1, chart2 = st.beta_columns(2)
    with chart1:
        st.subheader("Pie Chart (Gender)")
        gen_df = Sony_Data['Gender'].value_counts().to_frame()
        gen_df = gen_df.reset_index()
        gen_df.columns = ['Gender Type','Counts']
# st.dataframe(gen_df)
        p01 = px.pie(gen_df,names='Gender Type',values='Counts')
        st.plotly_chart(p01,use_container_width=True)
    with chart2:
#plo4
        st.subheader("Language of Labels")
        city_df = Sony_Data['Language'].value_counts().to_frame()
        city_df = city_df.reset_index()
        city_df.columns = ['Language','Counts']
        p01 = px.pie(city_df,names='Language',values='Counts')
        st.plotly_chart(p01,use_container_width=True)

    chart1, chart2 = st.beta_columns(2)
    with chart1:
#plot5
        st.subheader("Country and Genre")
        CountryGenre_df = Sony_Data.groupby(['Country','Genre']).size().to_frame().reset_index()
        CountryGenre_df.rename(columns={0:'Counts'},inplace=True)
        po2 = px.bar(CountryGenre_df,x='Country',y='Counts',color='Genre')
        st.plotly_chart(po2)
#Age vs Genre 6
    with chart2:
        st.subheader("Age vs Genre")
        AgeGenre_df = Sony_Data.groupby(['Age','Genre']).size().to_frame().reset_index()
        AgeGenre_df.rename(columns={0:'Counts'},inplace=True)
        po2 = px.bar(AgeGenre_df,x='Genre',y='Counts',color='Age')
        st.plotly_chart(po2)


    

if nav == "Prediction":
    st.subheader("""
     Sales Prediction App
    This app predicts the **Sales**!
    """)
    st.write('---')
    X = Sony_Data.drop(columns=[  'Album No.',"Age" ,'Piracy','Country', 'Gender', 'Genre', 'Language',"Top20 " , 'Price ', 'StreamingServices',"Sales ","Year Released"])

    y = Sony_Data["Sales "].copy()

    st.sidebar.header('Specify Input Parameters')


    def user_input_features():
        Adverts = st.sidebar.slider('Adverts ', float(X["Adverts "].min()), float(X["Adverts "].max()), float(X["Adverts "].mean()))
        Airplay = st.sidebar.slider('Airplay', float(X.Airplay.min()), float(X.Airplay.max()), float(X.Airplay.mean()))
        No_Previous_Albums = st.sidebar.slider('No. Previous Albums', float(X["No. Previous Albums"].min()), float(X["No. Previous Albums"].max()), 
        float(X["No. Previous Albums"].mean()))
        iTunes_Download = st.sidebar.slider('iTunes Download', float(X["iTunes Download"].min()), float(X["iTunes Download"].max()), float(X["iTunes Download"].mean()))
        Attract= st.sidebar.slider('Attract', float(X.Attract.min()), float(X.Attract.max()), float(X.Attract.mean()))
        
        data = {'Adverts ': Adverts,
            'Airplay': Airplay,
            'No. Previous Albums': No_Previous_Albums,
            "iTunes Download": iTunes_Download,
            "Attract": Attract,
            }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

# Main Panel

# Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')

# Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, y)
# Apply Model to Make Prediction
    prediction = model.predict(df)

    if st.button("Click to Predict Sales"):
        st.success(f" Predicted sales is {prediction}")
    if st.button("Show Feature Importance"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.header('Feature Importance')
        plt.title('Feature importance based on SHAP values')
        shap.summary_plot(shap_values, X)
        st.pyplot(bbox_inches='tight')
        st.write('---')



        plt.title('Feature importance based on SHAP values (Bar)')
        shap.summary_plot(shap_values, X, plot_type="bar")
        st.pyplot(bbox_inches='tight')

