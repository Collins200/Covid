from nbformat import write
from numpy import cov
import streamlit as st
st.set_page_config(layout = "wide")
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
sns.set(style='white',color_codes=True)

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from scipy.optimize import minimize
import statsmodels.tsa.api as smt
import statsmodels.api as sm

from tqdm import tqdm_notebook
from tqdm.notebook import tqdm


from itertools import product
from streamlit_option_menu import option_menu
# import streamlit.components.v1 as html
import streamlit.components.v1 as components
from  PIL import Image


with st.sidebar:
    choose = option_menu("App Gallery", ["Home", "Project Analysis", "Inferences", "Contact"],
                        icons=['house', 'kanban', 'book','person lines fill'],
                        menu_icon="app-indicator", default_index=0,
                        styles={
        "container": {"padding": "5!important", "background-color": "#696969"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
    )
image = Image.open('data/corona_image.jpg')
new_image=image.resize((900, 400))
covid= Image.open('data/analytics.png')

relation= Image.open('data/relation.png')
enquiry=Image.open('data/enquiry.png')

modelling=Image.open('data/modelling.png')
if choose == "Home":

    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Covid-19 Analysis For Predictive Analytics</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
        st.image(covid, width=130 )
    st.write("Coronavirus disease is an infectious disease caused by the SARS-CoV-2 virus.Most people infected with the virus will experience mild to moderate respiratory illness and recover without requiring special treatment. However, some will become seriously ill and require medical attention. Older people and those with underlying medical conditions like cardiovascular disease, diabetes, chronic respiratory disease, or cancer are more likely to develop serious illness. Anyone can get sick with COVID-19 and become seriously ill or die at any age.")
    st.image(new_image,use_column_width =True, caption = "Covid worldwide statistics")
    st.markdown("<h1 style='text-align: center; font-family: Cooper Black; color: white;'>Project objective:</h1>", unsafe_allow_html=True)
    st.write("1. Showing the relationship between Covid 19 and other related diseases.")
    st.write("2. Showing Gender vulnerability percentage to covid 19 and other related diseases.")
    st.write("3. Showcasing Covid 19 infection rate per age group to find the most vulnerable category.")
    st.write("4. Plotting statistical time series analysis of covid 19 overtime.")


elif choose == "Project Analysis":
    # st.markdown(""" <style> .font {
    # font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    # </style> """, unsafe_allow_html=True)
    # st.markdown('<p class="font">Showcasing various relations</p>', unsafe_allow_html=True)
    col10, col20 = st.columns( [0.8, 0.2])
    with col10:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Showcasing Existing Elements Relationships</p>', unsafe_allow_html=True)    
    with col20:               # To display brand log
        st.image(relation, width=130 )

    st.markdown("<p style='text-align: center; font-family: monospace monospace; font-size:18px; color: yellow;'>Data story from charts below-interpretation of charts.</p>", unsafe_allow_html=True)
   
    # st.text("a) Male are more vulnerable to Covid,Pneumonia and Influenza disease compared to female.")
    # st.text("b) The higher the number of years the more likelihood of getting the disease.")
    # st.text("c) Between 0-19 years there is a constant infection rate over time to mean minimal exposure to covid and other related diseases.")
    
    components.html(
    """
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
    <div class="card" style="width:100%" class="p-3 mb-2 bg-secondary text-white">
    <ul class="card-text">
        <li class="">Male are more vulnerable to Covid,Pneumonia and Influenza disease compared to female.</li>
        <li class="">The higher the number of years the more likelihood of getting the disease.</li>
        <li class="">Between 0-19 years there is a constant infection rate over time to mean minimal exposure to covid and other related diseases.</li>
    </ul>
    </div>
    <style>
    .card {
    border-radius: 5px;
    background: #395B64;
    padding: 20px;
    width: 200px;
    height: 300px;
    color:white;
    
    }
    .card-text { height: 300px; overflow-y: scroll; width: 100%; }
    </style>
        """
    )


    st.markdown("<p style='font-family: Cooper Black;  font-size:25px; color: #FF9693;'>1. Download and view dataset</p>", unsafe_allow_html=True)

    st.write("The data sources for these analysis were gotten from two sources.Press the following download buttons to download the files.")
    components.html(
        """
        <!-- Add icon library -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
        
        <!-- Auto width -->
        <a href="https://hub.mph.in.gov/dataset/6bcfb11c-6b9e-44b2-be7f-a2910d28949a/resource/cb146210-5e7b-4427-82d6-6da8c08dbb1d/download/covid_report_death_date_agegrp_dictionary.xlsx">
        <button class="btn" href="https://data.cdc.gov/api/views/3apk-4u4f/rows.csv?accessType=DOWNLOAD"><i class="fa fa-download"></i> Download File 1</button>
        </a>
        <p style="color:white">and</p>
        <a href="https://data.cdc.gov/api/views/3apk-4u4f/rows.csv?accessType=DOWNLOAD">
        <button class="btn" href="https://data.cdc.gov/api/views/3apk-4u4f/rows.csv?accessType=DOWNLOAD"><i class="fa fa-download"></i> Download File 2</button>
        </a>
        <style>
        /* Style buttons */
        .btn {
        background-color: DodgerBlue;
        border: none;
        color: white;
        padding: 12px 30px;
        cursor: pointer;
        font-size: 20px;
        }

        /* Darker background on mouse-over */
        .btn:hover {
        background-color: RoyalBlue;
        }
        </style>
        
        """
    )

    # @st.cache(allow_output_mutation=True)
    def load_data(filename):
        covid_data=pd.read_csv(filename)
        return covid_data

    

    covid_data=load_data('data/covid.csv')
    covid_data.rename(columns = {'_id':'id', 'date':'date', 'agegrp':'age_group'},inplace=True)
    covid_data['date'] = covid_data['date'].str[:-9]

    covid_data['date'] = pd.to_datetime(covid_data['date'])

    # st.write(covid_data.head(5))
    
    # st.subheader('Data exploratory and cleaning')
    # nRow, nCol = covid_data.shape
    # st.write('* **Shape of our data is :** ', nRow, nCol )
    # summary=covid_data.describe()
    # st.write('* **Statistical summary :** ', summary)
    # a=covid_data.isnull().sum()
    # st.write('* **Checking for null values** ', a)
    # w=covid_data['age_group'].unique()
    

    st.markdown("<p style='font-family: Cooper Black;  font-size:25px; color: #FF9693;'>2. Showcasing covid time series analysis across different age group </p>", unsafe_allow_html=True)
  

    covid_data.drop("id", axis=1, inplace=True)

    covid_data.to_csv('data/cleaned_data.csv',index=False)

    dat=pd.read_csv('data/cleaned_data.csv')
    dat['date']= pd.to_datetime(dat['date'])
    dat.to_csv('data/cleaned_data.csv',index=False)
    data=pd.read_csv('data/cleaned_data.csv',index_col=['date'], parse_dates=['date'])

    group1 = data.loc[data['age_group'] == '0-19']
    group2 = data.loc[data['age_group'] == '20-29']
    group3 = data.loc[data['age_group'] == '30-39']
    group4 = data.loc[data['age_group'] == '40-49']
    group5 = data.loc[data['age_group'] == '50-59']
    group6 = data.loc[data['age_group'] == '60-69']
    group7 = data.loc[data['age_group'] == '70-79']
    group8 = data.loc[data['age_group'] == '80+']

    a=plt.figure(figsize=(17, 8))
    plt.plot(group1.covid_deaths)
    plt.title('Infection Rate in 0-19 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
       

    b=plt.figure(figsize=(17, 8))
    plt.plot(group2.covid_deaths)
    plt.title('Infection Rate in 20-29 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
      

    c=plt.figure(figsize=(17, 8))
    plt.plot(group3.covid_deaths)
    plt.title('Infection Rate in 30-39 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
      

    d=plt.figure(figsize=(17, 8))
    plt.plot(group4.covid_deaths)
    plt.title('Infection Rate in 40-49 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)


    e=plt.figure(figsize=(17, 8))
    plt.plot(group5.covid_deaths)
    plt.title('Infection Rate in 50-59 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
    

    f=plt.figure(figsize=(17, 8))
    plt.plot(group6.covid_deaths)
    plt.title('Infection Rate in 60-69 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
  

    g=plt.figure(figsize=(17, 8))
    plt.plot(group7.covid_deaths)
    plt.title('Infection Rate in 70-79 Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
  

    h=plt.figure(figsize=(17, 8))
    plt.plot(group8.covid_deaths)
    plt.title('Infection Rate in 80+ Years')
    plt.ylabel('Number of Infection')
    plt.xlabel('Period')
    plt.grid(False)
       
    st.write("Infection rate between 0-19 years ")
    st.pyplot(a)
    st.write("Infection rate in 20-29 years")
    st.pyplot(b)
    st.write("Infection rate in 30-39 years")
    st.pyplot(c)
    st.write("Infection rate in 40-49 years")
    st.pyplot(d)
    st.write("Infection rate in 50-59 years")
    st.pyplot(e)
    st.write("Infection rate in 60-69 years")
    st.pyplot(f)
    st.write("Infection rate in 70-79 years")
    st.pyplot(g)
    st.write("Infection rate in 80+ years")
    st.pyplot(h)

    st.markdown("<p style='font-family: Cooper Black;  font-size:25px; color: #FF9693;'>3. Covid Relationship With Other Diseases </p>", unsafe_allow_html=True)
  

    df=pd.read_csv("data/Provisional_COVID-19_Deaths_by_Sex_and_Age.csv")
    df['End Date']=pd.to_datetime(df['End Date'])
    df['Start Date']=pd.to_datetime(df['Start Date'])
    df['Data As Of']=pd.to_datetime(df['Data As Of'])
    for col in df.select_dtypes(include=['datetime64']).columns.tolist():
        df.style.format({"df[col]":
                lambda t:t.strftime("%Y-%m-%d")})
    df['Year']=df['Year'].fillna(2020)
    df. drop(["Month","Footnote"], axis=1, inplace=True)
    df=df.dropna()
    Roww, Coll = df.shape
    # st.write('dataset 2 shape: ', Roww, Coll)
    df.index=df['End Date']

    df=df[df['Age Group'] !='All Ages']
    df.reset_index(drop=True)
    df=df[['Year','Sex','Age Group', 'COVID-19 Deaths', 'Pneumonia Deaths', 'Influenza Deaths']]

    jj=sns.lmplot('Pneumonia Deaths','COVID-19 Deaths',data=df,fit_reg=True,scatter_kws={'color':'red','marker':"D","s":20})
    plt.title("Relationship between Covid 19 and Pneumonia")
    # st.pyplot(jj)


    mm=sns.lmplot('Influenza Deaths','COVID-19 Deaths',data=df,fit_reg=True,scatter_kws={'color':'red','marker':"D","s":20})
    plt.title("Relationship between Covid 19 and Influenza")
    # st.pyplot(mm)

    nn=sns.lmplot('Influenza Deaths','Pneumonia Deaths',data=df,fit_reg=True,scatter_kws={'color':'red','marker':"D","s":20})
    plt.title("Relationship between Pneumonia and Influenza")
    # st.pyplot(nn)

    df=df[df['Age Group'] !='Under 1 year']
    df=df[df['Age Group'] !='0-17 years']
    df=df[df['Age Group'] !='18-29 years']
    df=df[df['Age Group'] !='30-39 years']
    df=df[df['Age Group'] !='40-49 years']

    # Finding the most affected Age Group towards Covid 19
    df.reset_index(drop=True)
    Group_1=df['COVID-19 Deaths'][df['Age Group']=='1-4 years'].to_list()
    Group_2=df['COVID-19 Deaths'][df['Age Group']=='5-14 years'].to_list()
    Group_3=df['COVID-19 Deaths'][df['Age Group']=='15-24 years'].to_list()
    Group_4=df['COVID-19 Deaths'][df['Age Group']=='25-34 years'].to_list()
    Group_5=df['COVID-19 Deaths'][df['Age Group']=='35-44 years'].to_list()
    Group_6=df['COVID-19 Deaths'][df['Age Group']=='45-54 years'].to_list()
    Group_7=df['COVID-19 Deaths'][df['Age Group']=='55-64 years'].to_list()
    Group_8=df['COVID-19 Deaths'][df['Age Group']=='65-74 years'].to_list()
    Group_9=df['COVID-19 Deaths'][df['Age Group']=='75-84 years'].to_list()
    Group_10=df['COVID-19 Deaths'][df['Age Group']=='85 years and over'].to_list()

    Infection_rate={'1-4':sum(Group_1),'5-14':sum(Group_2),'15-24':sum(Group_3),'25-34':sum(Group_4),'35-44':sum(Group_5),'45-54':sum(Group_6),'55-64':sum(Group_7),'65-74':sum(Group_8),'75-84':sum(Group_9),'Over 85':sum(Group_10)}
    names=list(Infection_rate.keys())
    values=list(Infection_rate.values())

    vv=plt.figure(figsize=(12, 8))
    plt.bar(range(len(Infection_rate)),values,tick_label=names)
    plt.xlabel('Age group{Years}')
    plt.ylabel('Number of Infections')
    plt.title("Covid Infection Rate in various Age group categories")
   

    df.to_csv('data/provisional_data.csv',index=False)
    provisional_data=pd.read_csv('data/provisional_data.csv',index_col=['Year'],parse_dates=['Year'])
    provisional_data.rename(columns = {'COVID-19 Deaths':'COVID_Deaths', 'Pneumonia Deaths':'Pneumonia_Deaths','Influenza Deaths':'Influenza_Deaths'}, inplace = True)

    # Analysis of infection rate per Gender
    Male_Covid=provisional_data['COVID_Deaths'][provisional_data['Sex']=='Male'].to_list()
    Female_Covid=provisional_data['COVID_Deaths'][provisional_data['Sex']=='Female'].to_list()
    Female_Pneumonia=provisional_data['Pneumonia_Deaths'][provisional_data['Sex']=='Female'].to_list()
    Male_Pneumonia=provisional_data['Pneumonia_Deaths'][provisional_data['Sex']=='Male'].to_list()
    Female_Influenza=provisional_data['Influenza_Deaths'][provisional_data['Sex']=='Female'].to_list()
    Male_Influenza=provisional_data['Influenza_Deaths'][provisional_data['Sex']=='Male'].to_list()

    Gender_Infection_rate={'F_Covid':sum(Female_Covid),'M_Covid':sum(Male_Covid),'F_Pneum..':sum(Female_Pneumonia),'M_Pneum..':sum(Male_Pneumonia),'F_Influenza':sum(Female_Influenza),'M_Influenza':sum(Male_Influenza)}
    names=list(Gender_Infection_rate.keys())
    values=list(Gender_Infection_rate.values())

    zz=plt.figure()
    plt.bar(range(len(Gender_Infection_rate)),values,tick_label=names,color=['black', 'red', 'green', 'blue', 'cyan','pink'],width=0.3)
    plt.xlabel('Gender')
    plt.ylabel('Number of Infections')
    plt.title("Analysis of infection rate per Gender")


    columnn1, columnn2 = st.columns( [0.5, 0.5])
    with columnn1:  
        st.write("Covid relationship with pneumonia disease")
        st.pyplot(jj)
    with columnn2:  
        st.write("Covid relationship with influenza disease")
        st.pyplot(mm)

    columnn3, columnn4 = st.columns( [0.5, 0.5])
    with columnn3:  
        st.write("Finding the most affected age category")
        st.pyplot(vv)
    with columnn4:  
        st.write("Gender disperity of infection")
        st.pyplot(zz)

elif choose=="Inferences":

        col15, col25 = st.columns( [0.8, 0.2])
        with col15:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Model Training</p>', unsafe_allow_html=True)    
        with col25:               # To display brand log
            st.image(modelling, width=130 )

        # @st.cache(allow_output_mutation=True)
        def load_data(filename):
            covid_data=pd.read_csv(filename)
            return covid_data

        

        covid_data=load_data('data/covid.csv')
        covid_data.rename(columns = {'_id':'id', 'date':'date', 'agegrp':'age_group'},inplace=True)
        covid_data['date'] = covid_data['date'].str[:-9]

        covid_data['date'] = pd.to_datetime(covid_data['date'])

        # st.write(covid_data.head(5))
        
        # st.subheader('Data exploratory and cleaning')
        # nRow, nCol = covid_data.shape
        # st.write('* **Shape of our data is :** ', nRow, nCol )
        # summary=covid_data.describe()
        # st.write('* **Statistical summary :** ', summary)
        # a=covid_data.isnull().sum()
        # st.write('* **Checking for null values** ', a)
        # w=covid_data['age_group'].unique()
        

        st.markdown("<p style='font-family: Cooper Black;  font-size:25px; color: #FF9693;'>Brief Explanation </p>", unsafe_allow_html=True)
        st.write("A time series is simply a series of data points ordered in time. In a time series, time is often the independent variable and the goal is usually to make a forecast for the future.")
        st.write("Autocorrelation is the similarity between observations as a function of the time lag between them.")
        st.write("Seasonality refers to periodic fluctuations")
        st.write("A time series is said to be stationary if its statistical properties do not change over time. In other words, it has constant mean and variance, and covariance is independent of time.")
        st.write("Dickey-Fuller. This is the statistical test that we run to determine if a time series is stationary or not. If p > 0, and the process is not stationary and if p=0,the process is stationary")

        st.write("Moving average model simply states that the next observation is the mean of all past observations.The green line smoothed the time series")

        covid_data.drop("id", axis=1, inplace=True)

        covid_data.to_csv('data/cleaned_data.csv',index=False)

        dat=pd.read_csv('data/cleaned_data.csv')
        dat['date']= pd.to_datetime(dat['date'])
        dat.to_csv('data/cleaned_data.csv',index=False)
        data=pd.read_csv('data/cleaned_data.csv',index_col=['date'], parse_dates=['date'])

        group1 = data.loc[data['age_group'] == '0-19']
        group2 = data.loc[data['age_group'] == '20-29']
        group3 = data.loc[data['age_group'] == '30-39']
        group4 = data.loc[data['age_group'] == '40-49']
        group5 = data.loc[data['age_group'] == '50-59']
        group6 = data.loc[data['age_group'] == '60-69']
        group7 = data.loc[data['age_group'] == '70-79']
        group8 = data.loc[data['age_group'] == '80+']


        def plot_moving_average(series, window, plot_intervals=False, scale=1.96):
            rolling_mean = series.rolling(window=window).mean()
        
            aa=plt.figure(figsize=(12,8))
            plt.title('Moving average\n window size = {}'.format(window))
            plt.plot(rolling_mean, 'g', label='Rolling mean trend')
            
            #Plot confidence intervals for smoothed values
            if plot_intervals:
                mae = mean_absolute_error(series[window:], rolling_mean[window:])
                deviation = np.std(series[window:] - rolling_mean[window:])
                lower_bound = rolling_mean - (mae + scale * deviation)
                upper_bound = rolling_mean + (mae + scale * deviation)
                plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
                plt.plot(lower_bound, 'r--')
                    
            plt.plot(series[window:], label='Actual values')
            plt.legend(loc='best')
            plt.grid(True)
            st.pyplot(aa)

        
        # #Smooth by the previous 5 days (by week)
        # plot_moving_average(group6.covid_deaths, 5)

        #Smooth by the previous month (30 days)
        plot_moving_average(group6.covid_deaths, 30)

        #Smooth by previous quarter (90 days)
        plot_moving_average(group6.covid_deaths, 60, plot_intervals=True)
            
        st.write("Using Exponential smoothening")
        st.markdown('* Determines how fast the weight decreases from previous observations.The smaller the smoothing factor, the smoother the time series will be.')
        def exponential_smoothing(series, alpha):

            result = [series[0]] # first value is same as series
            for n in range(1, len(series)):
                result.append(alpha * series[n] + (1 - alpha) * result[n-1])
            return result
        
        def plot_exponential_smoothing(series, alphas):
        
            bb=plt.figure(figsize=(12, 8))
            for alpha in alphas:
                plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))
            plt.plot(series.values, "c", label = "Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Exponential Smoothing")
            plt.grid(True)
            st.pyplot(bb)

        plot_exponential_smoothing(group6.covid_deaths, [0.05, 0.2])

        def double_exponential_smoothing(series, alpha, beta):

            result = [series[0]]
            for n in range(1, len(series)+1):
                if n == 1:
                    level, trend = series[0], series[1] - series[0]
                if n >= len(series): # forecasting
                    value = result[-1]
                else:
                    value = series[n]
                last_level, level = level, alpha * value + (1 - alpha) * (level + trend)
                trend = beta * (level - last_level) + (1 - beta) * trend
                result.append(level + trend)
            return result

        def plot_double_exponential_smoothing(series, alphas, betas):
            
            cc=plt.figure(figsize=(17, 8))
            for alpha in alphas:
                for beta in betas:
                    plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))
            plt.plot(series.values, label = "Actual")
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Double Exponential Smoothing")
            plt.grid(True)
            st.pyplot(cc)

            
        plot_double_exponential_smoothing(group6.covid_deaths, alphas=[0.9, 0.02], betas=[0.9, 0.02])

        st.subheader("USING SARIMA MODEL")
        st.write("Seasonal autoregressive integraded moving average model (SARIMA) is the combination of simpler models to make a complex model that can model time series exhibiting non-stationary properties and seasonality.At first, we have the autoregression model AR(p). We assume that the current value depends on its previous values with some lag. It takes a parameter p which represents the maximum lag. To find it, we look at the partial autocorrelation plot and identify the lag.")
        def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
            
            if not isinstance(y, pd.Series):
                y = pd.Series(y)
                
            with plt.style.context(style='bmh'):
                fig = plt.figure(figsize=figsize)
                layout = (2,2)
                ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
                acf_ax = plt.subplot2grid(layout, (1,0))
                pacf_ax = plt.subplot2grid(layout, (1,1))
                
                y.plot(ax=ts_ax)
                p_value = sm.tsa.stattools.adfuller(y)[1]
                ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
                smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
                smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
                plt.tight_layout()
                st.pyplot(fig)
        tsplot(group6.covid_deaths, lags=30)

        # Take the first difference to remove to make the process stationary
        data_diff = group6.covid_deaths - group6.covid_deaths.shift(1)

        tsplot(data_diff[1:], lags=30)

        import warnings
        warnings.filterwarnings("ignore",category=FutureWarning)
        #Set initial values and some bounds
        ps = range(0, 5)
        d = 1
        qs = range(0, 5)
        Ps = range(0, 5)
        D = 1
        Qs = range(0, 5)
        s = 5

        #Create a list with all possible combinations of parameters
        parameters = product(ps, qs, Ps, Qs)
        parameters_list = list(parameters)
        len(parameters_list)

        # Train many SARIMA models to find the best set of parameters
        def optimize_SARIMA(parameters_list, d, D, s):
            """
                Return dataframe with parameters and corresponding AIC
                
                parameters_list - list with (p, q, P, Q) tuples
                d - integration order
                D - seasonal integration order
                s - length of season
            """
            
            results = []
            best_aic = float('inf')
            
            for param in tqdm_notebook(parameters_list):
                try: model = sm.tsa.statespace.SARIMAX(group6.covid_deaths, order=(param[0], d, param[1]),
                                                        seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
                except:
                    continue
                    
                aic = model.aic
                
                #Save best model, AIC and parameters
                if aic < best_aic:
                    best_model = model
                    best_aic = aic
                    best_param = param
                results.append([param, model.aic])
                
            result_table = pd.DataFrame(results)
            result_table.columns = ['parameters', 'aic']
            #Sort in ascending order, lower AIC is better
            result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
            
            return result_table

        # result_table = optimize_SARIMA(parameters_list, d, D, s)

        #Set parameters that give the lowest AIC (Akaike Information Criteria)
        # p, q, P, Q = result_table.parameters[0]

        best_model = sm.tsa.statespace.SARIMAX(group6.covid_deaths, order=(1, 1, 1),
                                            seasonal_order=(1, 1, 1, 7)).fit(disp=-1)

        st.write("Summary of the best trained model is as follows:")
        st.write(best_model.summary())
        st.subheader('Finding Correlation between different diseases')

        provisional_data=pd.read_csv('data/provisional_data.csv',index_col=['Year'],parse_dates=['Year'])
        provisional_data.rename(columns = {'COVID-19 Deaths':'COVID_Deaths', 'Pneumonia Deaths':'Pneumonia_Deaths','Influenza Deaths':'Influenza_Deaths'}, inplace = True)

        # The correlation between Covid 19 and Pneumonia
        correlation1=provisional_data['COVID_Deaths']. corr(provisional_data['Pneumonia_Deaths'])
        st.write('The correlation between Covid 19 and Pneumonia',correlation1) 

        # The correlation between Covid 19 and Influenza
        correlation2=provisional_data['COVID_Deaths']. corr(provisional_data['Influenza_Deaths'])
        st.write('The correlation between Covid 19 and Influenza',correlation2) 

        # The correlation between Pneumonia and Influenza Disease
        correlation3=provisional_data['Pneumonia_Deaths']. corr(provisional_data['Influenza_Deaths'])
        st.write('The correlation between Pneumonia and Influenza Disease',correlation3) 

        st.markdown("<p style='text-align: center; font-family: monospace monospace; font-size:18px; color: yellow;'>KEY TAKE AWAYS: Person with Pneumonia is more vulnerable to Covid 19 compared to person with Influenza</p>", unsafe_allow_html=True)


elif choose == "Contact":
    # st.markdown(""" <style> .font {
    # font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    # </style> """, unsafe_allow_html=True)
    # st.markdown('<p class="font">Enquiry Contact Form</p>', unsafe_allow_html=True)

    col115, col225 = st.columns( [0.8, 0.2])
    with col115:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Enquiry Contact Form</p>', unsafe_allow_html=True)    
    with col225:               # To display brand log
        st.image(enquiry, width=130 )

    with st.form(key='columns_in_form2',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
        #st.write('Please help us improve!')
        Name=st.text_input(label='Please Enter Your Name') #Collect user feedback
        Email=st.text_input(label='Please Enter Email') #Collect user feedback
        Message=st.text_input(label='Please Enter Your Message') #Collect user feedback
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write('Thanks for your contacting us. We will respond to your questions or inquiries as soon as possible!')


