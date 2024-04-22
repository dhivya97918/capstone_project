import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Read the data
df = pd.read_excel(r"https://github.com/dhivya97918/capstone_project/raw/master/Main%20regions.xlsx")
#df=pd.read_excel(r"C:\Users\dhivyashree.a.lv\Desktop\capstone_project\Main regions.xlsx")
df['dt_iso'] = pd.to_datetime(df['dt_iso'], utc=True)

# Streamlit code starts here
st.title('Energy Load Prediction')
st.write(f"This is a predictive analytics tool designed to forecast energy demand with precision. By analyzing historical data and various influencing factors, our app helps optimize resource allocation, promote grid stability in the energy sector")
st.title('Country:Spain')
st.write(f"Reason for Selection: Spain, with its diverse geography and commitment to renewable energy, offers a unique landscape for energy prediction. With a growing emphasis on green energy and sustainable practices, Spain provides valuable data for understanding energy consumption patterns and driving future strategies.\n")
st.write(f"For our prototype, we have chosen five key regions across Spain to analyze energy consumption and weather patterns: Barcelona, Madrid, Seville, Valencia, and Bilbao. These regions represent a diverse geographical and climatic landscape, offering valuable insights into energy demand variations and weather-related influences.\n")
st.write(f"Our prototype uses historical energy data and weather data of the years 2015,2016,2017 and 2018.")

lb = LabelEncoder()
df['weather_main_lb'] = lb.fit_transform(df['weather_main'])
df['city_name_lb'] = lb.fit_transform(df['city_name'])
df['Season_lb'] = lb.fit_transform(df['Season'])
df['Time_of_day_lb'] = lb.fit_transform(df['Time of Day'])
df_lb = df.drop(columns=['weather_main', 'Season', 'Time of Day', 'city_name', 'Date'])
corr_matrix = df_lb.corr()
# Define features and target
datasets = []
for city_label in range(df_lb['city_name_lb'].nunique()):
    city_data = df_lb[df_lb['city_name_lb'] == city_label]
    X = city_data[['humidity', 'temp', 'Generation wind onshore', 'Day']]
    y = city_data['Total load actual']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    datasets.append((X_train, X_test, y_train, y_test))

# Combine data from all cities
X_combined_train = pd.concat([X_train for X_train, _, _, _ in datasets])
X_combined_test = pd.concat([X_test for _, X_test, _, _ in datasets])
y_combined_train = pd.concat([y_train for _, _, y_train, _ in datasets])
y_combined_test = pd.concat([y_test for _, _, _, y_test in datasets])

# Train XGBoost model on the combined dataset
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_combined_train, y_combined_train)

# Predict on the combined testing dataset
xgb_pred_combined = xgb_model.predict(X_combined_test)

# Calculate overall evaluation metrics
mse_combined = mean_squared_error(y_combined_test, xgb_pred_combined)
rmse_combined = mean_squared_error(y_combined_test, xgb_pred_combined, squared=False)
mae_combined = mean_absolute_error(y_combined_test, xgb_pred_combined)
r2_combined = r2_score(y_combined_test, xgb_pred_combined)
explained_variance_combined = explained_variance_score(y_combined_test, xgb_pred_combined)
mape_combined = np.mean(np.abs((y_combined_test - xgb_pred_combined) / y_combined_test)) * 100
cv_scores_combined = cross_val_score(xgb_model, pd.concat([X_combined_train, X_combined_test]), pd.concat([y_combined_train, y_combined_test]), cv=4, scoring='neg_mean_squared_error')

st.sidebar.title('Accuracy')
analysis_option = st.sidebar.button('Evaluation Metrics')
analysis_option8 = st.sidebar.checkbox('Summary Statistics')
if analysis_option8:
    st.write(df.describe())
# Display overall evaluation metrics
if analysis_option:
    st.write(f'Mean Squared Error (MSE): {mse_combined}')
    st.write(f'Root Mean Squared Error (RMSE): {rmse_combined}')
    st.write(f'Mean Absolute Error (MAE): {mae_combined}')
    st.write(f'R-squared (R^2) Score: {r2_combined}')
    st.write(f'Explained Variance Score: {explained_variance_combined}')
    cv_scores = cross_val_score(xgb_model, X, y, cv=4, scoring='neg_mean_squared_error')
    cv_scores = np.abs(cv_scores)
    st.write('## Cross-validation Scores')
    st.write(cv_scores)
    st.write('## True and Predicted Values')
    st.write('True values:')
    st.write(y_combined_test)
    st.write('Predicted values:')
    st.write(xgb_pred_combined)
analysis_option1 = st.sidebar.checkbox('Correlation Heatmap')
if analysis_option1:
    #st.subheader('Correlation Heatmap')
    #fig, ax=plt.subplots()
    #fig1 = ax.figure
    #fig1.set_size_inches(10,8)
    #st.write("Correlation Heatmap:")
    #st.write(sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f").figure)
    #st.pyplot(fig)
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8}, ax=ax)
    ax.set_title('Heatmap of Correlation')
    plt.tight_layout()
    st.pyplot(fig)
df_train=pd.read_excel(r"https://github.com/dhivya97918/capstone_project/raw/master/Train%20Data.xlsx")
#df_train=pd.read_excel(r"C:\Users\dhivyashree.a.lv\Desktop\capstone_project\Train Data.xlsx")
#df_test=pd.read_excel(r"C:\Users\dhivyashree.a.lv\Desktop\capstone_project\Test Data.xlsx")
df_test=pd.read_excel(r"https://github.com/dhivya97918/capstone_project/raw/master/Test%20Data.xlsx")
df['Temp_C'] = df['temp'] - 273.15
df_train['Temp_C']=df_train['temp']-273.15
df_test['Temp_C']=df_test['temp']-273.15


st.sidebar.title('Load Details Based on Actual and Predicted Values')
analysis_option2 = st.sidebar.checkbox("Actual Load Vs Predicted Load")
if analysis_option2:
    df_test['Date'] = pd.to_datetime(df_test['Date'])
    numerical_columns = df_test.select_dtypes(include=[np.number]).columns.tolist()
    if 'Date' in numerical_columns:
        numerical_columns.remove('Date')

    # Group by month and sum only numerical columns
    df_monthly_sum = df_test.groupby(df_test['Date'].dt.to_period('M'))[numerical_columns].sum()

    fig, ax=plt.subplots()
    fig1 = ax.figure
    fig1.set_size_inches(10, 6)
    ax.plot(df_monthly_sum.index.astype(str), df_monthly_sum['Load forecast'], label='Load forecast', color='blue')
    ax.plot(df_monthly_sum.index.astype(str), df_monthly_sum['Total load actual'], label='Total load actual', color='red')

    # Add labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Load')
    ax.set_title('Monthly Sum of Load Forecast vs Total Load Actual')
    ax.legend()

    # Show plot
    ax.grid(True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

analysis_option3 = st.sidebar.checkbox("Peak Hour Demand")
if analysis_option3:
    df_hourly_mean = df.groupby('hour')['Total load actual'].mean()

    # Find the peak and off-peak hours
    peak_hour = df_hourly_mean.idxmax()
    off_peak_hour = df_hourly_mean.idxmin()

    # Plotting
    fig, ax=plt.subplots()
    fig1 = ax.figure
    fig1.set_size_inches(10, 6)
    bars = ax.bar(df_hourly_mean.index, df_hourly_mean, color='skyblue', edgecolor='black', linewidth=1)  # Added borders

    # Highlight the bars with highest and lowest load
    for bar in bars:
        if bar.get_height() == df_hourly_mean.max():
            bar.set_color('red')
        elif bar.get_height() == df_hourly_mean.min():
            bar.set_color('green')

    legend_handles = [
        patches.Rectangle((0, 0), 1, 1, color='red'),
        patches.Rectangle((0, 0), 1, 1, color='green')
    ]

    legend_labels = [f'Peak Hour ({peak_hour}:00 - {peak_hour+1}:00)', f'Off-Peak Hour ({off_peak_hour}:00 - {off_peak_hour+1}:00)']
    ax.legend(legend_handles, legend_labels)

    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Load')
    ax.set_title('Average Load by Hour')
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)


analysis_option4 = st.sidebar.checkbox("Season Wise Average Load")
if analysis_option4:
    season_avg_load = df.groupby('Season')['Total load actual'].mean().sort_values()

    # Define custom x-axis labels for the seasons
    season_labels = ['Summer', 'Autumn', 'Spring', 'Winter']

    # Plotting
    fig, ax=plt.subplots()
    fig1 = ax.figure
    fig1.set_size_inches(10, 6)
    bars = ax.bar(season_avg_load.index, season_avg_load, color='skyblue', edgecolor='black')  # Added edgecolor

    # Highlight the highest and lowest bars
    max_load_season = season_avg_load.idxmax()
    min_load_season = season_avg_load.idxmin()

    for bar in bars:
        if bar.get_height() == season_avg_load[max_load_season]:
            bar.set_color('red')
        elif bar.get_height() == season_avg_load[min_load_season]:
            bar.set_color('green')

        # Add borders to bars
        bar.set_linewidth(1)
        bar.set_edgecolor('black')

    # Set custom x-axis labels
    ax.set_xticks(season_avg_load.index)
    ax.set_xticklabels(season_labels)

    ax.set_xlabel('Season')
    ax.set_ylabel('Average Load')
    ax.set_title('Average Load by Season')
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)

analysis_option5 = st.sidebar.checkbox("Temperature Vs Load")
if analysis_option5:
    temperature_bins = list(range(-20, 50, 10))

    # Create a new column 'Temperature Group' based on the bins
    df['Temperature Group'] = pd.cut(df['Temp_C'], bins=temperature_bins, right=False)

    # Group by 'Temperature Group' and calculate the average load
    df_grouped = df.groupby('Temperature Group')['Total load actual'].sum()

    # Plotting
    fig, ax=plt.subplots()
    fig1 = ax.figure
    fig1.set_size_inches(10, 6)
    df_grouped.plot(kind='line', marker='o', color='blue')
    ax.set_xlabel('Temperature Range(°C)')
    ax.set_ylabel('Total Load')
    ax.set_title('Total Load by Temperature Range')
    ax.grid(False)
    plt.tight_layout()
    st.pyplot(fig)


analysis_option6 = st.sidebar.checkbox("Load Vs Price")
if analysis_option6:
    min_price = df['Price'].min()
    max_price = df['Price'].max()
    price_bins = np.arange(min_price, max_price + 5, 5).tolist()
    df['Price Group'] = pd.cut(df['Price'], bins=price_bins, right=False)
    df_grouped = df.groupby('Price Group')['Total load actual'].mean()
    fig, ax=plt.subplots()
    fig1 = ax.figure
    fig1.set_size_inches(10, 6)
    df_grouped.plot(kind='line', marker='o', color='green')
    ax.set_xlabel('Price Range')
    ax.set_ylabel('Average Load')
    ax.set_title('Average Load by Price Range')
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

analysis_option7 = st.sidebar.checkbox("Comparison Between Weekdays and Weekend")
if analysis_option7:
    df['Date'] = pd.to_datetime(df['Date'])

    # Function to check if a date is a weekend
    def is_weekend(date):
        """Check if the given date is a weekend."""
        return date.weekday() >= 5  # Returns True if Saturday or Sunday, False otherwise

    # Create a new column 'DayType' to indicate weekday (0) or weekend (1)
    df['DayType'] = df['Date'].apply(lambda x: 1 if is_weekend(x) else 0)

    # Group by 'DayType' and calculate the average load
    day_type_avg_load = df.groupby('DayType')['Total load actual'].mean()

    # Plotting
    fig, ax=plt.subplots()
    fig1 = ax.figure
    fig1.set_size_inches(8, 6)
    ax.pie(day_type_avg_load, labels=['Weekday', 'Weekend'], autopct='%1.1f%%', colors=['skyblue', 'orange'],
            startangle=140, textprops={'fontsize': 12}, wedgeprops={'edgecolor': 'black'})
    ax.set_title('Average Load by Day Type')
    ax.set_aspect('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    st.pyplot(fig)
