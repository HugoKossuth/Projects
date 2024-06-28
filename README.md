# Data Science Portfolio

Welcome to my Data Science Portfolio! This repository contains various data science projects I have worked on, showcasing my skills in data analysis, machine learning, data visualization, and more.

## Introduction
This repository showcases a collection of data science projects I've worked on. These projects cover a wide range of topics, including data cleaning, exploratory data analysis, machine learning, and more.

## Projects

### Project 1: [Brain Tumor Classification](https://github.com/HugoKossuth/Projects/tree/main/Brain%20Tumor%20Classification)
**Description:** This project involves building and evaluating an image classification model to detect brain tumors using deep learning techniques. The notebook explores two different pretrained models, MobileNetV2 and VGG16, to classify images into two categories: presence or absence of brain tumors. The project includes data preprocessing, model building, training, evaluation, and a comparative analysis of model performance.

**Technologies Used:**
  * Pandas
  * Numpy
  * Tensorflow
  * Scikit-learn
  * Matplotlib

**Highlights:**
  * Model Implementation and Training:
    * Successfully implemented and trained two pretrained models, MobileNetV2 and VGG16, for brain tumor classification, demonstrating the application of transfer learning in medical image analysis.
      
  * Performance Evaluation:
    * Achieved a high accuracy of 94% with MobileNetV2, outperforming VGG16 which achieved an accuracy of 81%. The lower validation loss of MobileNetV2 (0.13) compared to VGG16 (0.26) indicates better generalization to unseen data.

### Project 2: [Credit Risk Analysis](https://github.com/HugoKossuth/Projects/tree/main/Credit%20Risk%20Analysis)
**Description:** This project involves a comprehensive analysis of credit risk using various machine learning models. The goal is to predict the likelihood of loan default based on borrower characteristics and loan attributes. The notebook includes data preprocessing, model implementation, training, evaluation, and comparative analysis of multiple classifiers.

**Technologies Used:** 
  * Pandas
  * Numpy
  * Scikit-learn
  * Matplotlib
  * Seaborn
  * XGBoost

**Highlights:**
  * Data Preprocessing and Feature Engineering:
    * Successfully preprocessed the dataset by encoding categorical variables using OneHotEncoder and normalizing numerical features with MinMaxScaler. This ensured that the data was in the optimal format for model training.
      
  * Implementation of Multiple Classifiers:
    * Implemented and compared the performance of several machine learning classifiers, including Logistic Regression, Random Forest, XGBoost, Decision Tree, Gradient Boosting, AdaBoost, and Extra Trees. This allowed for a comprehensive analysis of model effectiveness in predicting credit risk.
    
  * Performance Evaluation and Model Selection:
    * Conducted a thorough evaluation of each model's performance using accuracy scores and classification reports. This enabled the identification of the most effective model for credit risk prediction.

### Project 3: [Restaurant Revenue Prediction](https://github.com/HugoKossuth/Projects/tree/main/Restaurant%20Revenue%20Prediction)
**Description:** This project aim is to predict the revenue of various restaurants based on several attributes. The dataset includes features such as location, cuisine type, ratings, seating capacity, marketing budget, and more. The notebook applies various machine learning techniques and data analysis to achieve accurate revenue predictions and daa insights.

**Technologies Used:** 
  * Pandas
  * Numpy
  * Scikit-learn
  * Matplotlib
  * Seaborn
  * XGBoost

**Highlights:**
  * Data visualization:
      * Created various charts by the use of matplotlib and seaborn for identifying data outliers, anomalies, class imbalancement and correlation between the given atributes of the data set, enhancing the models accuracy.

  * Data Preprocessing and Feature Engineering:
    * Successfully handled missing values, encoded categorical variables, and scaled numerical features to prepare the dataset for model training. The applied techniques are OneHotEncoding and StandardScaler to ensure that the features were appropriately transformed for model input.
   
  * Performance Evaluation and Model Selection:
    * Conducted a thorough evaluation of each model's performance using accuracy scores and visualized the predicted values comapred with the actual values. This enabled the identification of the most effective model for revenue prediction.

### Project 4: [Stock Market Analysis](https://github.com/HugoKossuth/Projects/tree/main/Stock%20Market%20Analysis)
**Description:** This notebook focuses on analyzing and visualizing stock market data by matplotlib to understand price trends and make predictions. Using data from the Alpha Vantage API, it includes steps for data fetching, preprocessing, visualization, and implementing predictive models.

**Technologies Used:** 
  * Pandas
  * Numpy
  * Scikit-learn
  * Matplotlib
  * Seaborn
  * Alpha Vantage API

**Highlights:**
  * Data Collection and Preprocessing:
    * Successfully retrieved stock market data for a specific company (e.g., Apple Inc.) using the Alpha Vantage API. Preprocessed the data by converting timestamps, and transforming string values to numerical formats for analysis.

  * Data Visualization and Exploration:
    * Created informative visualizations such as line plots of closing prices over time and histograms of daily percentage changes to explore stock price trends and volatility. Developed advanced visualizations and indicators like candlestick charts, SMA, EMA, VWAP, MACD, Signal line, and RSI.

  * Predictive Modeling:
    * Implemented regression models including Linear Regression, Ridge, Lasso to predict future stock prices.
Evaluated model performance using appropriate metrics and visualizations to determine the accuracy and reliability of the predictions.
