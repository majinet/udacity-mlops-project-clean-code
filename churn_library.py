# library doc string
"""
The churn_library.py is a library of functions to find customers who are likely to churn

Author: Atauro Chow
Date: 4-Jan-2023
"""

# import libraries
import os
import sys
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth: Any) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    :param pth: a path to the csv
    :return: pandas dataframe
    """

    df = pd.read_csv(pth)
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    perform eda on df and save figures to images folder

    :param df: pandas dataframe
    :return:
    """

    # create and save Univariate, quantitative plot for Customer Age
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].plot(xlabel='Age', ylabel='Frequency', kind='hist', title='Customer Age')
    plt.savefig('./images/eda/Customer_Age.png')

    # create and save Univariate, categorical plot
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar', ylabel='Number of counts (Normalize)',
                                                     title='Martial Status')
    plt.savefig('./images/eda/Marital_Status.png')

    # create and save Bivariate plot
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].plot(xlabel='Customer type', ylabel='Frequency', kind='hist', title='Customer type')

    plt.savefig('./images/eda/Churn.png')


def encoder_helper(df: pd.DataFrame, category_lst: list, response: Any) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    :param df: pandas dataframe
    :param category_lst: list of columns that contain categorical features
    :param response: string of response name [optional argument that could be used for naming variables or index y column]
    :return: pandas dataframe with new columns for
    """

    for colname in category_lst:

        lst = []
        groups = df.groupby(colname).mean()[response]

        for val in df[colname]:
            lst.append(groups.loc[val])

        df[colname + '_' + response] = lst

    return df


def perform_feature_engineering(df: pd.DataFrame, response: Any):
    """
    Split train and test set for training

    :param df: pandas dataframe
    :param response: string of response name [optional argument that could be used for naming variables or index y column]
    :returns: X_train, X_test, y_train, y_test
    """

    # define y (label)
    y = df['Churn']

    # define x
    X = pd.DataFrame()

    # encoded column
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = encoder_helper(df, category_lst, response)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train: Any,
                                y_test: Any,
                                y_train_preds_lr: Any,
                                y_train_preds_rf: Any,
                                y_test_preds_lr: Any,
                                y_test_preds_rf: Any) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    :param:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    :returns:
             None

    """

    with open('./images/results/classification_report.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        # scores
        print('random forest results')
        print('test results')
        print(classification_report(y_test, y_test_preds_rf))
        print('train results')
        print(classification_report(y_train, y_train_preds_rf))

        print('logistic regression results')
        print('test results')
        print(classification_report(y_test, y_test_preds_lr))
        print('train results')
        print(classification_report(y_train, y_train_preds_lr))

        sys.stdout = sys.stdout


def feature_importance_plot(model: Any, X_data: pd.DataFrame, output_pth: str) -> None:
    """
    creates and stores the feature importances in pth

    :param model: model object containing feature_importances_
    :param X_data: pandas dataframe of X values
    :param output_pth: path to store the figure
    :return: None
    """

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame) -> None:
    """
    train, store model results: images + scores, and store models

    :param X_train: X training data
    :param X_test: X testing data
    :param y_train: y training data
    :param y_test: y testing data
    :return: None
    """

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # calculate model performance metrics for RandomForestClassifier and LogisticRegression
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # plots roc for RandomForestClassifier and LogisticRegression
    _, ax = plt.subplots()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8, name="RandomForestClassifier")
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8, name="LogisticRegression")
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0.2, 1)
    _ = ax.set_title('ROC curve')
    plt.show()
    plt.savefig('./images/results/roc.png')

    rfc_model = joblib.load('./models/rfc_model.pkl')

    # plots feature importance graph for RandomForestClassifier
    feature_importance_plot(
        rfc_model,
        X_train,
        './images/results/feature_importants.png')


if __name__ == "__main__":
    df_bank = import_data("./data/bank_data.csv")
    print(df_bank.head())

    perform_eda(df_bank)

    X_train_bank, X_test_bank, y_train_bank, y_test_bank = perform_feature_engineering(
        df_bank, 'Churn')

    train_models(X_train_bank, X_test_bank, y_train_bank, y_test_bank)
