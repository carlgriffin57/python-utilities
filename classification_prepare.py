import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# from env import host, username, password

def prep_iris(df):
    '''
        This function accepts the untransformed iris data and returns the data with tranformations applied.
    '''
    df = df.drop(columns=(['species_id', 'measurement_id']))
    df = df.rename(columns = {'species_name':'species'})
    dummy_df = pd.get_dummies(df[['species']], dummy_na = False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

# ************************************ TITANIC DATA***********************************
def titanic_split(df):
    '''
    This function take in the titanic data acquired by get_titanic_data,
    performs a split and stratifies survived column.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate.survived)
    return train, validate, test



def impute_mean_age(train, validate, test):
    '''
    This function imputes the mean of the age column for
    observations with missing values.
    Returns transformed train, validate, and test df.
    '''
    # create the imputer object with mean strategy
    imputer = SimpleImputer(strategy = 'mean')
    
    # fit on and transform age column in train
    train['age'] = imputer.fit_transform(train[['age']])
    
    # transform age column in validate
    validate['age'] = imputer.transform(validate[['age']])
    
    # transform age column in test
    test['age'] = imputer.transform(test[['age']])
    
    return train, validate, test


def prep_titanic_data(df, column, method ,dummies):
    '''
    takes in a dataframe of the titanic dataset that was  acquired before and returns a cleaned dataframe
    arguments:
    - df: a pandas DataFrame with the expected feature names and columns
    - column : the name of the column to fill or impute the missing values in
    - method: type of strategy (median, mean, most_frequent) for SimpleImputer
    - dummies: list of columns to create a dummy variable 
    return: 
    train, validate, test (three dataframes with the cleaning operations performed on them)
    '''
    #clean data
    df = df.drop_duplicates()
    df = df.drop(columns=['deck', 'embark_town', 'class'])
    
    #create a dummy df
    dummy_df = pd.get_dummies(df[dummies], drop_first=[True, True])
    ## Concatenate the dummy_df dataframe above with the original df
    df = pd.concat([df, dummy_df], axis=1)
    
    # drop the deck column
    df = df.drop(columns= dummies)
    #split data
    
    # split data into train, validate, test dfs
    train, validate, test = titanic_split(df)

    # impute the chosen strategy (median)  for  the selected column (age) into null values in age column
    train, validate, test = impute_mean_age(train, validate, test)
   
    return train, validate, test
