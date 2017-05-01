
# coding: utf-8

# In[2]:

# Code to implement the optimized version of the PLS Algorithm

import pandas as pd
import numpy as np
import numba
from numba import jit

@jit
def mean_center_scale(dataframe):
    '''Scale dataframe by subtracting mean and dividing by standard deviation'''
    dataframe = dataframe - dataframe.mean()
    dataframe = dataframe/dataframe.std()
    return dataframe

@jit
def y_pred(Y_pred, i,b_dictionary,t_hat_dictionary,q_new_dictionary):
    '''Find prediction for Y based on the number of components in this iteration'''
    for j in range(1,i+1):
        Y_pred = Y_pred + (b_dictionary[j]*t_hat_dictionary[j]).dot(q_new_dictionary[j].T)
    return Y_pred    

@jit
def rmse(i,Y_true, Y_pred, response_std, RMSE_dictionary):
    '''Find training RMSE''' 
    RMSE = np.sqrt(sum((Y_true - Y_pred)**2)/Y_true.shape[0])
    RMSE_scaled = RMSE * response_std 
    RMSE_dictionary[i] = RMSE_scaled

    return RMSE_dictionary

@jit        
def core_pls(i,Y, X, q_new_dictionary, b_dictionary, t_hat_dictionary) :
    '''Core PLS algorithm'''
    
    #Here we have one variable in the Y block so q = 1 
    #and omit steps 5-8
    q = 1

    #For the X block, u = Y
    u = Y #random y column from Y #Step 1
    w_old = np.dot(u.T,X)/np.dot(u.T,u) #Step 2
    w_new = w_old/np.linalg.norm(w_old) #Step 3
    t = np.dot(X,w_new.T)/np.dot(w_new,w_new.T) #Step 4

    #For the Y block can be omitted if Y only has one variable
    q_old = np.dot(t.T,Y)/np.dot(t.T,t) #Step 5
    q_new = q_old/np.linalg.norm(q_old) #Step 6
    q_new_dictionary[i] = q_new
    u = np.dot(Y,q_new.T)/np.dot(q_new,q_new.T) #Step 7

    #Step 8: Check convergence

    #Calculate the X loadings and rescale the scores and weights accordingly
    p = np.dot(t.T,X)/np.dot(t.T,t) #Step 9
    p_new = p.T/np.linalg.norm(p.T) #Step 10
    t_new = t/np.linalg.norm(p.T) #Step 11
    w_new = w_old/np.linalg.norm(p)  #Step 12

    #Find the regression coefficient for b for th inner relation
    b = np.dot(u.T,t_new)/np.dot(t.T,t) #Step 13
    b_dictionary[i] = b

    #Calculation of the residuals
    E_h = X - np.dot(t_new,p_new.T)
    F_h = Y - b.dot(t_new.T).T.dot(q) #WORKS BUT IS THIS RIGHT?        
    
    #Set outer relation for the X block
    #Xres_dictionary[i] = E_h #MAYBE REMOVE
    X = E_h
        
    #Set the mixed relation for the Y block
    #Yres_dictionary[i] = F_h 3MAYBE REMOVE
    Y = F_h
        
    #Find estimated t hat
    t_hat = np.dot(E_h,w_new.T)
    t_hat_dictionary[i] = t_hat
    E_h = E_h - np.dot(t_hat,p_new.T)
    
    return X,Y, u, w_new, q_new, t_new, p_new, q_new_dictionary, t_hat_dictionary, b_dictionary,E_h, F_h 
          

def pls_optimized(path, path_test, predictors, response):
    '''Function that takes a dataframe and runs partial least squares on numeric predictors for a numeric response.
    Returns the residuals of the predictor (X block), response (Y block), and traininig RMSE'''
    ###TRAINING DATA
    combined = predictors
    #Load data
    data = pd.DataFrame.from_csv(path)
    combined.append(response)
    data = data[combined]
    response_std = data[response].std()
    
    #Subtract the mean and scale each column
    data = mean_center_scale(data)

    #Separate in to design matrix (X block) and response column vector (Y block)
    predictors.pop()
    X = data[predictors].as_matrix()
    Y = data[[response]].as_matrix()
    Y_true = Y #For prediction
    
    #Get rank of matrix
    rank = np.linalg.matrix_rank(X)
   
    u = Y #set initial u as Y
    Xres_dictionary = {}
    Yres_dictionary = {}
    q_new_dictionary ={}
    b_dictionary = {}
    t_hat_dictionary = {}
    t_hat_train_dictionary = {}
    t_hat_test_dictionary = {}
    RMSE_dictionary = {}
    RMSE_test_dictionary = {}
    
    ###TEST DATA
    #Load data
    data_test = pd.DataFrame.from_csv(path_test)
    combined.append(response)
    data_test = data_test[combined]
    response_std_test = data_test[response].std()
    
    #Subtract the mean and scale each column
    data_test = mean_center_scale(data_test)

    #Separate in to design matrix (X block) and response column vector (Y block)
    predictors.pop()
    X_test = data[predictors].as_matrix()
    Y_test = data[[response]].as_matrix()
    Y_true_test = Y_test #For prediction
      
    #Get rank of matrix
    rank_test = np.linalg.matrix_rank(X_test)
    
    #Iterate through each component
    for i in range(1,(rank+1)):
        Y_pred = np.zeros((Y_true.shape[0],1))
        Y_pred_test = np.zeros((Y_true_test.shape[0],1))
        
        #Core algo
        X,Y, u, w_new, q_new, t_new, p_new, q_new_dictionary, t_hat_dictionary, b_dictionary,E_h, F_h = core_pls(i,Y, X, q_new_dictionary, b_dictionary, t_hat_dictionary)
                
        #NEW Sum over different compenents
        for g in range(1,i+1):
            t_hat_train = np.dot(E_h,w_new.T)
            t_hat_train_dictionary[g] = t_hat_train
            E_h = E_h - np.dot(t_hat_train, p_new.T)
            Y_pred = y_pred(Y_pred, g,b_dictionary,t_hat_dictionary,q_new_dictionary)
        
        #Find training RMSE 
        RMSE_dictionary = rmse(i,Y_true, Y_pred, response_std, RMSE_dictionary)
        
        #Set initial E_h as X_test data
        E_h_test = X_test
        
        #Sum over different compenents
        for k in range(1,i+1):
            t_hat_test = np.dot(E_h_test,w_new.T)
            t_hat_test_dictionary[k] = t_hat_test
            E_h_test = E_h_test - np.dot(t_hat_test, p_new.T)
            Y_pred_test = y_pred(Y_pred_test, k,b_dictionary,t_hat_test_dictionary,q_new_dictionary)
        
        #Find test RMSE 
        RMSE_test_dictionary = rmse(i,Y_true_test, Y_pred_test, response_std_test, RMSE_test_dictionary)
        
    return RMSE_dictionary, RMSE_test_dictionary

