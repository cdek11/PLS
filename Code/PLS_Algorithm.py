
# coding: utf-8

# In[ ]:

# Code to implement the initial version of the PLS Algorithm

import pandas as pd
import numpy as np

def pls(path, path_test, predictors, response):
    '''Function that takes a dataframe and runs partial least squares on numeric predictors for a numeric response.
    Returns the residuals of the predictor (X block), response (Y block), and traininig RMSE'''
    combined = predictors #Ready list to combine the predictors and response to get both sets of data
    ###Data preparation
    data = pd.DataFrame.from_csv(path) #Retrieve full csv data from local machine
    combined.append(response) #Add the response to the list of variables to get from data set
    data = data[combined] #Only retrieve the predictors and response
    response_std = data[response].std() #Store the response variable standard deviation to scale the RMSE to real units at end
    
    #Subtract the mean from each column
    data = data - data.mean()

    #Scale each column by the column standard deviation
    data = data/data.std()

    #Separate in to design matrix (X block) and response column vector (Y block)
    predictors.pop() #Remove the response variable from the predictors list
    X = data[predictors].as_matrix() #Create a matrix of predictor values
    Y = data[[response]].as_matrix() #Create a matrix of predictor values
    Y_true = Y #Store the true Y values for prediction later
    
    #Get rank of matrix
    rank = np.linalg.matrix_rank(X) #Store rank of matrix because this is the maximum number of components the model can have
    
    #PLS algorithm
    u = Y #Set intital u value as response variables
    Xres_dictionary = {} #Create a dictionary for the residuals from the decomposition of the X block
    Yres_dictionary = {} #Create a dictionary for the residuals from the decomposition of the Y block
    q_new_dictionary ={} #Create a dictionary for row vectors of q loadings for the Y block
    b_dictionary = {} #Create a dictionary for scalar regression coefficient for PLS components 
    t_hat_dictionary = {} #Create a dictionary for the matrix of X scores 
    t_hat_train_dictionary = {} #Create a dictionary for the matrix of X scores for training data
    t_hat_test_dictionary = {} #Create a dictionary for the matrix of X scores for test data
    RMSE_dictionary = {} #Create a dictionary to store RMSE for training data
    RMSE_test_dictionary = {} #Create a dictionary to store RMSE for test data
    for i in range(1,(rank+1)):
        Y_pred = np.zeros((Y_true.shape[0],1))
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
        F_h = Y - b.dot(t_new.T).T.dot(q)
        
        #Set outer relation for the X block
        Xres_dictionary[i] = E_h
        X = E_h
        
        #Set the mixed relation for the Y block
        Yres_dictionary[i] = F_h
        Y = F_h
        
        #Find estimated t hat
        t_hat = np.dot(E_h,w_new.T)
        t_hat_dictionary[i] = t_hat
        E_h = E_h - np.dot(t_hat,p_new.T)
        
        #Predict training set response by summing over different compenents
        E_h = X
        for j in range(1,i+1):
            t_hat_train = np.dot(E_h,w_new.T)
            t_hat_train_dictionary[j] = t_hat_train
            E_h = E_h - np.dot(t_hat_train, p_new.T)
            for g in range(1,i+1):
                Y_pred = Y_pred + (b_dictionary[g]*t_hat_dictionary[g]).dot(q_new_dictionary[g].T)
        
        #Find training RMSE 
        RMSE = np.sqrt(sum((Y_true - Y_pred)**2)/Y_true.shape[0]) 
        RMSE_scaled = RMSE * response_std 
        RMSE_dictionary[i] = RMSE_scaled
        
        #Code chunk to find test RMSE
        #Load data
        data_test = pd.DataFrame.from_csv(path_test)
        combined.append(response)
        data_test = data_test[combined]
        response_std_test = data_test[response].std()
    
        #Subtract the mean from each column
        data_test = data_test - data_test.mean()

        #Scale each column by the column standard deviation
        data_test = data_test/data_test.std()

        #Separate in to design matrix (X block) and response column vector (Y block)
        predictors.pop()
        X_test = data[predictors].as_matrix()
        Y_test = data[[response]].as_matrix()
        Y_true_test = Y_test #For prediction
        
        Y_pred_test = np.zeros((Y_true_test.shape[0],1)) 
        
        #Get rank of matrix
        rank_test = np.linalg.matrix_rank(X)
        
        E_h_test = X_test
        
        #Sum over different compenents
        for k in range(1,i+1):
            t_hat_test = np.dot(E_h_test,w_new.T)
            t_hat_test_dictionary[k] = t_hat_test
            E_h_test = E_h_test - np.dot(t_hat_test, p_new.T)
            Y_pred_test = Y_pred_test + (b_dictionary[k]*t_hat_test_dictionary[k]).dot(q_new_dictionary[k].T)
        
        #Find test RMSE 
        RMSE = np.sqrt(sum((Y_true_test - Y_pred_test)**2)/Y_true_test.shape[0]) 
        RMSE_scaled_test = RMSE * response_std_test # I believe this is the RMSE since the Y had to be scaled.
        RMSE_test_dictionary[i] = RMSE_scaled_test
        
    return RMSE_dictionary, RMSE_test_dictionary

