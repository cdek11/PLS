
# coding: utf-8

# In[ ]:

# generate dataset with 11 predictors (x1-x11) and a response variable y, with only 10 observations in each dataset.
# variables are made to be correlated with one another and response variable doesn't rely on all variables. 
np.random.seed(9856)
x1 = np.random.normal(5, .2, 20)
x2 = np.random.normal(7, .4, 20)
x3 = np.random.normal(9, .8, 20)

sim_data = {'x1' : x1,
            'x2' : x2, 
            'x3' : x3,
            'x4' : 5*x1,
            'x5' : 2*x2,
            'x6' : 4*x3,
            'x7' : 6*x1,
            'x8' : 5*x2,
            'x9' : 4*x3,
            'x10' : 2*x1,
            'x11' : 3*x2,
            'y' : 6*x1 + 3*x2}

# convert data to csv file
pd.DataFrame(sim_data)[0:10].to_csv("sim_data_train.csv")
pd.DataFrame(sim_data)[10:20].to_csv("sim_data_test.csv")

# set variables for input to pls function
sim_predictors = pd.DataFrame(sim_data).drop("y", axis = 1).columns.tolist()
sim_response = "y"
sim_data_path = 'sim_data_train.csv'
sim_data_test_path = 'sim_data_test.csv'

# run pls regression on simulated data
pls_optimized_sim_results = pls_optimized(sim_data_path, sim_data_test_path, sim_predictors, sim_response)

