import numpy as np
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt


beta_0 = 1
beta_1 = 1
gamma = 1
sigma_0_squared = 2
alpha_0 = 0
alpha_1 = 1

def g_0(w_vec):

    return expit(alpha_0 + alpha_1 * w_vec)


def generate_data(n):

    W = np.random.randn(n).reshape(-1, 1)
    A = np.random.binomial(n = 1, p = g_0(W)).reshape(-1, 1)

    mean = beta_0 * A + beta_1 * W + gamma * A * W
    Y = np.random.normal(mean, sigma_0_squared).reshape(-1, 1)

    return W, A, Y


#type 
def estimators(type, W, A, Y):

    def est_1(): #regression

        AW = A * W
        design = np.column_stack((W, A, AW))

        model = LinearRegression()
        model.fit(design, Y)
        
        coefficients = model.coef_[0]

        return coefficients[0]
        
    def est_2(): #known g(W)
        return 1 / len(Y) * sum(A * Y / g_0(W))
        
    def est_3(): #estimate of g_n(W) (logistic regression)

        ones = np.ones((len(Y), 1))
        design = np.column_stack((ones, W))

        model = LogisticRegression()
        model.fit(design, A)

        g_n_predict = model.predict(design)
        
        return 1 / len(Y) * sum(A * Y / g_n_predict)
    
    if type == 0:
        return est_1()
    elif type == 1:
        return est_2()
    elif type == 2:
        return est_2()
    else:
        raise TypeError("Argument must be an one, two, or three")
            
        
    
def simulation_plot():

    bias, sd, mse = np.empty((3, 4)), np.empty((3, 4)), np.empty((3, 4))

    enum = [100, 250, 1000, 5000]

    for k in range(4):

        n = enum[k] 
        true_mean = np.ones((3, 1000))
        
        est_mean = np.empty((3, 1000))

        for i in range(3): #given a specific estimator 

            for trial in range(1000):

                W, A, Y =  generate_data(n)
                est_mean[i, trial] = estimators(i, W, A, Y)
        
        bias_matrix = est_mean - true_mean 

        for row in range(3):

            bias[row, k] = np.mean(bias_matrix[row, : ])
            sd[row, k] = np.std(est_mean[row, : ])
            mse[row, k] = np.mean(bias_matrix[row, : ] * bias_matrix[row, : ])
        
    
    X = [1,2,3,4]
    plt.figure()
    plt.plot(X, bias[0], label='estimator 0')
    plt.plot(X, bias[1], label='estimator 1')
    plt.plot(X, bias[2], label='estimator 2')
    plt.legend()
    plt.title("Bias comparison")
    plt.savefig('bias.png')
    plt.show()


    plt.figure()
    plt.plot(X, sd[0], label='estimator 0')
    plt.plot(X, sd[1], label='estimator 1')
    plt.plot(X, sd[2], label='estimator 2')
    plt.legend()
    plt.title("sd comparison")
    plt.savefig('sd.png')
    plt.show()

    plt.figure()
    plt.plot(X, mse[0], label='estimator 0')
    plt.plot(X, mse[1], label='estimator 1')
    plt.plot(X, mse[2], label='estimator 2')
    plt.legend()
    plt.title("MSE comparison")
    plt.savefig('mse.png')
    plt.show()

    

    
simulation_plot()


"""
W,A,Y = generate_data(100000)
print(estimators(2, W,A,Y))
"""

    