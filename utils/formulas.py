# -*- coding: utf-8 -*-
__author__ = 'Muhammad Ghafoor'
__version__ = '1.0.1'
__email__ = "mdhgafoor@outlook.com"

"""
File Name: formulas.py
Description: Collection of formulas obtained throuhgout Machine Learning Specialization
             offered by Stanford University and Deeplearning.ai. 
             https://www.coursera.org/specializations/machine-learning-introduction
"""

import math
import np 

"""
LINEAR REGRESSION
"""

def compute_linear_regression_cost(x,y,w,b):
    """
    Computes cost function for a standard linear regression model.
    f_wb(x^(i)) = wx^(i) + b
    cost^(i) = (f_wb-y(i))^2
    J_wb = (1/(2m))*SUM_(m-1)_(i=0)[cost^(i)]

    Args:
        x: [ndarray] Input Data
        y: [ndarray] Label
        w,b: [scalar] Scalar parameters of the model

    Returns:
        total_cost: [float] Cost of using w,b as the parameters for linear regressions 
        to fit the data points in x and y
    """

    m = x.shape[0]     
    total_cost = 0
    
    for i in range(m):
        f_wb_i = w*x[i]+b
        cost = (f_wb_i-y[i])**2    
        total_cost = total_cost + cost
    
    total_cost = (1/(2*m))*total_cost
    
    return total_cost


def compute_linear_regression_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression.
    dJ(w,b)^(i)/db = (f_w,b(x^(i))-y^(i))
    dJ(w,b)^(i)/dw = (f_w,b(x^(i))-y^(i))*x^(i)
    
    dj(w,b)/db = (1/m)*SUM_(m-1)_(i=0)[dJ(w,b)^(i)/db]
    dj(w,b)/dw = (1/m)*SUM_(m-1)_(i=0)[dJ(w,b)^(i)/dw]

    Args:
        x: [ndarray] Input Data
        y: [ndarray] Label
        w,b: [scalar] Scalar parameters of the model
    
    Returns:
        dj_dw: [scalar] Gradient of the cost with respect to the parameters w
        dj_db: [scalar] Gradient of the cost with respect to the paremeter b
    """

    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb_i = w*x[i]+b
        dj_db_i = f_wb_i-y[i]
        dj_dw_i = (f_wb_i-y[i])*x[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i 
    
    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db

    return dj_dw, dj_db


def execute_gradient_decent_linear_regression(x, y, w_in, b_in, alpha, num_iters):
    """
    Executes gradient decent to learn theta (w,b) for linear_regression

    Args:
        x: [ndarray] Input Data
        y: [ndarray] Label
        w_in, b_in: [scalar] Scalar parameters of the model
        alpha: [float] learning rate
        num_iters: [int] number of iterations to run gradient decent
    Returns:
        w: [ndarray] Updated values of parameters of the model after running gradient decent
        b: [scalar] Updated value of the parameter of the model after running gradient decent
        J_history: [ndarray] Array to store cost calculations
        w_history: [ndarray] Array to store w value calculations
    """
    m = len(x)

    J_history = [] 
    w_history = [] 
    w = w_in
    b = b_in

    for i in range(num_iters):
        
        dj_dw, dj_db = compute_linear_regression_gradient(x, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i<100000:
            cost = compute_linear_regression_cost(x, y, w, b)
            J_history.append(cost)
        
        if i%math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
    
    return w, b, J_history, w_history


"""
LOGISTIC REGRESSION
"""


def calculate_sigmoid(z):
    """
    Computes the sigmoid of z
    g(z) = 1/(1+e^-z)

    Args:
        z: [ndarray] A scalar, numpy array 
    
    Returns:
        g: [ndarray] sigmoid(z) with same shape as z
    """

    g = 1/(1+math.e**-z)
    
    return g


def compute_logistic_regression_cost(X, y, w, b, lambda_):
    """
    Computes cost for logistic regression model
    J_wb = (1/m)*SUM_(m-1)_(i=0)[loss(f_wb(x^(i)),y^(i)]
    loss(f_wb(x^(i),y^(i)) = (-y^(i)log(f_wb(x^(i))-(1-y^(i))log(1-f_wb(x^(i))
    f_wb = g(w•x^(i) + b)
    f_wb = g(z_wb(x^(i)))

    Args:
        X: [ndarray] Input data
        y: [ndarray] Target value
        w: [ndarray] Values of parameters of the model
        b: [scalar] Value of bias parameters of the model
        lambda_ = [scalar, float] Regularization constant. Can default to 1.
    
    Returns:
        total_cost: [scalar] cost of logistic regression
    """
    m, n = X.shape
    loss_sum = 0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j]*X[i][j]
            z_wb += z_wb_ij
        z_wb += b

        f_wb = sigmoid(z_wb)
        loss = -y[i]*np.log(f_wb)-(1-y[i])*np.log(1-f_wb)
        loss_sum += loss

    total_cost = (1/m)*loss_sum

    reg_cost = 0
    for j in range(n):
        reg_cost_j = w[j]**2
        reg_cost += reg_cost_j
        
        reg_cost = (lambda_/(2*m))*reg_cost
        total_cost += reg_cost

    return total_cost


def compute_logistic_regression_gradient(X, y, w, b, lambda_):
    """
    Computes the gradient for linear regression.
    b := b - alpha*(dJ(w,b)/db)
    w_j := w_j - alpha(dJ(w,b)/dw_j) for j := 0..n-1

    dJ(w,b)/db = (1/m)*SUM_(m-1)_(i=0)[f_wb(x^(i))-y(i)]
    dJ(w,b)/dw_j = (1/m)*SUM_(m-1)_(i=0)[f_wb(x^(i))-y^(i)]x_j^(i

    Args:
        X: [ndarray] Input Data
        y: [ndarray] Label
        w,b: [scalar] Scalar parameters and bias of the model
        lambda_: [scalar,float] Regularization constant. Can default to 1.
    
    Returns:
        dj_dw: [ndarray] Gradient of the cost with respect to the parameters w
        dj_db: [scalar] Gradient of the cost with respect to the paremeter b
    """   

    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0

    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb_ij = w[j]*X[i][j]
            z_wb += z_wb_ij
        z_wb += b

        f_wb = calculate_sigmoid(z_wb)

        dj_db_i = f_wb-y[i]
        dj_db += dj_db_i 

        for j in range(n):
            dj_dw_ij = (f_wb-y[i]*X[i][j])
            dj_dw[j] += dj_dw_ij 
    
    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db

    for j in range(n):
        dj_dw_j_reg =(lambda_/m)*w[j]
        dj_dw[j] = dj_dw[j] + dj_dw_j_reg 

    return dj_db, dj_dw 


def execute_gradient_decent_logistic_regression(X, y, w_in, b_in, alpha, num_iters, lambda_=0):
    """
    Executes gradient decent to learn theta (w,b) for logistic regression

    Args:
        X: [ndarray] Input data
        y: [ndarray] Target value
        w_in: [ndarray] Initial values of parameters of the model
        b_in: [scalar] Initial bias parameter of the model
        alpha: [float] Learning rate
        num_iters: [int] numnber of iterations to run gradient decent
        lambda_: [scalar, float] reguarlization constant
    """

    m = len(X)

    J_history = []
    w_history = []

    for i in range(num_iters):
        dj_db, dj_dw = compute_logistic_regression_gradient(X, y, w_in, b_in, lambda_)
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db 

        if i<100000:
            cost =  compute_logistic_regression_cost(X, y, w_in, b_in, lambda_) 
            J_history.append(cost)
        
        if i%math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")
    
    return w_in, b_in, J_history, w_history


def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters w

    Args:
        X: [ndarray] Input data
        w: [ndarray] values of parameters of the model
        b: [scalar] value of bias parameters of the model

    Returns:
        p: [ndarray] Predictions for X using a threshold of 0.5
    """

    m, n = X.shape 
    p = np.zeros(m)

    for i in range(m):
        z_wb = 0 
        for j in range(n):
            z_wb_ij = w[j]*X[i][j]
            z_wb += z_wb_ij
        
        z_wb += b

        f_wb = calculate_sigmoid(z_wb)

        p[i] = 0 if f_wb < 0.1 else 1
    
    return p