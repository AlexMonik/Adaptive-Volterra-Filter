# Adaptive-Volterra-Filter
## System identification
### Optimal filter
To find optimal weights of filter its required the array of the input signal (rows is the realization of signal, columns is the signal's samples) and output signal.  

Briefly how to use:  
_Define filter model_  
1 - linear filter, 2 - nonliner  
`model = OptimalFilter(1)`  
_Find the optimal weights_  
`model.fit(X, y)`  
_To predict the output values_  
`predicted_values = model.predict(X)`

### Adaptive LMS Volterra filter
For now its requared to train filter before make predictions, I'll add real-time learning later.  
Briefly how to use:  
_Define filter model_  
1 - linear filter, 2 - nonliner, mu1 - step-size for first order weights, mu2 -  step-size for second order weights, complx=False - is data complex or not  
`model = LMSFilter(2, mu1, mu2)`  
_Find the weights; return error on each step_  
`errors = model.train(X, y)`  
_To predict the output values_  
`predicted_values = model.predict(X)`
