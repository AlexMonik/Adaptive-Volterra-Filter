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
