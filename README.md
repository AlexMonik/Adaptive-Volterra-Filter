# Adaptive-Volterra-Filter
## System identification
### Optimal filter
To find optimal weights of filter its required the array of the input signal (rows is the realization of signal, columns is the signal's samples) and output signal.  

Briefly how to use:  
_Define filter model_  
`model = OptimalFilter()`  
_Find the optimal weights_  
`model.fit(X, y)`  
_To predict the output values_  
`predicted_values = model.predict(X)`
