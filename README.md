# Adaptive-Volterra-Filter
## System identification
### Optimal filter
To find optimal weights of filter its required the array of the input signal (rows is the realization of signal, columns is the signal's samples) and output signal.  

Briefly how to use:
Define filter model
1. model = OptimalFilter()  
Find the optimal weights    
2. model.fit(X, y)  
To predict the output values  
3. predicted_values = model.predict(X)
