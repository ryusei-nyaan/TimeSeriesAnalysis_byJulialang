# TimeSeriesAnalysis_byJulialang

###### Julia version = 1.5.1 

###### Developing and modifying now.

## TSA.jl
### DFT(x)
DFT(x) is discrete fourier transformation. x is Array. 

### hamming(x)
hamming(x) is hamming window. x is Array.

### hanning(x)
hanning(x) is hanning window. x is Array.



### Nonlinear time series analysis

#### embedding(x,m,tau)
embedding is transformation of time-series into phase-space.  
x is Array, m means the dimension of phase-space, tau means the time-delay.

#### FNN(x,tau,m)
FNN enable us to estimate the appropriate embedding dimension.  
x is Array, tau means the time-delay, m is maximum dimension in order to search for the appropriate dimension.

#### TE(x,tau,m,k,q,M)
TE is "translation error" which confirm determinism in your time series.  
x is Array, tau means the time-delay, m is maximum dimension, k means nearest neighbors.  
q means the counts of calculation and the average of that results is return of this function.  
M is the number of sampling of datasets from your reconstructed time-series. The median of this sampling is one of the calculations repeated q times.   
If return of this function are low relatively throughout embedding dimensions you researched, your time-series might be deterministic.  
Furthermore, the appropriate dimension is thought to be the dimension whose return of this function is lowest.  

#### MaxLyapunov(x,m,tau,itr,meth=0,epoch=100)
This function is the estimation of maximum lyapunov exponent.  
x is Array, m is the embedding dimension, tau is the time-delay, itr means the iteration of predicting reconstructed vectors.   
The default is "meth=0". If meth is 1, we predict the reconstructed vector by neural network.   
In this case, we estimate the dynamics of our time-series.  
Finaly, epoch means counts of descent method.  

