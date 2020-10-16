# TimeSeriesAnalysis_byJulialang

###### Julia version = 1.5.1 

## TSA.jl
### DFT(x)
DFT(x) is discrete fourier transformation. x is Array. 

### hamming(x)
hamming(x) is hamming window. x is Array.

### hanning(x)
hanning(x) is hanning window. x is Array.


### NTSA
NTSA is nonlinear time series analysis. 


#### embedding(x,m,tau)
embedding is transformation of time-series into phase-space.
x is Array, m means the dimension of phase-space, tau means the time-delay.

#### FNN(x,tau,m)
FNN enable us to estimate the appropriate embedding dimension.
x is Array, tau means the time-delay, m is maximum dimension in order to search for the appropriate dimension.
