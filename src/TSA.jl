module TSA
using Statistics
using Distributions
using LinearAlgebra


function DFT(x)
    N = length(x)
    X̂ = zeros(Complex,N)
    μ = collect(Int64,0:1:N-1)
    for i = 1:N
        X̂ .+= x[i]*(exp(-2π*im*(i-1)/N)).^μ
    end
    return X̂
end

function hamming(x)
    N = length(x)
    n = collect(Int64,0:1:N-1)
    wind = map(x->x+0.54,-0.46*cos.(2π*n/(N-1)))
    return wind
end

function hanning(x)
    N = length(x)
    n = collect(Int64,0:1:N-1)
    wind = map(x->x+0.5,-0.5*cos.(2π*n/(N-1)))
    return wind
end
module NTSA
function embedding(x,m,τ)
    N = length(x)
    X = zeros(N-(m-1)τ,m)
    for i = 1:m
        X[1:N-(m-1)τ,i] .= x[1+(i-1)τ:N-(m-1)τ+(i-1)τ]
    end
    return X
end
function FNN(x,τ,m)
    E = []
    for i = 1:m+1
        X = embedding(x,i,τ)
        a = zeros(size(X)[1])
        mindist = zeros(size(X)[1])
        for j = 1:size(X)[1]
            if mindist[j] == 0
                d = zeros(size(X)[1])
                for k = 1:size(X)[1]
                    d[k] = norm(X[j,:]-X[k,:])
                end
                d[j] = Inf
                k = argmin(d)
                mindist[j] = d[k]
                mindist[k] = d[k]
            end
        end

        X2 = embedding(x,i+1,τ)
        mindist2 = zeros(size(X2)[1])
        for j = 1:size(X2)[1]
            if mindist2[j] == 0
                d = zeros(size(X2)[1])
                for k = 1:size(X2)[1]
                    d[k] = norm(X2[j,:]-X2[k,:])
                end
                d[j] = Inf
                k = argmin(d)
                mindist2[j] = d[k]
                mindist2[k] = d[k]
            end
        end

        a = mindist2./mindist[1:size(X2)[1]]
        E = push!(E,sum(a)/size(X2)[1])
    end
    E1 = []
    E1 = E[2:length(E)]./E[1:length(E)-1]
    return E1
end


end
end
