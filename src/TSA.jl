module TSA
using Statistics
using Distributions
using LinearAlgebra
using Random

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

function TE(x,τ,m,k,q,M)
    Etrans = []
    for i = 1:m
        X = embedding(x,i,τ)
        d =zeros(size(X)[1]-τ,k+1)
        for j = 1:size(X)[1]-τ
            dist = map(x->x.-X[j,:],X[1:size(X)[1]-τ,:])
            dist = sum.(abs2,dist[:,1:i])
            d[j,1:k+1] .= sortperm(dist[:,1])[1:k+1]
        end
        Et = []
        d = Int.(d)
        for j = 1:q
            rn = map(Int,rand(1:size(X)[1]-τ,M))
            E = []
            for n in rn
                v = X[d[n,:].+τ,:] .- X[d[n,:],:]
                v̄ = sum(v,dims=1)/(k+1)
                E = push!(E,sum(abs2,(v.-v̄))/((k+1)*sum(abs2,v̄)))
            end
            Et = push!(Et,median(E))
        end
        Etrans = push!(Etrans,mean(Et))
    end
    return Etrans
end

end
end
