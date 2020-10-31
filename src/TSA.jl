module TSA

using Statistics
using Distributions
using LinearAlgebra
using Random
using Flux

function DFT(x)
    N = length(x)
    X̂ = zeros(Complex,N)
    μ = collect(Int64,0:1:N-1)
    for i = 1:N
        X̂ .+= x[i]*(exp(-2π*im*(i-1)/N)).^μ
    end
    return X̂
end

function IDFT(x)
    N = length(x)
    X = zeros(Complex,N)
    μ = collect(Int64,0:1:N-1)
    for i = 1:N
        X .+= x[i]/N*(exp(2π*im*(i-1)/N)).^μ
    end
    return real(X)
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
            dist = zeros(size(X)[1]-τ)
            for l = 1:size(X)[1]-τ 
                dist[l] = norm(X[j,:]-X[l,:])
            end
            
            d[j,1:k+1] .= sortperm(dist)[1:k+1]
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

function MaxLyapunov(x,m,τ,itr,k,meth=0,epoch=100)
    σ = std(x)
    X = embedding(x,m,τ)
    N = size(X)[1]-1
    ϵ1 = 0
    if meth == 0
        X̂ = copy(X[1:N,:])
        for j = 1:itr
            d = zeros(BigInt,N,k)
            for i = 1:N
                dist = zeros(N)
                for l = 1:N
                    dist[l] = norm(X̂[i,:]-X[l,:])
                end
                d[i,1:k] = sortperm(dist)[2:k+1]
                X̂[i,:] .= sum(X[d[i,:].+1,:])/k
            end
            if j==1
                ϵ1 = sqrt(sum((X[1+j:end,1].-X̂[1:end,1]).^2)/N)/σ
            end
        end
        ϵ = sqrt(sum((X[1+itr:end,1].-X̂[1:end-itr+1,1]).^2)/(size(X)[1]-itr))/σ
    elseif meth==1
        X̂ = copy(X[1:end-1,:])
        m2 = 2*m
        d1 = Dense(m,m2,sigmoid) |>f64
        d2 = Dense(m2,m) |>f64
        model = Chain(d1,d2)
        loss(x,x̂) = Flux.Losses.mse(model(x),x̂)
        ps = params(model)
        opt = Descent()
        xt = []
        yt = []
        for i = 1:N
            xt = push!(xt,X̂[i,:])
            yt = push!(yt,X[i+1,:])
        end
        for i = 1:epoch
            data = zip(xt,yt)
            Flux.train!(loss,ps,data,opt)
        end
        for j = 1:itr
            for i = 1:N-j
                X̂[i,:] = model(X̂[i,:])
            end
            if j==1
                ϵ1 = sqrt(sum((X[1+j:end,1].-X̂[1:N-j+1,1]).^2)/N)/σ
            end
        end
        ϵ = sqrt(sum((X[1+itr:end,1].-X̂[1:end-itr+1,1]).^2)/(size(X)[1]-itr))/σ
    end
    λ = log(ϵ/ϵ1)/(itr-1)
    return λ
end

function surrogate(x,m,τ,itr,c)
    N = length(x)
    E = []E0 = MaxLyapunov(x,m,τ,itr,1,1)
    for i = 1:c
        sX = copy(x)
        fsX = DFT(sX)
        fsX = fsX.*exp(2*π*im).^rand(Float64,N)
        sX = IDFT(fsX)sX = real(sX)
        E1 = MaxLyapunov(sX,m,τ,itr,1,1)
        E = push!(E,E1)
        end
    t = (E0-mean(E))/(std(E)/(c^0.5))
    return t
end
end
