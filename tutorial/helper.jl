# Miscellaneous helper functions
#
# Copyright (c) 2020: Jeffrey W. Miller.
# This file is released under the MIT "Expat" License.

using BilinearModels
using Statistics
using Distributions
using CSV
using RData
using GLM
using CategoricalArrays
using DataFrames
using PyPlot

using Random
using LinearAlgebra: pinv, diag


center(x,dim=1) = x .- mean(x,dims=dim)
center_scale(x,dim=1) = (x .- mean(x,dims=dim))./std(x,dims=dim)

markers = ["o","v","^","s","*","<",">","p","+","x","D","d","."]

# Color palette from https://sashat.me/2017/01/11/list-of-20-simple-distinct-color_palette/
color_palette = ["#e6194b", "#0082c8", "#3cb44b", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe", "#008080", "#e6beff", "#aa6e28", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000080", "#808080", "#ffe119", "#FFFFFF", "#000000"]


function pca(X,k)
    X_centered = X .- mean(X,dims=1)
    U,D,V = BilinearModels.tsvd(X_centered,k=k)
    directions = V
    scales = D / sqrt(size(X,1)-1)
    scores = X*directions
    return scores,directions,scales
end


function plotgroups(x,y,labels;kwargs...)
    for (i_v,v) in enumerate(sort(unique(labels,dims=1)))
        subset = findall(labels.==v)
        col = color_palette[mod.(i_v-1,length(color_palette))+1]
        marker = markers[mod.(div.(i_v-1,length(color_palette)),length(markers))+1]
        scatter(x[subset],y[subset]; marker=marker, label=string(v), c=col, kwargs...)
    end
end


function pcaplots(X,colorvalues; s=20, ec="k", lw=0.25, kwargs...)
    scores = pca(X,2)[1]
    for name in names(colorvalues)
        x = colorvalues[!,name]
        figure(); clf(); grid(true)
        if isa(x,CategoricalArray)
            plotgroups(scores[:,1], scores[:,2], string.(x); s=s, ec=ec, lw=lw, kwargs...)
            subplots_adjust(right=0.83)
            legend()
        else
            scatter(scores[:,1], scores[:,2]; c=x, s=s, ec=ec, lw=lw, kwargs...)
            subplots_adjust(right=0.9)
            colorbar(fraction=0.04)
        end
        xlabel("PC1")
        ylabel("PC2")
        title("PCA scores ($(string(name)))")
    end
    return scores
end


# Compute RPM, RPKM, and TPM from a count matrix C (features-by-samples).
# This assumes that lengths[i] is the length of feature i in bases.
function standard_normalizations(C,lengths)
    S = sum(C,dims=1)/1e6
    RPM = C./S  # "reads per million"
    RPKM = RPM./(lengths/1000.0)
    TPM = RPKM ./ (sum(RPKM,dims=1)/1e6)
    # RPK = C./(lengths/1000.0)
    # TPM = RPK ./ (sum(RPK,dims=1)/1e6)
    return RPM, RPKM, TPM
end


function design_matrix0(formula, dataframe)
    modelframe = ModelFrame(formula, dataframe)
    X = ModelMatrix(modelframe).m
    X[:,2:end] = center_scale(X[:,2:end],1)
    return X,coefnames(modelframe)
end


function design_matrix(formula, dataframe)
    n = size(dataframe,1)
    X_blocks = Matrix{Float64}[]
    coefnames = String[]
    terms = (isa(formula.rhs,Tuple) ? formula.rhs : (formula.rhs,))
    for term in terms
        if isa(term,ConstantTerm)
            push!(X_blocks, ones(n,1))
            push!(coefnames,"(Intercept)")
        elseif isa(term,InteractionTerm)
            if (length(term.terms) > 2); return design_matrix0(formula, dataframe); end
            s1,s2 = term.terms[1].sym,term.terms[2].sym
            x1,x2 = dataframe[!,s1],dataframe[!,s2]
            
            if isa(x1,CategoricalArray) && isa(x2,CategoricalArray); return design_matrix0(formula, dataframe); end
            
            if isa(x1,CategoricalArray) || isa(x2,CategoricalArray)
                if isa(x1,CategoricalArray)
                    x1,x2 = x2,x1
                    s1,s2 = s2,s1
                end
                A,values = one_hot(x2)
                push!(X_blocks, x1.*A[:,1:end-1])
                labelnames = [string(s1)*" & "*string(s2)*": "*string(v) for v in values]
                coefnames = [coefnames; labelnames[1:end-1]]
            else
                push!(X_blocks, reshape(x1.*x2,n,1))
                push!(coefnames, string(s1)*" & "*string(s2))
            end
        else
            x = dataframe[!,term.sym]
            if isa(x,CategoricalArray)
                A,values = one_hot(x)
                push!(X_blocks, A[:,1:end-1])
                labelnames = [string(term.sym)*": "*string(v) for v in values]
                coefnames = [coefnames; labelnames[1:end-1]]
            else
                push!(X_blocks, reshape(x,n,1))
                push!(coefnames, string(term.sym))
            end
        end
    end
    X = hcat(X_blocks...)        
    X[:,2:end] = center_scale(X[:,2:end],1)
    return X,coefnames
end


# Compute the "one-hot" representation corresponding to given the sequence of labels.
function one_hot(labels)
    u = unique(labels)
    K = length(u)
    D = Dict(zip(u,1:K))
    A = Int[(D[l]==k) for l in labels, k=1:K]
    return A,u
end


benjamini_hochberg(p) = (o = sortperm(p); m=length(p); padj = zeros(Float64,m); padj[o] = p[o].*(m./(1:m)); padj)


# Perform estimation and inference for standard linear regression model:
#    y = A*beta + e
#    where e_i ~ N(0,sigma^2) for i=1,...,n.
#
# Inputs:
#   X = n-by-d design matrix
#   y = length n vector of outcomes
#
# Outputs:
#   beta_hat = length d vector of estimated coefficients
#   sigma_hat = estimated standard deviation of outcomes
#   stderr = length d vector of standard errors for entries of beta_hat
#   t = length d vector of t-statistics for entries of beta_hat
#   logp = natural log of p-value for the specified test (gt: alt>null, lt: alt<null, tt: alt!=null)
function infer_linear_regression(X,y; beta_null=zeros(size(X,2)), test="tt")
    n,d = size(X)
    @assert(n>d, "Sample size (n) must be greater than number of coefficients (d).")
    beta_hat = vec(pinv(X)*y)
    y_hat = vec(X*beta_hat)
    e = y - y_hat
    sigma_hat = sqrt.(sum(e.*e)/(n-d))
    stderr = sigma_hat .* sqrt.(diag(inv(X'*X)))
    t = (beta_hat - beta_null) ./ stderr
    logp_gt = logccdf.(TDist(n-d),t)
    logp_lt = logcdf.(TDist(n-d),t)
    logp_tt = [min(logp_gt[j],logp_lt[j])+log(2) for j=1:d]
    logp = Dict("gt"=>logp_gt, "lt"=>logp_lt, "tt"=>logp_tt)[test]
    RSS = sum(e.*e)
    TSS = sum((y .- mean(y)).^2)
    R_squared = 1 - RSS/TSS
    return beta_hat,sigma_hat,stderr,t,logp,R_squared
end


function plot_association(x,y,xs,xs_names; showline=false)
    Random.seed!(0)
    J = length(x)
    jitter = (rand(J) .- 0.5)*3
    plot(x .+ jitter, y, "b.", ms=3)
    if showline
        beta_hat,sigma_hat,stderr,t,logp,R_squared = infer_linear_regression([ones(J) x], y; beta_null=zeros(2), test="tt")
        plot(xs, [ones(length(xs)) xs]*beta_hat, "k-", lw=1)
    end
    grid(linewidth=0.25)
    xticks(xs,xs_names)
end












