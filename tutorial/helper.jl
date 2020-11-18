# Miscellaneous helper functions
#
# Author: Jeffrey W. Miller
# Date: 11/8/2020
#
# This file is released under the MIT "Expat" License.

using BilinearModels
using Statistics
using CSV
using PyPlot
using GLM
using RData


center(x,dim=1) = x .- mean(x,dims=dim)
scale(x,dim=1) = (x .- mean(x,dims=dim))./std(x,dims=dim)
moment2(x) = (x .- mean(x)).^2

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
        # fmt = markers[div.(i_v-1,length(color_palette))+1]
        fmt = markers[mod.(div.(i_v-1,length(color_palette)),length(markers))+1]
        plot(x[subset],y[subset],fmt,label=v,color=col;kwargs...)
    end
    # legend()
end

function pcaplot(X,colorvalue=ones(Int,size(X,2));numeric=false,kwargs...)
    scores,directions,scales = pca(X,2)
    if numeric
        # scatter(scores[:,1], scores[:,2], 2, c=colorvalue; edgecolor="k", linewidth=0.05, cmap="seismic")
        scatter(scores[:,1], scores[:,2]; c=vec(colorvalue), kwargs...)
    else
        plotgroups(scores[:,1],scores[:,2],colorvalue;kwargs...)
    end
    xlabel("PC1")
    ylabel("PC2")
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


function design_matrix(formula, dataframe)
    modelframe = ModelFrame(formula, dataframe)
    X = ModelMatrix(modelframe).m
    X[:,2:end] = scale(X[:,2:end],1)
    return X,coefnames(modelframe)
end


##############################################################################


if false

using PyPlot
using Random
using Distributions
using DataFrames
using GLM
using Interpolations





# _________________________________________________________________________________________________
# Functions

rms(x) = sqrt(mean(x.^2))
rms(x,dim) = sqrt.(mean(x.^2,dims=dim))
avgdif(x,y) = ((x+y)/2, x-y)
unitscale(x,dim=1) = (z = x .- mean(x,dims=dim); z./sqrt.(sum(z.^2,dims=dim)))
ranks(x) = invperm(sortperm(x))
logit(p) = log.(p./(1-p))

function plotcdf(x; log=false, kwargs...)
    n = length(x)
    if log; semilogx(sort(x), (1:n)/n; kwargs...)
    else; plot(sort(x), (1:n)/n; kwargs...)
    end
end

function plot_avgdif(fig,E,a,b)
    figure(fig); clf(); grid(true)
    avg,dif = avgdif(E[:,a],E[:,b])
    plot(avg,dif,".",ms=1)
    title("MA plot of $a and $b")
    xlabel("($a + $b)/2")
    ylabel("$a - $b")
end

# This interface is better:
function plot_avgdif(x,y)
    avg,dif = avgdif(x,y)
    plot(avg,dif,".",ms=1)
    title("MA plot")
    xlabel("(x + y)/2")
    ylabel("x - y")
end

function plotdiffs(fig,E,a,b,c,d)
    figure(fig); clf(); grid(true)
    avg,dif1 = avgdif(E[:,a],E[:,b])
    avg,dif2 = avgdif(E[:,c],E[:,d])
    plot(dif1,dif2,".",ms=1)
    corr = round(cor(dif1,dif2),3)
    title("Diff vs Diff plot (corr = $corr)")
    xlabel("$a - $b")
    ylabel("$c - $d")
end

# Requires MultivariateStats
function mdsplot(E,labels;kwargs...)
    # M = fit(PCA,E; maxoutdim=2)
    # z = transform(M,E)
    D = gram2dmat(E'*E)
    z = classical_mds(D,2)
    # if size(z,1)<2; error("Rank of input is insufficient for 2 principal components."); end
    figure(); clf(); grid(true)
    plotgroups(z[1,:],z[2,:],labels;kwargs...)
    xlabel("Dim1")
    ylabel("Dim2")
    legend()
end

function qqplot(z)
    n = length(z)
    q = quantile.(Normal(),((1:n)-0.5)/n)
    plot(q,q,"k--")
    plot(q,sort(z),".")
    xlabel("Normal")
    ylabel("Observed")
    title("Q-Q plot")
end


function figure_row(j,ncol=4)
    if mod(j,ncol)==0
        figure(figsize=(14,3)); clf()
        subplots_adjust(wspace=0.25,bottom=0.2,left=0.05,right=0.95)
    end
    subplot(1,ncol,mod(j-1,ncol)+1)
end

function quantile_normalize_normal(E)
    I,J = size(E)
    F = zeros(I,J)
    for j = 1:J
        F[:,j] = quantile.(Normal(0,1),(ranks(E[:,j])-0.5)/I)
    end
    return F
end

function quantile_normalize(E)
    I,J = size(E)
    F = zeros(I,J)
    D = vec(mean(sort(E,1),dims=2))
    for j = 1:J
        F[:,j] = D[ranks(E[:,j])]
    end
    return F
end

function variance_explained(E,factor)
    # M = ModelMatrix(ModelFrame(@formula(sample_id ~ 0 + factor), sample_info)).m
    M = [float(factor[j]==c) for j = 1:length(factor), c in unique(factor)]
    M = M[:,vec(sum(M,dims=1).!=0)]
    mu = pinv(M)*E'
    RSS = sum((E' - M*mu).^2)
    TSS = sum((E' .- mean(E',dims=1)).^2)
    R_squared = 1 - RSS/TSS
    println("R-squared = $R_squared (fraction of variance explained by factor)")
end

# Recode a list of arbitrary labels as numbers
function recode_as_numbers(labels)
    u = unique(labels)
    K = length(u)
    D = Dict(zip(u,1:K))
    return [D[l] for l in labels], D
end

# Compute the "one-hot" representation corresponding to given the sequence of labels.
function one_hot(labels)
    u = unique(labels)
    K = length(u)
    D = Dict(zip(u,1:K))
    A = Int[(D[l]==k) for l in labels, k=1:K]
    return A,u
end

# function dummy_vars_by_tags(tag_lists)
    # unique_tags = sort(unique(vcat(tag_lists...)))
    # K = length(unique_tags)
    # tag2ind = Dict(zip(unique_tags,1:K))
    # ind_lists = map(L->[tag2ind[t] for t in L], tag_lists)
    # X = zeros(length(ind_lists),K)
    # for (i,L) in enumerate(ind_lists); X[i,L] = 1; end
    # return X
# end


agg(x,dim) = vec(sum(x,dims=dim))./sqrt.(size(x,dim))

# Estimate stddev and provide a (1-alpha)*100% confidence interval for it.
function std_with_error(x,alpha,dim)
    n = size(x,dim)
    m = vec(mean(x,dims=dim))
    ss = vec(sum((x .- m).^2, dims=dim))
    q1,q2 = quantile.(Chisq(n-1), [alpha/2, 1-alpha/2])
    lower,upper = sqrt.(ss/q1),sqrt.(ss/q2)
    stddev = sqrt.(ss./(n-1))
    return stddev,lower,upper
end


# F-test for H0: var(x) = var(y) versus H1: var(x) > var(y).
function F_test(x,y)
    n,m = length(x),length(y)
    if (n<2)||(m<2); return log10(1.0); end   # What is the technically correct thing to do here?
    F = var(x; corrected=true) / var(y; corrected=true)
    log10_pvalue = logccdf(FDist(n-1,m-1), F) / log(10)
    return log10_pvalue
end

# Compute 100*P % trimmed mean for each column of x (discard the top and bottom 100*P % of the data)
trimmed_mean(x,P) = (m=floor(Int,P*size(x,1)); mean(sort(x;dims=1)[m:end-m+1, :],dims=1))

function trimmed_scale(x,loc,P)
    a,b = quantile.(Normal(0,1),[P,1-P])
    z = 1 - 2*P
    pa,pb = pdf.(Normal(0,1),[a,b])
    factor = 1 + (a*pa-b*pb)/z - ((pa-pb)/z)^2
    m = floor(Int,P*size(x,1))
    v = mean((sort(x;dims=1)[m:end-m+1, :].-loc).^2, dims=1)
    return sqrt.(v/factor)
end

# Robust percent variance explained (PVE) and signal-to-noise ratio (SNR) for each column of x.
function metrics(x)
    P = 0.1  # percent to trim for robust estimation AND removal of jumps in the signal
    # m_total = trimmed_mean(x,P)
    # s_total = trimmed_scale(x,m_total,P)
    x_diff = diff(x;dims=1)
    m_diff = trimmed_mean(x_diff, P)
    s_diff = trimmed_scale(x_diff,m_diff,P)
    x_avg = (x[1:end-1,:] + x[2:end,:])/2
    m_avg = trimmed_mean(x_avg, P)
    s_avg = trimmed_scale(x_avg,m_avg,P)
    v_noise = 0.5*s_diff.^2
    v_total = s_avg.^2 + 0.25*s_diff.^2
    PVE = vec(1 .- v_noise./v_total)
    SNR = vec(v_total./v_noise .- 1)
    return PVE,SNR
end

# Compute quantile p for a discrete random variable taking values x=[x1,...,xn] with probabilities proportional to w=[w1,...,wn].
function weighted_quantile(x,w,p)
    c = 0.0
    sum_w = sum(w)
    for i in sortperm(x)
        c += w[i]/sum_w
        if c >= p; return x[i]; end
    end
    error("Probabilities only add up to $c, which is less than requested quantile, $p.")
end



function plot_relationship(x,y)
    plot(x, y, "b,")
    grid(axis="y",lw=0.2)
    xp = range(minimum(x)+eps(),stop=maximum(x)-eps(), length=1000)
    percentiles = [.01; .05; .1; .2; .4; .6; .8; .9; .95; .99]
    # yp = spline_fit_and_predict(x,y,xp; lambda=0.0, natural=false, intercept=true, percentiles=percentiles, degree=1)
    # plot(xp[1:end],yp[1:end], "r-")
    xs,y_mean = regression(x,y)
    plot(xs,y_mean,"co-",lw=1,ms=1)
end

function plot_track(x,contigs, groups=ones(length(x)), colorkey=String[], fmt=","; kwargs...)
    if isempty(colorkey); colorkey = zip(sort(unique(groups)), color_palette[1:length(unique(groups))]); end
    for (value,color) in colorkey
        plot(findall(groups.==value),x[groups.==value], fmt; color=color, kwargs...)
    end
    edges = [1; findall(diff(recode_as_numbers(contigs)[1]) .!= 0); length(contigs)]
    mid = (edges[1:end-1] + edges[2:end])/2
    for edge in edges; plot([edge,edge],[-1e10,1e10],"k-",lw=0.15); end
    grid(axis="y",lw=0.2)
    ylim(minimum(x),maximum(x))
    xticks(mid,unique(contigs),fontsize=9)
end

function regression(x,y; nbins=round(Int,2*length(unique(x))^(1/2)))
    edges = [minimum(x)-eps(); quantile(x,(1:nbins-1)/nbins); maximum(x)+eps()]
    # edges = range(minimum(x)-eps(), stop = maximum(x)+eps(), length = nbins+1)
    xs = (edges[1:end-1] + edges[2:end])/2
    # y_median = [median(y[edges[i] .<= x .< edges[i+1]]) for i = 1:nbins]
    y_mean = [mean(y[edges[i] .<= x .< edges[i+1]]) for i = 1:nbins]
    return xs,y_mean
end

function bivariate_regression(x,y,z; nx=30, ny=30)
    ex = range(minimum(x)-eps(), stop = maximum(x)+eps(), length = nx+1)
    ey = range(minimum(y)-eps(), stop = maximum(y)+eps(), length = ny+1)
    xr = (ex[1:end-1] + ex[2:end])/2
    yr = (ey[1:end-1] + ey[2:end])/2
    zr = permutedims([mean(z[(ex[i] .<= x .< ex[i+1]) .& (ey[j] .<= y .< ey[j+1])]) for i=1:nx, j=1:ny])
    return xr,yr,zr
end

# Find the indices of entries of B that are in A, assuming they appear in the same order that they appear in A.
function findin_sorted(A,B)
    subset = Int[]
    i = 1; j = 1
    while (i<=length(A)) && (j<=length(B))
        if A[i]==B[j]; push!(subset,j); i += 1; end
        j += 1
    end
    return subset
end

function head(filename; n=10)
    f = open(filename,"r")
    s = join([readline(f) for i=1:n],"\n")
    close(f)
    return s
end

function F1score(hits,actual_positives)
    A,B = hits,actual_positives
    precision = length(intersect(A,B)) / length(A)
    recall = length(intersect(A,B)) / length(B)
    F1 = 2/(1/precision + 1/recall)
    return F1,precision,recall
end

impute_missing!(x) = (mis = ismissing.(x); x[mis] .= mean(x[.!mis]); x)


# Return vector of booleans indicating whether each element of a is in b.
function subset_in(a,b)
    x = falses(length(a))
    x[findall(in(b),a)] .= 1
    return x
end


benjamini_hochberg(p) = (o = sortperm(p); m=length(p); padj = zeros(Float64,m); padj[o] = p[o].*(m./(1:m)); padj)


function standardize(X)
    max_iter = 100
    tol = 1e-6
    X_old = copy(X)
    for iter = 1:max_iter 
        X = scale(X,1)
        X = scale(X,2)
        if (maximum(abs.(X - X_old)) < tol); break; end
        X_old = copy(X)
    end
    return X
end




end


