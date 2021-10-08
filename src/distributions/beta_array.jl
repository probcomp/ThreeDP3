struct BetaArray <: Distribution{Array{Float64}} end
const beta_array = BetaArray()

function Gen.logpdf(::BetaArray, x::Array{Float64}, a, b)
    pdf((val, a_i, b_i)) = logpdf(beta,val,a_i,b_i) 
    logprobs = pdf.(zip(x,a,b))
    sum(logprobs)
end

function Gen.random(::BetaArray, a, b)
    [random(beta,a[i],b[i]) for i in 1:length(a)]
end

function Gen.logpdf_grad(::BetaArray, x::Array{Float64}, a, b)
    pdf_grad((val, a_i, b_i)) = logpdf_grad(beta,val,a_i,b_i) 
    grads = pdf_grad.(zip(x,a,b))
    map(i->i[1], grads),map(i->i[2], grads),map(i->i[3], grads)
end

export beta_array