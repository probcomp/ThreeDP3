struct BernoulliArray <: Distribution{Array{Bool}} end
const bernoulli_array = BernoulliArray()

function Gen.logpdf(::BernoulliArray, x::Array{Bool}, weights::Array{Float64})
    heads = x
    tails =.!x    
    sum(log.(weights[heads])) + sum(log.(1.0 .- weights[tails]))
end

function Gen.random(::BernoulliArray, weights::Array{Float64})
    [random(bernoulli,p) for p in weights]
end

export bernoulli_array