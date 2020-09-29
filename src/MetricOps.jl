function det(g::NTuple{6,T}) where T
  return g[1]*(g[3]*g[6]-g[5]^2) + g[2]*(g[5]*g[4] - g[2]*g[6]) + g[4]*(g[2]*g[5]-g[3]*g[4])
end

function det(g::MetricTensor{3,T,N}) where {T,N}
  res = Array{T,N}(undef,size(g))
  map!(det,res,g)
  return res
end
