
struct Vector3D{T<:Real}
  x::Vector{T}
  y::Vector{T}
  z::Vector{T}
  length::Int
  basisType::Symbol
  function Vector3D{T}(size::Int=0,basis::Symbol=:co) where T<:Real
    x = Vector{T}(undef,size)
    y = Vector{T}(undef,size)
    z = Vector{T}(undef,size)
    length = size
    return new(x,y,z,length,basis)
  end
  function Vector3D{T}(xVec::Vector{T},yVec::Vector{T},zVec::Vector{T},basis::Symbol=:co) where T<:Real
    @assert Base.length(xVec) == Base.length(yVec) "xVec and yVec have incompatible lengths"
    @assert Base.length(yVec) == Base.length(zVec) "yVec and zVec have incompatible lengths"
    x = xVec
    y = yVec
    z = zVec
    length = Base.length(x)
    return new(x,y,z,length,basis)
  end

end
#=
# Define the constructors
=#

struct Surface3D{T<:Real}
  x::Array{T,2}
  y::Array{T,2}
  z::Array{T,2}
  size::Tuple{Int,Int}
  basisType::Symbol
  function Surface3D{T}(dim1::Int=0,dim2::Int=0,basis::Symbol=:co) where T<:Real
    x = Array{T}(undef,dim1,dim2)
    y = Array{T}(undef,dim1,dim2)
    z = Array{T}(undef,dim1,dim2)
    size = tuple(dim1,dim2) 
    return new(x,y,z,size,basis)
  end
  function Surface3D{T}(dims::Tuple{Int,Int},basis::Symbol=:co) where T <: Real
    return Surface3D{T}(dims[1],dims[2],basis)
  end
  function Surface3D{T}(xArr::Array{T,2},yArr::Array{T,2},zArr::Array{T,2},basis::Symbol=:co) where T<:Real
    @assert Base.size(xArr) == Base.size(yArr) "xVec and yVec have incompatible lengths"
    @assert Base.size(yArr) == Base.size(zArr) "yVec and zVec have incompatible lengths"
    x = xArr
    y = yArr
    z = zArr
    size = Base.size(x)
    return new(x,y,z,size,basis)
  end

end

#=
# Define the constructors
=#

mutable struct Fieldline{T<:Real}
  s::Float64
  alpha::Float64
  zeta::Vector{Float64}
  B::Vector3D{T}
  length::Int
  function Fieldline{T}(s_in::Float64,alpha_in::Float64,zeta_in::Vector{Float64}) where T <: Real
    s = s_in
    alpha = alpha_in
    zeta = zeta_in
    B = Vector3D{T}(Base.length(zeta))
    length = Base.length(zeta)
    return new(s,alpha,zeta,B,length)
  end
  Fieldline{T}() where T <: Real = new()
end
