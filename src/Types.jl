
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
    @assert Base.size(xArr) == Base.size(yArr) "xArr and yVec have incompatible lengths"
    @assert Base.size(yArr) == Base.size(zArr) "yVec and zVec have incompatible lengths"
    x = xArr
    y = yArr
    z = zArr
    size = Base.size(x)
    return new(x,y,z,size,basis)
  end

end


struct MetricTensor{T<:Real,N}
  xx::Array{T,N}
  xy::Array{T,N}
  yy::Array{T,N}
  xz::Array{T,N}
  yz::Array{T,N}
  zz::Array{T,N}
  size::Tuple

end

#Define the MetricTensor constructors for different instances
function MetricTensor(xxData::Array{T},xyData::Array{T},yyData::Array{T},
                      xzData::Array{T},yzData::Array{T},zzData::Array{T}) where T<: Real
  @assert Base.size(xxData) == Base.size(xyData) == Base.size(yyData) == Base.size(xzData) == Base.size(yzData) == Base.size(zzData) "Incompatible array sizes"
  size = Base.size(xxData)
  N = Base.length(size)
  xx = xxData
  xy = xyData
  yy = yyData
  xz = xzData
  yz = yzData
  zz = zzData
  return MetricTensor{T,N}(xx,xy,yy,xz,yz,zz,size)
end

function MetricTensor(xxData::Array{T},xyData::Array{T},yyData::Array{T},zzData::Array{T}) where T<: Real
  @assert Base.size(xxData) == Base.size(xyData) == Base.size(yyData) == Base.size(zzData) "Incompatible array sizes"
  size = Base.size(xxData)
  N = base.length(size)
  xx = xxData
  xy = xyData
  yy = yyData
  xz = zeros(T,size) 
  yz = zeros(T,size)
  zz = zzData
  return MetricTensor{T,N}(xx,xy,yy,xz,yz,zz,size)
end

function MetricTensor(xxData::Array{T},yyData::Array{T},zzData::Array{T}) where T<: Real
  @assert Base.size(xxData) == Base.size(yyData) == Base.size(zzData) "Incompatible array sizes"
  size = Base.size(xxData)
  N = Base.length(size)
  xx = xxData
  xy = zeros(T,size)
  yy = yyData
  xz = zeros(T,size) 
  yz = zeros(T,size)
  zz = zzData
  return MetricTensor{T,N}(xx,xy,yy,xz,yz,zz,size)
end

function IdentityMetric(::Type{T},size::SizeT) where T <: Real where SizeT <: Union{Int,Tuple}
  N = Base.length(size)
  return MetricTensor(ones(T,size),ones(T,size),ones(T,size))
end

#=
# Define the constructors
=#

mutable struct Fieldline{T<:Real}
  x1::Vector{Float64}
  x2::Vector{Float64}
  x3::Vector{Float64}
  B::Vector3D{T}
  coBasis_x1::Vector3D{T}
  coBasis_x2::Vector3D{T}
  coBasis_x3::Vector3D{T}
  contraBasis_x1::Vector3D{T}
  contraBasis_x2::Vector3D{T}
  contraBasis_x3::Vector3D{T}
  metric::MetricTensor{T,1}
  jacobian::Vector{T}
  curvature::Vector3D{T}
  torsion::Vector3D{T}
  length::Int
  rotationalTransform::Float64
  Fieldline{T}() where T <: Real = new()
end
