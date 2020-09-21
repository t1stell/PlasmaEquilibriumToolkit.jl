import Interpolations

struct Vector3D{T<:AbstractFloat}
  x::Vector{T}
  y::Vector{T}
  z::Vector{T}
  length::Int
  function Vector3D{T}(xVec::Vector{T},yVec::Vector{T},zVec::Vector{T},component=:contravariant,basis=:xyz) where T<:AbstractFloat
    @assert Base.length(xVec) == Base.length(yVec) == Base.length(zVec) "Vectors have incompatible lengths"
    x = xVec
    y = yVec
    z = zVec
    length = Base.length(x)
    return new(x,y,z,length)
  end
end


struct Vector3DSpline{T<:AbstractFloat}
  t::Vector{T}
  xSpline::Interpolations.Extrapolation
  ySpline::Interpolations.Extrapolation
  zSpline::Interpolations.Extrapolation
  function Vector3DSpline{T}(u::Vector3D{T},tVec::Vector{TT}) where T <: Real where TT <: Real
    @assert Base.length(tVec) == getfield(u,:length) "Vectors have incompatible sizes"
    uLength = getfield(u,:length)
    t = tVec
    if sum(t .- collect(t[1]:t[2]-t[1]:t[uLength])) <= uLength*eps(T)
      tRange = range(t[1],step=t[2]-t[1],stop=t[uLength])
      xSpline = Interpolations.CubicSplineInterpolation(tRange,getfield(u,:x))
      ySpline = Interpolations.CubicSplineInterpolation(tRange,getfield(u,:y))
      zSpline = Interpolations.CubicSplineInterpolation(tRange,getfield(u,:z))
    else
      xSpline = Interpolations.LinearInterpolation(t,getfield(u,:x))
      ySpline = Interpolations.LinearInterpolation(t,getfield(u,:y))
      zSpline = Interpolations.LinearInterpolation(t,getfield(u,:z))
    end
    return new(t,xSpline,ySpline,zSpline)
  end
end


#=
# Define the constructors
=#

struct Surface3D{T<:AbstractFloat}
  x::Array{T,2}
  y::Array{T,2}
  z::Array{T,2}
  size::Tuple{Int,Int}
  componentType::Symbol
  function Surface3D{T}(dim1::Int=0,dim2::Int=0,component::Symbol=:co) where T<:AbstractFloat
    x = Array{T}(undef,dim1,dim2)
    y = Array{T}(undef,dim1,dim2)
    z = Array{T}(undef,dim1,dim2)
    size = tuple(dim1,dim2) 
    return new(x,y,z,size,component)
  end
  function Surface3D{T}(dims::Tuple{Int,Int},component::Symbol=:co) where T <: Real
    return Surface3D{T}(dims[1],dims[2],component)
  end
  function Surface3D{T}(xArr::Array{T,2},yArr::Array{T,2},zArr::Array{T,2},component::Symbol=:co) where T<:AbstractFloat
    @assert Base.size(xArr) == Base.size(yArr) "xArr and yVec have incompatible lengths"
    @assert Base.size(yArr) == Base.size(zArr) "yVec and zVec have incompatible lengths"
    x = xArr
    y = yArr
    z = zArr
    size = Base.size(x)
    return new(x,y,z,size,component)
  end

end


struct MetricTensor{T<:AbstractFloat,N}
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
"""
  Fieldline{T}

  Defines a struct that contains all the relevant information for a magnetic field line, including
  the vector B, the covariant and contravariant components, the metric tensor and the jacobian
  along fieldline
"""
mutable struct Fieldline{T<:AbstractFloat}
  x::Vector{T}
  y::Vector{T}
  z::Vector{T}
  B::Vector3D{T}
  coBasis_x::Vector3D{T}
  coBasis_y::Vector3D{T}
  coBasis_z::Vector3D{T}
  contraBasis_x::Vector3D{T}
  contraBasis_y::Vector3D{T}
  contraBasis_z::Vector3D{T}
  metric::MetricTensor{T,1}
  jacobian::Vector{T}
  curvature::Vector3D{T}
  torsion::Vector3D{T}
  length::Int
  rotationalTransform::Float64
  Fieldline{T}() where T <: Real = new()
end
