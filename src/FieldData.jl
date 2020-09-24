import Base.+, Base.-, Base.*, Base./, Base.abs, Base.sqrt
import Base.eltype, Base.size, Base.length, Base.getindex, Base.setindex!, Base.copyto!

abstract type AbstractFieldData end

struct FieldData{D,T,N} <: AbstractFieldData
  data::Array{NTuple{D,T},N}
end

function eltype(F::AbstractFieldData)
  return Base.eltype(getfield(F,:data))
end

function size(F::AbstractFieldData)
  return Base.size(getfield(F,:data))
end

function length(F::AbstractFieldData)
  return Base.length(getfield(F,:data))
end

function getindex(F::AbstractFieldData,inds...) where N
  return getfield(F,:data)[inds...]
end

function setindex!(F::AbstractFieldData,value::NTuple{D,T},inds...) where {D,T,N}
  getfield(F,:data)[inds...] = value
end

function copyto!(dest::NTuple{D,T},src::NTuple{D,T}) where {D,T}
  map!(i->i,dest,src)
  return dest
end

function VectorField(xData::AbstractArray{T,N},yData::AbstractArray{T,N},zData::AbstractArray{T,N}) where {T,N}
  @assert size(xData) == size(yData) == size(zData) "Incompatible array sizes"

  data = Array{NTuple{3,T},N}(undef,size(xData))
  map!((i,j,k)->tuple(i,j,k),data,xData,yData,zData)
  return FieldData{3,T,N}(data)
end

function ScalarField(xData::AbstractArray{T,N}) where {T,N}
  return FieldData{1,T,N}(tuple.(xData))
end

function cross(a::NTuple{3,T},b::NTuple{3,T}) where T
  return tuple(a[2]*b[3]-b[2]*a[3],a[3]*b[1]-b[3]*a[1],a[1]*b[2]-b[1]*a[2])
end

function cross(u::FieldData{3,T,N},v::FieldData{3,T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible vector sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->cross(a,b),data,getfield(u,:data),getfield(v,:data))
  return FieldData{3,T,N}(data)
end

function cross!(w::FieldData{3,T,N}, u::FieldData{3,T,N},v::FieldData{3,T,N}) where {T,N}
  @assert size(u) == size(v) == size(w) "Incompatible vector sizes"
  map!((a,b)->cross(a,b),getfield(w,:data),getfield(u,:data),getfield(v,:data))
end

function dot(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return mapreduce((i,j)->i*j,+,a,b;init=zero(T))
end

function dot(u::FieldData{D,T,N},v::FieldData{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  result = Array{T,N}(undef,size(u))
  map!((a,b)->dot(a,b),result,getfield(u,:data),getfield(v,:data))
  return result
end

function +(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i+j,a,b)
end

function +(u::FieldData{D,T,N},v::FieldData{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a+b,data,getfield(u,:data),getfield(v,:data))
  return FieldData{D,T,N}(data)
end

function -(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i-j,a,b)
end

function -(u::FieldData{D,T,N},v::FieldData{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a-b,data,getfield(u,:data),getfield(v,:data))
  return FieldData{D,T,N}(data)
end

function *(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i*j,a,b)
end

function *(a::NTuple{D,T},c::T) where {D,T}
  return map(i->c*i,a)
end

function *(c::T,a::NTuple{D,T}) where {D,T}
  return a*c
end

function *(u::FieldData{D,T,N},v::FieldData{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),getfield(v,:data))
  return FieldData{D,T,N}(data)
end

function *(u::FieldData{D,T,N},v::AbstractArray{T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),v)
  return FieldData{D,T,N}(data)
end

function *(v::AbstractArray{T,N},u::FieldData{D,T,N}) where {D,T,N}
  return u*v
end

function *(u::FieldData{D,T,N},v::T) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!(a->v*a,data,getfield(u,:data))
  return FieldData{D,T,N}(data)
end

function *(v::T,u::FieldData{D,T,N}) where {D,T,N}
  return u*v
end

function /(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i/j,a,b)
end

function /(a::NTuple{D,T},c::T) where {D,T}
  return map(i->i/c,a)
end

function /(u::FieldData{D,T,N},v::FieldData{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a/b,data,getfield(u,:data),getfield(v,:data))
  return FieldData{D,T,N}(data)
end

function /(u::FieldData{D,T,N},v::AbstractArray{T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a/b,data,getfield(u,:data),v)
  return FieldData{D,T,N}(data)
end

function /(u::FieldData{D,T,N},v::T) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!(a->a/v,data,getfield(u,:data))
  return FieldData{D,T,N}(data)
end

function abs(a::NTuple{D,T}) where {D,T}
  res = zero(T)
  for i = 1:length(a)
    res += a[i]^2
  end
  return sqrt(res)
end

function abs(u::FieldData{D,T,N}) where {D,T,N}
  data = Array{T,N}(undef,size(u))
  map!(a->abs(a),data,getfield(u,:data))
  return ScalarField(data)
end

function component(u::FieldData{D,T,N},index::I) where I <: Number where {D,T,N}
  res = Array{T,N}(undef,size(u))
  map!(i->i[index],res,getfield(u,:data))
  return res
end

function component!(x::Array{T,N},u::FieldData{D,T,N},index::I) where I <: Number where {D,T,N}
  map!(i->i[index],x,getfield(u,:data))
end
