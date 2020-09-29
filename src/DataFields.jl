import Base.+, Base.-, Base.*, Base./, Base.abs, Base.sqrt
import Base.eltype, Base.size, Base.length, Base.getindex, Base.setindex!, Base.copyto!

#=
# Provide abstract types that can be extended
=#
abstract type AbstractDataField{T,N} end
abstract type AbstractVectorField{D,T,N} <: AbstractDataField{T,N} end
abstract type AbstractCoordinateField{D,T,N} <: AbstractDataField{T,N} end

struct VectorField{D,T,N} <: AbstractVectorField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
end

struct ScalarField{T,N} <: AbstractDataField{T,N}
  data::AbstractArray{T,N}
end

# The TupleField has the same structure, but is
# distinct from VectorField so that vector operations
# can be limited to VectorFields
struct TupleField{D,T,N} <: AbstractDataField{T,N}
  data::AbstractArray{NTuple{D,T},N}
end

# Functions for operating on AbstractDataFields
function eltype(F::AbstractDataField)
  return Base.eltype(getfield(F,:data))
end

function size(F::AbstractDataField)
  return Base.size(getfield(F,:data))
end

function length(F::AbstractDataField)
  return Base.length(getfield(F,:data))
end

function getindex(F::AbstractDataField,inds...) where N
  return getfield(F,:data)[inds...]
end

function setindex!(F::AbstractDataField{T,N},value::VT,inds...) where VT <: Union{T,NTuple{D,T}} where {D,T,N}
  getfield(F,:data)[inds...] = value
end

"""
    copyto!(dest::NTuple{D,T},src::NTuple{D,T}) where {D,T}

Extends the Base.copyto! function to NTuples of the same dimension D and type T
"""
function copyto!(dest::NTuple{D,T},src::NTuple{D,T}) where {D,T}
  map!(i->i,dest,src)
  return dest
end


# Constructors for the different data types
"""
    VectorField(dataFields::Vararg{AbstractArray{T,N},D})
  
Constructs a D-dimensional vector field of type T over N dimensions from the data
specified by dataFields

# Examples
```julia-repl
julia> x = reshape(collect(Float64,1:9),(3,3)); y = x; z = x;

julia> v = VectorField(x,y,z);

julia> typeof(v)
VectorField{3,Float64,2}

julia> v[1]
(1.0,1.0,1.0)

```
"""
function VectorField(dataFields::Vararg{AbstractArray{T,N},D}) where {D,T,N}
  @assert D >= 1 "VectorField requires at least one component!"
  if D > 1
    for i = 1:D-1
      @assert size(dataFields[i]) == size(dataFields[i+1]) "Incompatible array sizes"
    end
  end
  data = Array{NTuple{D,T},N}(undef,size(dataFields[1]))
  map!(inds::Vararg{T,D}->inds,data,dataFields...)
  return VectorField{D,T,N}(data)
end

function TupleField(dataFields::Vararg{AbstractArray{T,N},D}) where {D,T,N}
  @assert D >= 1 "TupleField requires at least one component!"
  if D > 1
    for i = 1:D-1
      @assert size(dataFields[i]) == size(dataFields[i+1]) "Incompatible array sizes"
    end
  end
  data = Array{NTuple{D,T},N}(undef,size(dataFields[1]))
  map!(inds::Vararg{T,D}->inds,data,dataFields...)
  return TupleField{D,T,N}(data)
end

function cross(a::NTuple{3,T},b::NTuple{3,T}) where T
  return tuple(a[2]*b[3]-b[2]*a[3],a[3]*b[1]-b[3]*a[1],a[1]*b[2]-b[1]*a[2])
end

function cross(u::VectorField{3,T,N},v::VectorField{3,T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible vector sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->cross(a,b),data,getfield(u,:data),getfield(v,:data))
  return VectorField{3,T,N}(data)
end

function cross!(w::VectorField{3,T,N}, u::VectorField{3,T,N},v::VectorField{3,T,N}) where {T,N}
  @assert size(u) == size(v) == size(w) "Incompatible vector sizes"
  map!((a,b)->cross(a,b),getfield(w,:data),getfield(u,:data),getfield(v,:data))
end

function dot(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return mapreduce((i,j)->i*j,+,a,b;init=zero(T))
end

function dot(u::VectorField{D,T,N},v::VectorField{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  result = Array{T,N}(undef,size(u))
  map!((a,b)->dot(a,b),result,getfield(u,:data),getfield(v,:data))
  return result
end

function +(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i+j,a,b)
end

function +(u::VectorField{D,T,N},v::VectorField{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a+b,data,getfield(u,:data),getfield(v,:data))
  return VectorField{D,T,N}(data)
end

function -(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i-j,a,b)
end

function -(u::VectorField{D,T,N},v::VectorField{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a-b,data,getfield(u,:data),getfield(v,:data))
  return VectorField{D,T,N}(data)
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

function *(u::VectorField{D,T,N},v::VectorField{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),getfield(v,:data))
  return VectorField{D,T,N}(data)
end

function *(u::VectorField{D,T,N},v::AbstractArray{T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),v)
  return VectorField{D,T,N}(data)
end

function *(v::AbstractArray{T,N},u::VectorField{D,T,N}) where {D,T,N}
  return u*v
end

function *(u::VectorField{D,T,N},v::T) where {D,T,N}
  data = similar(getfield(u,:data))
  map!(a->v*a,data,getfield(u,:data))
  return VectorField{D,T,N}(data)
end

function *(v::T,u::VectorField{D,T,N}) where {D,T,N}
  return u*v
end

function *(u::ScalarField{T,N},v::ScalarField{T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),getfield(v,:data))
  return data
end

function /(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i/j,a,b)
end

function /(a::NTuple{D,T},c::T) where {D,T}
  return map(i->i/c,a)
end

function /(u::VectorField{D,T,N},v::VectorField{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a/b,data,getfield(u,:data),getfield(v,:data))
  return VectorField{D,T,N}(data)
end

function /(u::VectorField{D,T,N},v::AbstractArray{T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a/b,data,getfield(u,:data),v)
  return VectorField{D,T,N}(data)
end

function /(u::VectorField{D,T,N},v::T) where {D,T,N}
  data = similar(getfield(u,:data))
  map!(a->a/v,data,getfield(u,:data))
  return VectorField{D,T,N}(data)
end

function abs(a::NTuple{D,T}) where {D,T}
  res = zero(T)
  for i = 1:length(a)
    res += a[i]^2
  end
  return sqrt(res)
end

function abs(u::VectorField{D,T,N}) where {D,T,N}
  data = Array{T,N}(undef,size(u))
  map!(a->abs(a),data,getfield(u,:data))
  return data
end

"""
    component(::VectorField{D,T,N},index::I)

Extract the component of the VectorField specified by the index, the result is returned
in an Array of type T with dimension N
"""
function component(u::VectorField{D,T,N},index::I) where I <: Number where {D,T,N}
  res = Array{T,N}(undef,size(u))
  map!(i->i[index],res,getfield(u,:data))
  return res
end

"""
    component!(x::Array{T,N},u::VectorField{D,T,N},index::I)

Extract the component of the Vectorfield u specified by the index to the preallocated array x
"""
function component!(x::Array{T,N},u::VectorField{D,T,N},index::I) where I <: Number where {D,T,N}
  map!(i->i[index],x,getfield(u,:data))
end

function component(a::Array{NTuple{D,T},N},index::I) where I <: Number where {D,T,N}
  res = Array{T,N}(undef,size(a))
  map!(i->i[index],res,a)
  return res
end

function component(u::TupleField{D,T,N},index::I) where I <: Number where {D,T,N}
  res = Array{T,N}(undef,size(u))
  map!(i->i[index],res,getfield(u,:data))
  return res
end

