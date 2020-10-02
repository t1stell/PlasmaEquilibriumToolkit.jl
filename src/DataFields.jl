import Base.+, Base.-, Base.*, Base./, Base.abs, Base.sqrt
import Base.eltype, Base.size, Base.length, Base.getindex, Base.setindex!, Base.copyto!, Base.iterate

#=
# Provide abstract types that can be extended
=#
"""
    AbstractDataField{T,N}

Supertype abstract representation of data fields of arbitrary type T over N dimensions.
All subtypes of AbstractDataField have a member labeled `data` containing the data 
"""
abstract type AbstractDataField{T,N} end

"""
    AbstractTupleField{D,T,N}

Abstract type for AbstractData specialized to data representable by tuples with D components.
Data types representable by an AbstractTupleField are three-dimensional coordinates or three dimensional vector fields.
"""
abstract type AbstractTupleField{D,T,N} <: AbstractDataField{T,N} end

"""
    AbstractVectorField{D,T,N}

Abstract type for AbstractTupleData specialized to vector fields with D components.
Specializing the AbstractVectorField allows for defining operations that act on subtypes of AbstractVectorField but AbstractTupleField.
"""
abstract type AbstractVectorField{D,T,N} <: AbstractTupleField{D,T,N} end

"""
    AbstractCoordinateField{D,T,N}

Abstract type for specializing coordinates, such as magnetic coordinates
"""
abstract type AbstractCoordinateField{D,T,N} <: AbstractDataField{T,N} end

"""
    VectorField{D,T,N}

Data type to represent a D-component vector field of type T in N dimensions.  A subtype
of `AbstractVectorField{D,T,N}`

# Examples

```julia-repl
julia> VectorField{3,Float64,3} <: AbstractVectorField{3,Float64,3}
true

julia> VectorField{3,Float64,1} <: AbstractDataField{Float64,3}
false
```

See also: [`TupleField`](@ref)
"""
struct VectorField{D,T,N} <: AbstractVectorField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
end

"""
    ScalarField{T,N}

Wraps an N-dimensional array of type T.
Examples of data naturally represented by type ScalarField are the magnitude of the magnetic field and the scalar Jacobian of coordinate transformations.
"""
struct ScalarField{T,N} <: AbstractDataField{T,N}
  data::AbstractArray{T,N}
end

# The TupleField has the same structure, but is
# distinct from VectorField so that vector operations
# can be limited to VectorFields
"""
    TupleField{D,T,N}

Data type for data that can be decomposed into tuples of size D with type T over N-dimensions.
The underlying data layout is identical to `VectorField{D,T,N}`.
Provides safety against applying operations that can act on VectorField data, such as the cross product, from acting on TupleField data.

# Examples
```julia-repl
julia> data = fill(tuple(randn(3)...),5);
julia> typeof(data) <: Array{NTuple{3,Float64},1}
true

julia> tf = TupleField{3,Float64,1}(data); vf = VectorField{3,Float64,1}(data);
julia> typeof(vf) <: typeof(tf)
false

julia> typeof(vf.data) <: typeof(tf.data)
true
```

See also: [`VectorField`](@ref)
"""
struct TupleField{D,T,N} <: AbstractTupleField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
end

# Functions for operating on AbstractDataFields
"""
    eltype(F::AbstractDataField)

Extends the eltype function by apply eltype to the data field of the `AbstractFieldData`
"""
function eltype(F::AbstractDataField)
  return Base.eltype(getfield(F,:data))
end

"""
    size(F::AbstractDataField)

Extends the size function by apply size to the data field of the `AbstractFieldData`
"""
function size(F::AbstractDataField)
  return Base.size(getfield(F,:data))
end

"""
    length(F::AbstractDataField)

Extends the length function by apply length to the data field of the `AbstractFieldData`
"""
function length(F::AbstractDataField)
  return Base.length(getfield(F,:data))
end

"""
    getindex(F::AbstractDataField,inds::Vararg{I}) where I <: Number

Extends the getindex function by apply getindex to the data field of the `AbstractFieldData`
"""
function getindex(F::AbstractDataField,inds::Vararg{I}) where I <: Number
  return getfield(F,:data)[inds...]
end

"""
    setindex(F::AbstractDataField,value::VT,inds::Vararg{I}) where VT <: Union{T,NTuple{D,T}} where {D,T,N} where I <: Number

Extends the setindex function by apply setindex to the data field of the `AbstractFieldData`
"""
function setindex!(F::AbstractDataField{T,N},value::VT,inds::Vararg{I}) where VT <: Union{T,NTuple{D,T}} where {D,T,N} where I <: Number
  getfield(F,:data)[inds...] = value
end

"""
    iterate(F::AbstractDataField,step=1)

Extends the iterate function by apply iterate to the data field of the `AbstractFieldData`
"""
function iterate(F::AbstractDataField{T,N},i=1) where {T,N}
  return Base.iterate(getfield(F,:data),i)
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

"""
    VectorField(c::I,::Type{T},dims::Vararg{I}) where I <: Number where T

Construct an uninitialized vector field with `C` components of type `T` with the size specified by `size`
"""
function VectorField(c::I,::Type{T},size::Vararg{I}) where I <: Number where T
  @assert c >= 1 "VectorField requires at least one component!"
  D = c
  N = length(size)
  data = Array{NTuple{D,T},N}(undef,size...)
  return VectorField{D,T,N}(data)
end

"""
    TupleField(dataFields::Vararg{AbstractArray{T,N},D})
  
Constructs a field tuples of length D of type T over N dimensions from the data
specified by dataFields

# Examples
```julia-repl
julia> x = reshape(collect(Float64,1:9),(3,3)); y = x; z = x;

julia> v = TupleField(x,y,z);

julia> typeof(v)
TupleField{3,Float64,2}

julia> v[1]
(1.0,1.0,1.0)

```
"""
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

"""
    TupleField(c::I,::Type{T},dims::Vararg{I}) where I <: Number where T

Construct an uninitialized tuple field with `C` components of type `T` with the size specified by `size`
"""
function TupleField(c::I,::Type{T},size::Vararg{I}) where I <: Number where T
  @assert c >= 1 "VectorField requires at least one component!"
  D = c
  N = length(size)
  data = Array{NTuple{D,T},N}(undef,size...)
  return TupleField{D,T,N}(data)
end

function cross(a::NTuple{3,T},b::NTuple{3,T}) where T
  return tuple(a[2]*b[3]-b[2]*a[3],a[3]*b[1]-b[3]*a[1],a[1]*b[2]-b[1]*a[2])
end

"""
    cross(u::VectorField{3,T,N},v::VectorField{3,T,N}) where {T,N}

Performs the element-wise cross product between two 3-dimensional vector fields
where the components of the vectors are given in 3D Cartesian representation.
The cross product is only defined for 3 component vector fields.

# Examples
```julia-repl
julia> x = collect(Float64,1:6); y = collect(Float64,7:12); z = collect(Float64,-6:-1);
julia> u = VectorField(x,y,z); v = VectorField(y,z,x);
julia> cross(u,v)
VectorField{3,Float64,1}([(-29.0, -43.0, -55.0), (-9.0,-44.0, -74.0), (11.0, -45.0, -93.0), (31.0, -46.0, -112.0), (51.0, -47.0, -131.0), (71.0, -48.0, -150.0)])
```
"""
function cross(u::VectorField{3,T,N},v::VectorField{3,T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible vector sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->cross(a,b),data,getfield(u,:data),getfield(v,:data))
  return VectorField{3,T,N}(data)
end

"""
    cross!(w::VectorField{3,T,N},u::VectorField{3,T,N},v::VectorField{3,T,N}) 

Fills the preallocated VectorField `w` with the result of the cross product of `u` and `v`
"""
function cross!(w::VectorField{3,T,N}, u::VectorField{3,T,N},v::VectorField{3,T,N}) where {T,N}
  @assert size(u) == size(v) == size(w) "Incompatible vector sizes"
  map!((a,b)->cross(a,b),getfield(w,:data),getfield(u,:data),getfield(v,:data))
end

function dot(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return mapreduce((i,j)->i*j,+,a,b;init=zero(T))
end

"""
    dot(u::VectorField{3,T,N},v::VectorField{3,T,N}) where {T,N}

Performs the element-wise dot product between two 3-dimensional vector fields
where the components of the vectors are given in 3D Cartesian representation.
The dot product is defined for any number of components.

# Examples
```julia-repl
julia> x = collect(Float64,1:6); y = collect(Float64,7:12); z = collect(Float64,-6:-1);
julia> u = VectorField(x,y,z); v = VectorField(y,z,x);
julia> dot(u,v)
6-element Array{Float64,1}:
 -41.0
 -34.0
 -21.0
  -2.0
  23.0
  54.0
```
"""
function dot(u::VectorField{D,T,N},v::VectorField{D,T,N}) where {D,T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  result = Array{T,N}(undef,size(u))
  map!((a,b)->dot(a,b),result,getfield(u,:data),getfield(v,:data))
  return ScalarField{T,N}(result)
end

function +(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i+j,a,b)
end

function +(a::NTuple{D,T},b::T) where {D,T}
  return a .+ b
end

function +(b::T,a::NTuple{D,T}) where {D,T}
  return a .+ b
end

function +(u::AbstractDataField{T,N},v::AbstractDataField{T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a+b,data,getfield(u,:data),getfield(v,:data))
  return typeof(u)(data)
end

function +(u::AbstractDataField{T,N},v::T) where {T,N}
  data = similar(getfield(u,:data))
  map!(i->i*v,data,getfield(u,:data))
  return typeof(u)(data)
end

function -(a::NTuple{D,T},b::NTuple{D,T}) where {D,T}
  return map((i,j)->i-j,a,b)
end

function -(a::NTuple{D,T},b::T) where {D,T}
  return a - b
end

function -(b::T,a::NTuple{D,T}) where {D,T}
  return b - a
end

function -(u::AbstractDataField{T,N},v::AbstractDataField{T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a-b,data,getfield(u,:data),getfield(v,:data))
  return typeof(u)(data)
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

function *(u::AbstractDataField{T,N},v::AbstractDataField{T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),getfield(v,:data))
  return typeof(u)(data)
end

function *(u::AbstractDataField,v::AbstractArray{T,N}) where {T,N}
  @assert size(u) == size(v) "Incompatible field sizes"
  data = similar(getfield(u,:data))
  map!((a,b)->a*b,data,getfield(u,:data),v)
  return typeof(u)(data)
end

function *(v::AbstractArray{T,N},u::AbstractDataField) where {T,N}
  return u*v
end

function *(u::AbstractDataField{T,N},v::T) where {T,N}
  data = similar(getfield(u,:data))
  map!(a->v*a,data,getfield(u,:data))
  return typeof(u)(data)
end

function *(v::T,u::AbstractDataField) where T
  return u*v
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

function /(u::VectorField{D,T,N},v::ScalarField{T,N}) where {D,T,N}
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
    component(::AbstractTupleField{D,T,N},index::I)

Extract the component of the AbstractTupleField specified by the index, the result is returned
in an Array of type T with dimension N
"""
function component(u::AbstractTupleField{D,T,N},index::I) where I <: Number where {D,T,N}
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

