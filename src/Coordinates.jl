
"""
    VmecCoordinates{D,T,N}

Composite type containing coordinates and the covariant and contravariant basis vectors in with respect to the VMEC coordinates.

# Fields
- `data::AbstractArray{NTuple{D,T},N}`: Coordinates in VMEC coordinates for each point in the field
- `eS::VectorField{D,T,N}`: Covariant basis vector with respect to `s` coordinate
- `eThetaVmec::VectorField{D,T,N}`: Covariant basis vector with respect to `θᵥ` coordinate
- `eZeta::VectorField{D,T,N}`: Covariant basis vector with respect to `ζ` coordinate
- `gradS::VectorField{D,T,N}`: Contravariant basis vector with respect to `s` coordinate
- `gradThetaVmec::VectorField{D,T,N}`: Contravariant basis vector with respect to `θᵥ` coordinate
- `gradZeta::VectorField{D,T,N}`: Contravariant basis vector with respect to `ζ` coordinate
"""
struct VmecCoordinates{D,T,N} <: AbstractCoordinateField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
  eS::VectorField{D,T,N}
  eThetaVmec::VectorField{D,T,N}
  eZeta::VectorField{D,T,N}
  gradS::VectorField{D,T,N}
  gradThetaVmec::VectorField{D,T,N}
  gradZeta::VectorField{D,T,N}
end

"""
    PestCoordinates{D,T,N}

Composite type containing coordinates and the covariant and contravariant basis vectors in with respect to PEST coordinates.

# Fields
- `data::AbstractArray{NTuple{D,T},N}`: Coordinates in VMEC coordinates for each point in the field
- `ePsi::VectorField{D,T,N}`: Covariant basis vector with respect to `ψ` coordinate
- `eAlpha::VectorField{D,T,N}`: Covariant basis vector with respect to `α` coordinate
- `eZeta::VectorField{D,T,N}`: Covariant basis vector with respect to `ζ` coordinate
- `gradPsi::VectorField{D,T,N}`: Contravariant basis vector with respect to `ψ` coordinate
- `gradAlpha::VectorField{D,T,N}`: Contravariant basis vector with respect to `α` coordinate
- `gradZeta::VectorField{D,T,N}`: Contravariant basis vector with respect to `ζ` coordinate
"""
struct PestCoordinates{D,T,N} <: AbstractCoordinateField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
  ePsi::VectorField{D,T,N}
  eAlpha::VectorField{D,T,N}
  eZeta::VectorField{D,T,N}
  gradPsi::VectorField{D,T,N}
  gradAlpha::VectorField{D,T,N}
  gradZeta::VectorField{D,T,N}
end

"""
    SThetaZetaCoordinates{D,T,N}

Composite type containing coordinates and the covariant and contravariant basis vectors in with respect to the (s,θ,ζ) coordinates.

# Fields
- `data::AbstractArray{NTuple{D,T},N}`: Coordinates in VMEC coordinates for each point in the field
- `eS::VectorField{D,T,N}`: Covariant basis vector with respect to `s` coordinate
- `eTheta::VectorField{D,T,N}`: Covariant basis vector with respect to `θ` coordinate
- `eZeta::VectorField{D,T,N}`: Covariant basis vector with respect to `ζ` coordinate
- `gradS::VectorField{D,T,N}`: Contravariant basis vector with respect to `s` coordinate
- `gradTheta::VectorField{D,T,N}`: Contravariant basis vector with respect to `θ` coordinate
- `gradZeta::VectorField{D,T,N}`: Contravariant basis vector with respect to `ζ` coordinate
"""
struct SThetaZetaCoordinates{D,T,N} <: AbstractCoordinateField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
  eS::VectorField{D,T,N}
  eTheta::VectorField{D,T,N}
  eZeta::VectorField{D,T,N}
  gradS::VectorField{D,T,N}
  gradTheta::VectorField{D,T,N}
  gradZeta::VectorField{D,T,N}
end


"""
    eX1(::AbstractCoordinateField{D,T,N}) where {D,T,N}

Returns the covariant basis vector with respect to the first coordinate (i.e. X1)

# Returns
`VectorField{D,T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

eX1 = (∂x/∂x1,∂y/∂x1,∂z/∂x1)
"""
function eX1(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,2)
end

"""
    eX1(::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer

Returns the component `index` of the covariant basis vector with respect to the first coordinate (i.e. X1)

# Returns
`Array{T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

eX1 = (∂x/∂x1,∂y/∂x1,∂z/∂x1)
"""
function eX1(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,2),index)
end

"""
    eX2(::AbstractCoordinateField{D,T,N}) where {D,T,N}

Returns the covariant basis vector with respect to the second coordinate (i.e. X2)

# Returns
`VectorField{D,T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

eX2 = (∂x/∂x2,∂y/∂x2,∂z/∂x2)
"""
function eX2(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,3)
end

"""
    eX2(::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer

Returns the component `index` of the covariant basis vector with respect to the second coordinate (i.e. X2)

# Returns
`Array{T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

eX2 = (∂x/∂x2,∂y/∂x2,∂z/∂x2)
"""
function eX2(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,3),index)
end

"""
    eX3(::AbstractCoordinateField{D,T,N}) where {D,T,N}

Returns the covariant basis vector with respect to the third coordinate (i.e. X3)

# Returns
`VectorField{D,T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

eX3 = (∂x/∂x3,∂y/∂x3,∂z/∂x3)
"""
function eX3(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,4)
end

"""
    eX3(::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer

Returns the component `index` of the covariant basis vector with respect to the third coordinate (i.e. X3)

# Returns
`Array{T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

eX3 = (∂x/∂x3,∂y/∂x3,∂z/∂x3)
"""
function eX3(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,4),index)
end

"""
    gradX1(::AbstractCoordinateField{D,T,N}) where {D,T,N}

Returns the contravariant basis vector with respect to the first coordinate (i.e. X1)

# Returns
`VectorField{D,T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

gradX1 = (∂x/∂x1+∂y/∂x1+∂z/∂x1)*∇x1
"""
function gradX1(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,5)
end

"""
    gradX1(::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer

Returns the component `index` of the contravariant basis vector with respect to the first coordinate (i.e. X1)

# Returns
`Array{T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

gradX1 = (∂x/∂x1+∂y/∂x1+∂z/∂x1)*∇x1
"""
function gradX1(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,5),index)
end

"""
    gradX2(::AbstractCoordinateField{D,T,N}) where {D,T,N}

Returns the contravariant basis vector with respect to the second coordinate (i.e. X2)

# Returns
`VectorField{D,T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

gradX2 = (∂x/∂x2+∂y/∂x2+∂z/∂x2)*∇x2
"""
function gradX2(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,6)
end

"""
    gradX2(::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer

Returns the component `index` of the contravariant basis vector with respect to the second coordinate (i.e. X2)

# Returns
`Array{T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

gradX2 = (∂x/∂x2+∂y/∂x2+∂z/∂x2)*∇x2
"""
function gradX2(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,6),index)
end

"""
    gradX3(::AbstractCoordinateField{D,T,N}) where {D,T,N}

Returns the contravariant basis vector with respect to the third coordinate (i.e. X3)

# Returns
`VectorField{D,T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

gradX3 = (∂x/∂x3+∂y/∂x3+∂z/∂x3)*∇x3
"""
function gradX3(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,7)
end

"""
    gradX3(::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer

Returns the component `index` of the contravariant basis vector with respect to the third coordinate (i.e. X3)

# Returns
`Array{T,N}`

# Examples
For a coordinate mapping X = (x(x1,x2,x3),y(x1,x2,x3),z(x1,x2,x3)):

gradX3 = (∂x/∂x3+∂y/∂x3+∂z/∂x3)*∇x3
"""
function gradX3(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,7),index)
end

