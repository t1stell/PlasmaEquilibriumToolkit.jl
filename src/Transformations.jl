import Base.abs
using LinearAlgebra

abstract type BasisTransformation <: Transformation end

#Define the singleton types for the coordinate transformations.  All transformations are
#subtypes of the abstract supertype Transformation provided by CoordinateTransformations.jl.
struct FluxFromPest <: Transformation; end
struct PestFromFlux <: Transformation; end
struct CylindricalFromFlux <: Transformation; end
struct CylindricalFromPest <: Transformation; end
struct CartesianFromFlux <: Transformation; end
struct CartesianFromPest <: Transformation; end
struct ContravariantFromCovariant <: BasisTransformation; end
struct CovariantFromContravariant <: BasisTransformation; end

#Define the singleton types 
struct Covariant end
struct Contravariant end

"""
    abs(e::BasisVectors{T}[,component=0)

Compute the L2 norm of the basis vector `e`.  If `component` > 0 and <= 3, the L2 norm of the component is returned.

#  Examples
```jldoctest
julia> using StaticArrays

julia> e = @SArray([1 4 7;2 5 8;3 6 9]);

julia> abs(e)
16.881943016134134

julia> abs(e,1)
3.7416573867739413

```
"""
function abs(e::BasisVectors{T},component::Int=0) where T
  if component > 0 && component <= 3
    return norm(e[:,component])
  else
    return norm(e)
  end
end


"""
    jacobian(::Covariant,e::BasisVectors{T})

Compute the Jacobian of the covariant basis vectors `e` given by ``J = e₁ ⋅ e₂ × e₃``.
"""
function jacobian(::Covariant,e::BasisVectors{T}) where T
  return dot(e[:,1],cross(e[:,2],e[:,3]))
end

"""
    jacobian(::Contravariant,e::BasisVectors{T})

Compute the Jacobian of the contracovariant basis vectors `e` given by ``J = 1.0/(∇e₁ ⋅ ∇e₂ × ∇e₃)``.
"""
function jacobian(::Contravariant, e::BasisVectors{T}) where T
  return 1.0 /dot(e[:,1],cross(e[:,2],e[:,3]))
end


"""
transform_basis()
"""
function transform_basis(::ContravariantFromCovariant,e::BasisVectors{T},J::T) where T
  grad_1 = cross(e[:,2],e[:,3])/J
  grad_2 = cross(e[:,3],e[:,1])/J
  grad_3 = cross(e[:,1],e[:,2])/J
  return hcat(grad_1,grad_2,grad_3)
end

function transform_basis(::CovariantFromContravariant,e::BasisVectors{T},J::T) where T
  e_1 = cross(e[:,2],e[:,3])*J
  e_2 = cross(e[:,3],e[:,1])*J
  e_3 = cross(e[:,1],e[:,2])*J
  return hcat(e_1,e_2,e_3)
end

function transform_basis(::ContravariantFromCovariant,e::BasisVectors{T}) where T
  J = jacobian(Covariant(),e)
  return transform_basis(ContravariantFromCovariant(),e,J)
end

function transform_basis(::CovariantFromContravariant,e::BasisVectors{T}) where T
  J = jacobian(Contravariant(),e)
  return transform_basis(CovariantFromContravariant(),e,J)
end

function transform_basis(t::Transformation,e::AbstractArray{BasisVectors{T}},J::AbstractArray{T}) where T
  res = similar(e)
  Threads.@threads for i = 1:length(e)
    res[i] = transform_basis(t,e[i],J[i])
  end
  return res
end

function transform_basis(t::Transformation,e::AbstractArray{BasisVectors{T}}) where T
  res = similar(e)
  Threads.@threads for i = 1:length(e)
    res[i] = transform_basis(t,e[i])
  end
  return res
end

function transform_basis(t::Transformation,x::AbstractArray{CT},e::AbstractArray{BasisVectors{T}},eq::MagneticEquilibrium) where CT <: MagneticCoordinates where T
  @assert ndims(x) == ndims(e) && size(x) == size(e) "Incompatible coordinate/basis vector arrays!"
  res = similar(e)
  Threads.@threads for i = 1:length(x)
    res[i] = transform_basis(t,x[i],e[i],eq)
  end
  return res
end

function covariant_basis
end

function contravariant_basis
end

function covariant_basis(t::Transformation,x::AbstractArray{T},eq::MagneticEquilibrium) where T <: MagneticCoordinates
  res = Array{BasisVectors{typeof(getfield(first(x),1))},ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    res[i] = covariant_basis(t,x[i],eq)
  end
  return res
end

function contravariant_basis(t::Transformation,x::AbstractArray{T},eq::MagneticEquilibrium) where T <: MagneticCoordinates
  res = Array{BasisVectors{typeof(getfield(first(x),1))},ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    res[i] = contravariant_basis(t,x[i],eq)
  end
  return res
end

function (t::Transformation)(x::AbstractArray{T},eq::MagneticEquilibrium) where T <: MagneticCoordinates
  y = Array{typeof(t(first(x),eq)),ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    y[i] = t(x[i],eq)
  end
  return y
end
