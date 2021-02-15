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
    transform_basis(::ContravariantFromCovariant,e::BasisVectors{T},J::T)

Transfor the covariant basis given by `e` to a contravariant basis using the dual relations and the provided jacobian `J`.
"""
function transform_basis(::ContravariantFromCovariant,e::BasisVectors{T},J::T) where T
  grad_1 = cross(e[:,2],e[:,3])/J
  grad_2 = cross(e[:,3],e[:,1])/J
  grad_3 = cross(e[:,1],e[:,2])/J
  return hcat(grad_1,grad_2,grad_3)
end
"""
    transform_basis(::CovariantFromContravariant,e::BasisVectors{T})

Transform the contravariant basis given by `e` to a covariant basis using the dual relations and the provided jacobian `J`.
"""
function transform_basis(::CovariantFromContravariant,e::BasisVectors{T},J::T) where T
  e_1 = cross(e[:,2],e[:,3])*J
  e_2 = cross(e[:,3],e[:,1])*J
  e_3 = cross(e[:,1],e[:,2])*J
  return hcat(e_1,e_2,e_3)
end

"""
    transform_basis(::ContravariantFromCovariant,e::BasisVectors{T})

Transform the covariant basis given by `e` to a contravariant basis by computing the transformation Jacobian
from the basis vectors by ``J = e₁ ⋅ e₂ × e₃``.
"""
function transform_basis(::ContravariantFromCovariant,e::BasisVectors{T}) where T
  J = jacobian(Covariant(),e)
  return transform_basis(ContravariantFromCovariant(),e,J)
end

"""
    transform_basis(::CovariantFromContravariant,e::BasisVectors{T})

Transform the contravariant basis given by `e` to a covariant basis by computing the transformation Jacobian
from the basis vectors by ``J = 1.0/(e¹ ⋅ e² × e³)``.
"""
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

"""
    transform_basis(t::Transformation,x::AbstracyArray{CT},e::AbstractArray{BasisVectors{T}},eq::MagneticEquilibrium) where CT <: MagneticCoordinates

Perform a change of basis for magnetic coordinates denoted by the transformation `t`, using the provided coordinate charts and basis vectors.
"""
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

"""
    covariant_basis(t::Transformation,x::AbstractArray{T},eq::MagneticEquilibrium) where T <: MagneticCoordinates

Generate a covariant basis for the magnetic coordinates defined by the transformation `t` at the points given by `x` using data from the magnetic equilibrium `eq`.  The routine providing the point transformation designated by `t` is defined in the respective equilibrium module.

# Examples
Generate covariant basis vectors for points along a field line defined by PestCoordinates using a VMEC equilibrium
```julia-repl
julia> using PlasmaEquilibriumToolkit, VMEC, NetCDF
julia> wout = NetCDF.open("wout.nc"); vmec, vmec_data = readVmecWout(wout);
julia> s = 0.5; vmec_s = VmecSurface(s,vmec);
julia> x = PestCoordinates(s*vmec.phi(1.0)/2π*vmec.signgs,0.0,-π:2π/10:π);
julia> v = VmecFromPest()(x,vmec_s);
julia> e_v = covariant_basis(CartesianFromVmec(),v,vmec_s)
10-element Array{StaticArrays.SArray{Tuple{3,3},Float64,2,9},1}:
 ⋮
```
"""
function covariant_basis(t::Transformation,x::AbstractArray{T},eq::MagneticEquilibrium) where T <: MagneticCoordinates
  res = Array{BasisVectors{typeof(getfield(first(x),1))},ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    res[i] = covariant_basis(t,x[i],eq)
  end
  return res
end

"""
    contravariant_basis(t::Transformation,x::AbstractArray{T},eq::MagneticEquilibrium) where T <: MagneticCoordinates

Generate a contravariant basis for the magnetic coordinates defined by the transformation `t` at the points given by `x` using data from the magnetic equilibrium `eq`.  The routine providing the point transformation designated by `t` is defined in the respective equilibrium module.

# Examples
Generate contravariant basis vectors for points along a field line defined by PestCoordinates using a VMEC equilibrium
```julia-repl
julia> using PlasmaEquilibriumToolkit, VMEC, NetCDF
julia> wout = NetCDF.open("wout.nc"); vmec, vmec_data = readVmecWout(wout);
julia> s = 0.5; vmec_s = VmecSurface(s,vmec);
julia> x = PestCoordinates(s*vmec.phi(1.0)/2π*vmec.signgs,0.0,-π:2π/10:π);
julia> v = VmecFromPest()(x,vmec_s);
julia> e_v = contravariant_basis(CartesianFromVmec(),v,vmec_s)
10-element Array{StaticArrays.SArray{Tuple{3,3},Float64,2,9},1}:
 ⋮
```
"""
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
