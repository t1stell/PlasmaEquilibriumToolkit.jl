import Base.abs
using LinearAlgebra

abstract type BasisTransformation <: Transformation end

#Define the singleton types for the coordinate transformations.  All transformations are
#subtypes of the abstract supertype Transformation provided by CoordinateTransformations.jl.
struct FluxFromPest <: Transformation end
struct PestFromFlux <: Transformation end
struct CylindricalFromFlux <: Transformation end
struct CylindricalFromPest <: Transformation end
struct CartesianFromFlux <: Transformation end
struct CartesianFromPest <: Transformation end
struct ContravariantFromCovariant <: BasisTransformation end
struct CovariantFromContravariant <: BasisTransformation end


"""
    abs(e::BasisVectors{T},component=0)

Compute the L2 norm of the basis vector `e`.  If `component` > 0 and <= 3, the L2 norm of the component is returned.

#  Examples
```
julia> using StaticArrays

julia> e = @SArray([1 4 7;2 5 8;3 6 9]);

julia> abs(e)
16.881943016134134

julia> abs(e,1)
3.7416573867739413

```
"""
function abs(e::BasisVectors{T}, component::Int = 0) where {T}
  if component > 0 && component <= 3
    return norm(e[:, component])
  else
    return norm(e)
  end
end


"""
    jacobian(t::BasisType,e::BasisVectors{T})

Compute the Jacobian of the covariant/contravariatn basis vectors `e` given by
``J = e₁ ⋅ e₂ × e₃`` (covariant) or ``J = 1/(e¹ ⋅ e² × e³)`` (contravariant).
"""
function jacobian(::Covariant,
                  e::BasisVectors;
                 )
return dot(e[:,1],cross(e[:,2],e[:,3]))
end

function jacobian(::Contravariant,
                  e::BasisVectors;
                 )
return 1.0 /dot(e[:,1],cross(e[:,2],e[:,3]))
end

"""
    transform_basis(t::BasisTransformation, e::BasisVectors)
    transform_basis(t::BasisTransformation, e::BasisVectors, jacobian)
    transform_basis(t::BasisTransformation, e::AbstractArray{BasisVectors})
    transform_basis(t::BasisTransformation, e::AbstractArray{BasisVectors}, jacobian::AbstractArray)

Transform the covariant/contravariant basis given by `e` to a
contravariant/covariant basis by computing the transformation Jacobian
from the basis vectors; for `ContravariantFromCovariant()` the Jacobian is given
by ``J = e₁ ⋅ e₂ × e₃`` and for `CovariantFromContravariant()` the Jacobian is
given by ``J = 1.0/(e¹ ⋅ e² × e³)``.
"""
function transform_basis(::ContravariantFromCovariant,
                         e::BasisVectors{T},
                         J::T;
                        ) where {T}
  e_1 = cross(e[:, 2], e[:, 3]) / J
  e_2 = cross(e[:, 3], e[:, 1]) / J
  e_3 = cross(e[:, 1], e[:, 2]) / J
  hcat(e_1, e_2, e_3)
end
function transform_basis(::CovariantFromContravariant,
                         e::BasisVectors{T},
                         J::T;
                        ) where {T}
  e_1 = cross(e[:,2],e[:,3]) * J
  e_2 = cross(e[:,3],e[:,1]) * J
  e_3 = cross(e[:,1],e[:,2]) * J
  return hcat(e_1,e_2,e_3)
end

function transform_basis(::ContravariantFromCovariant,
                         e::BasisVectors;
                        )
  J = jacobian(Covariant(),e)
  return transform_basis(ContravariantFromCovariant(),e,J)
end

function transform_basis(::CovariantFromContravariant,
                         e::BasisVectors;
                        )
  J = jacobian(Contravariant(),e)
  return transform_basis(CovariantFromContravariant(),e,J)
end

function transform_basis(t::BasisTransformation,
                         e::AbstractArray{BasisVectors{T}},
                         J::AbstractArray{T};
                        ) where {T}
  size(e) == size(J) || throw(DimensionMismatch("Incompatible dimensions for basis vectors and Jacobian array"))
  res = similar(e)
  @batch minbatch = 16 for i in eachindex(e, J, res)
    res[i] = transform_basis(t, e[i], J[i])
  end
  return res
end

function transform_basis(t::BasisTransformation,
                         e::AbstractArray{BasisVectors};
                        )
  res = similar(e)
  @batch minbatch = 16 for i in eachindex(e, res)
    res[i] = transform_basis(t, e[i])
  end
  return res
end

"""
    transform_basis(t::Transformation,x::AbstracyArray{CT},e::AbstractArray{BasisVectors{T}},eq::AbstractMagneticEquilibrium) where CT <: AbstractMagneticCoordinates

Perform a change of basis for magnetic coordinates denoted by the transformation `t`, using the provided coordinate charts and basis vectors.
"""
function transform_basis(t::Transformation,
                         x::AbstractArray,
                         e::AbstractArray{BasisVectors},
                         eq::E,
                        ) where {E <: AbstractMagneticEquilibrium}
  ndims(x) == ndims(e) && size(x) == size(e) ||
    throw(DimensionMismatch("Incompatible coordinate/basis vector arrays!"))
  res = similar(e)
  @batch minbatch = 16 for i ∈ eachindex(x)
    res[i] = transform_basis(t, x[i], e[i], eq)
  end
  return res
end

function basis_vectors(B::BasisType,
                       T::Transformation,
                       c::C,
                       eq::E,
                      ) where {C <: AbstractMagneticCoordinates,
                               E <: AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "Basis vector construction for $(B) basis with mapping $(T) over $(typeof(c)) does not exist",
    ),
  )
end

"""
    basis_vectors(b::BasisType,t::Transformation,x::AbstractArray{T},eq::AbstractMagneticEquilibrium) where T <: AbstractMagneticCoordinates

Generate basis vectors of type `b` for the magnetic coordinates defined by the transformation `t` at the points given by `x` using data from the magnetic equilibrium `eq`.  The routine providing the point transformation designated by `t` is defined in the respective equilibrium module.

# Examples
Generate covariant basis vectors for points along a field line defined by PestCoordinates using a VMEC equilibrium
```julia-repl
julia> using PlasmaEquilibriumToolkit, VMEC, NetCDF
julia> wout = NetCDF.open("wout.nc"); vmec, vmec_data = readVmecWout(wout);
julia> s = 0.5; vmec_s = VmecSurface(s,vmec);
julia> x = PestCoordinates(s*vmec.phi(1.0)/2π*vmec.signgs,0.0,-π:2π/10:π);
julia> v = VmecFromPest()(x,vmec_s);
julia> e_v = basis_vectors(Covariant(),CartesianFromVmec(),v,vmec_s)
10-element Array{StaticArrays.SArray{Tuple{3,3},Float64,2,9},1}:
 ⋮
```
"""
function basis_vectors(B::BasisType,
                       T::Transformation,
                       c::AbstractArray,
                       eq::E,
                      ) where {E <: AbstractMagneticEquilibrium}
  res =
    Array{BasisVectors{typeof(getfield(first(c), 1))},ndims(c)}(undef, size(c))
  @batch minbatch = 16 for i ∈ eachindex(c)
    res[i] = basis_vectors(B, T, c[i], eq)
  end
  return res
end

# `t` needs to be a singleton type
function (t::Transformation)(x::AbstractArray{T},
                             eq::E,
                            ) where {T <: AbstractMagneticCoordinates, 
                                     E<:AbstractMagneticEquilibrium}
  res = Array{typeof(t(first(x), eq)),ndims(x)}(undef, size(x))
  @batch minbatch = 16 for i ∈ eachindex(x)
    res[i] = t(x[i], eq)
  end
  return res
end
