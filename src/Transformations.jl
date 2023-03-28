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
struct CylindricalFromFourier <: Transformation end
struct InternalFromFlux <: Transformation end
struct InternalFromPest <: Transformation end
struct PestFromInternal <: Transformation end
struct FluxFromInternal <: Transformation end
struct CylindricalFromInternal <: Transformation end
struct InternalFromCylindrical <: Transformation end
struct CartesianFromInternal <: Transformation end
struct InternalFromCartesian <: Transformation end

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
                  e::BasisVectors{T};
                 ) where {T}
return dot(e[:,1],cross(e[:,2],e[:,3]))
end

function jacobian(::Contravariant,
                  e::BasisVectors{T};
                 ) where {T}
return 1.0 /dot(e[:,1],cross(e[:,2],e[:,3]))
end


function jacobian(x::C,
                  eq::E,
                 ) where {C <: AbstractMagneticCoordinates,
                          E <: AbstractGeometry}
  throw(ArgumentError("jacobian with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented"))
end

function jacobian(x::AbstractArray,
                  eq::E,
                 ) where {E <: AbstractGeometry}
  res = Array{typeof(getfield(first(x), 1)), ndims(x)}(undef, size(x))
  @batch minbatch=16 for i in eachindex(x, res)
    res[i] = jacobian(x[i], eq)
  end
  return res
end

###Coordinate transformations
function (::CylindricalFromInternal)(x::C, surf::S
                                    ) where {C <: AbstractMagneticCoordinates, 
                                             S <: AbstractMagneticSurface
                                            }
  R = surface_get(x, surf, :r)
  Z = surface_get(x, surf, :z)
  return Cylindrical(R, x.ζ, Z)
end

function (::CartesianFromInternal)(x::C, surf::S
                                  ) where {C <: AbstractMagneticCoordinates, 
                                           S <: AbstractMagneticSurface
                                          }
  cc = CylindricalFromInternal()(x, surf)
  return CartesianFromCylindrical()(cc)
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
                         e::BasisVectors{T};
                        ) where {T}
  J = jacobian(Covariant(),e)
  return transform_basis(ContravariantFromCovariant(),e,J)
end

function transform_basis(::CovariantFromContravariant,
                         e::BasisVectors{T};
                        ) where {T}
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
                         e::AbstractArray{BasisVectors{T}};
                        ) where {T}
  res = similar(e)
  @batch minbatch = 16 for i in eachindex(e, res)
    res[i] = transform_basis(t, e[i])
  end
  return res
end

"""

    transform_basis(t::Transformation,x::AbstracyArray{AbstractMagneticCoordinates},e::AbstractArray{BasisVectors{T}},eq::AbstractGeometry)

Perform a change of basis for magnetic coordinates denoted by the transformation `t`, using the provided coordinate charts and basis vectors.
"""
function transform_basis(t::Transformation,
                         x::C,
                         e::BasisVectors{T},
                         eq::E;
                        ) where {T, C <: AbstractMagneticCoordinates,
                                 E <: AbstractGeometry}
  throw(ArgumentError("Basis transformation for $(typeof(t)) for coordinates $(typeof(x)) and equilibrium $(typoeof(eq)) not yet impleented"))
end

function transform_basis(t::Transformation,
                         x::AbstractArray,
                         e::AbstractArray{BasisVectors{T}},
                         eq::E,
                        ) where {T, E <: AbstractGeometry}
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
                               E <: AbstractGeometry}
  throw(
    ArgumentError(
      "Basis vector construction for $(B) basis with mapping $(T) over $(typeof(c)) does not exist",
    ),
  )
end

"""
    basis_vectors(b::BasisType,t::Transformation,x::AbstractArray{T},eq::AbstractGeometry) where T <: AbstractMagneticCoordinates

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
                      ) where {E <: AbstractGeometry}
  res =
    Array{BasisVectors{typeof(getfield(first(c), 1))},ndims(c)}(undef, size(c))
  @batch minbatch = 16 for i ∈ eachindex(c)
    res[i] = basis_vectors(B, T, c[i], eq)
  end
  return res
end

# `t` needs to be a singleton type
function (t::Transformation)(x::AbstractArray,
                             eq::E,
                            ) where {E <: AbstractGeometry}
  res = Array{typeof(t(first(x), eq)), ndims(x)}(undef, size(x))
  @batch minbatch = 16 for i ∈ eachindex(x, res)
    res[i] = t(x[i], eq)
  end
  return isstructtype(eltype(res)) ? StructArray(res) : res
end


"""
    θ_internal(x::FluxCoordinates,λ::Symbol,interval=0.25)

Computes the internal θ coordinate associated with the points `x` specified
in `FluxCoordinates`. The surface must have a λ component to relate the angles
"""
function θ_internal(x::FluxCoordinates,
                  surf::S,
                  interval=0.25) where {S <: AbstractSurface}

  function residual(θ::T) where {T <: AbstractFloat}
    return θ - x.θ + surface_get(FluxCoordinates(x.ψ, θ, x.ζ), surf, :λ)
  end

  bracket = (x.θ-interval,x.θ+interval)
  try
    return Roots.find_zero(residual,bracket,Roots.Order2())
  catch err
    if is(err,Roots.ConvergenceFailed) && attempt <= 5
      return thetaVmec(x,λ,2*interval)
    end
  end
end

function θ_internal(x::AbstractArray{FluxCoordinates},
                   surf::S;
                  ) where {S <: AbstractSurface}
  T = typeof(x.θ)
  y = Array{T, ndims(x)}(undef,size(x))
  @batch for i in eachindex(x)
    y[i] = thetaVmec(x[i], surf)
  end
  return y
end

""" 
    flux_surface_and_angle(x::AbstractMagneticCoordinates, E::AbstractMagneticEquilibrium)

Dummy function to guess both the radial and θ value for an equilibrium. This needs to be defined
in the files for the given equilibrium, since it is difficult to abstract
"""
function flux_surface_and_angle(x::Cylindrical{T, T}, eq::E) where {T, E <: AbstractMagneticEquilibrium}
    error("flux_surface_and_angle not implemented for equilibrium type $(E)")
end


function CoordinateTransformations.transform_deriv(::CylindricalFromFlux,
                                                   x::FluxCoordinates,
                                                   surf::S;
                                    ) where {S <: AbstractMagneticSurface}
  ic = InternalFromFlux()(x,surf)
  dV = derivatives(ic,surf)
  dsdψ = 2π/surf.phi[end]*surf.signgs
  dΛdθv = dV[2,2]
  dΛdζv = dV[2,3]

  dRdψ = dV[1,1]*dsdψ
  dZdψ = dV[3,1]*dsdψ
  dϕdψ = zero(typeof(x.θ))

  dRdθ = dV[1,2]/(1+dΛdθv)
  dZdθ = dV[3,2]/(1+dΛdθv)
  dϕdθ = zero(typeof(x.θ))

  dRdζ = -dV[1,3] + dV[1,2]*dΛdζv/(1+dΛdθv)
  dZdζ = -dV[3,3] + dV[3,2]*dΛdζv/(1+dΛdθv)
  dϕdζ = one(typeof(x.θ))
  return @SMatrix [dRdψ dRdθ dRdζ;
                   dϕdψ dϕdθ dϕdζ;
                   dZdψ dZdθ dZdζ]
end

function CoordinateTransformations.transform_deriv(::CylindricalFromFourier,
                                                   x::C,
                                                   surf::S;
         ) where {C <: AbstractMagneticCoordinates, S <: AbstractSurface}
  dRds = surface_get(x, surf, :r; deriv=:ds)
  dZds = surface_get(x, surf, :z; deriv=:ds)
  dϕds = zero(typeof(x.θ))

  dRdθ = surface_get(x, surf, :r; deriv=:dθ)
  dZdθ = surface_get(x, surf, :z; deriv=:dθ)
  dϕdθ = zero(typeof(x.θ))

  dRdζ = surface_get(x, surf, :r; deriv=:dζ)
  dZdζ = surface_get(x, surf, :z; deriv=:dζ)
  dϕdζ = one(typeof(x.θ))
  return @SMatrix [dRds dRdθ dRdζ;
                   dϕds dϕdθ dϕdζ;
                   dZds dZdθ dZdζ]
end

function CoordinateTransformations.transform_deriv(::CylindricalFromInternal,
                                                   x::C,
                                                   surf::S;
                                  ) where {C <: AbstractMagneticCoordinates, 
                                           S <: AbstractMagneticSurface}
  dRds = surface_get(x, surf, :r, deriv=:ds)
  dZds = surface_get(x, surf, :z, deriv=:ds)
  dϕds = zero(typeof(x.θ))

  dRdθ = surface_get(x, surf, :r, deriv=:dθ)
  dZdθ = surface_get(x, surf, :z, deriv=:dθ)
  dϕdθ = zero(typeof(x.θ))

  dRdζ = surface_get(x, surf, :r, deriv=:dζ)
  dZdζ = surface_get(x, surf, :z, deriv=:dζ)
  dϕdζ = one(typeof(x.θ))
  return @SMatrix [dRds dRdθ dRdζ;
                   dϕds dϕdθ dϕdζ;
                   dZds dZdθ dZdζ]
end

function derivatives(x::C, surf::S
        ) where {C <: AbstractMagneticCoordinates, S <: AbstractMagneticSurface}
  dRds = surface_get(x, surf, :r, deriv=:ds)
  dZds = surface_get(x, surf, :z, deriv=:ds)
  dλds = surface_get(x, surf, :λ, deriv=:ds)

  dRdθ = surface_get(x, surf, :r, deriv=:dθ)
  dZdθ = surface_get(x, surf, :z, deriv=:dθ)
  dλdθ = surface_get(x, surf, :λ, deriv=:dθ)

  dRdζ = surface_get(x, surf, :r, deriv=:dζ)
  dZdζ = surface_get(x, surf, :z, deriv=:dζ)
  dλdζ = surface_get(x, surf, :λ, deriv=:dζ)
  return @SMatrix [dRds dRdθ dRdζ;
                   dλds dλdθ dλdζ;
                   dZds dZdθ dZdζ]
end

function derivatives(x::C, surf::S, field::Symbol
        ) where {C <: AbstractMagneticCoordinates, S <: AbstractMagneticSurface}
  
  ds = surface_get(x, surf, field, deriv=:ds)
  dθ = surface_get(x, surf, field, deriv=:dθ)
  dζ = surface_get(x, surf, field, deriv=:dζ)
  return @SVector [ds, dθ, dζ]
end

