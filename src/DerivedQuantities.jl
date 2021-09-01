
function Bnorm(::Contravariant, e::BasisVectors)
  return norm(cross(e[:, 1], e[:, 2]), 2)
end

function Bnorm(::Covariant, e::BasisVectors, ι::Real)
  return norm(1.0 / jacobian(Covariant(), e) * (e[:, 3] + ι * e[:, 2]), 2)
end

function Bnorm(e::BasisVectors)
  return Bnorm(Contravariant(), e)
end

function Bfield(::Contravariant, e::BasisVectors)
  return cross(e[:, 1], e[:, 2])
end

function Bfield(::Covariant, e::BasisVectors, ι::Real)
  return 1.0 / jacobian(Covariant(), e) * (e[:, 3] + ι * e[:, 2])
end

function Bfield(e::BasisVectors)
  return Bfield(Contravariant(), e)
end

function Bnorm(::Contravariant,
               e::AbstractArray{BasisVectors{T}};
              ) where {T}
  res = Array{T}(undef,size(e))
  Bnorm!(res,Contravariant(),e)
  return res
end

function Bnorm!(B::AbstractArray{T},
                ::Contravariant,
                e::AbstractArray{BasisVectors{T}};
               ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in Bnorm!"))
  @batch minbatch=10 for i in eachindex(B,e)
    B[i] = Bnorm(Contravariant(),e[i])
  end
end

function Bnorm(::Covariant,
               e::AbstractArray{BasisVectors{T}},
               ι::Real;
              ) where {T}
  res = Array{T}(undef,size(e))
  Bnorm!(res,Covariant(),e,ι)
  return res
end

function Bnorm!(B::AbstractArray{T},
                ::Contravariant,
                e::AbstractArray{BasisVectors{T}},
                ι::Real;
               ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in Bnorm!"))
  @batch minbatch=16 for i in eachindex(B,e)
    B[i] = Bnorm(Covariant(),e[i],ι)
  end
end

function Bfield(::Contravariant,
               e::AbstractArray{BasisVectors{T}};
              ) where {T}
  res = Array{CoordinateVector{T}}(undef,size(e))
  Bfield!(res,Contravariant(),e)
  return res
end

function Bfield!(B::AbstractArray{CoordinateVector{T}},
                ::Contravariant,
                e::AbstractArray{BasisVectors{T}};
               ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in Bfield!"))
  @batch minbatch=16 for i in eachindex(B,e)
    B[i] = Bfield(Contravariant(),e[i])
  end
end

function Bfield(::Covariant,
               e::AbstractArray{BasisVectors{T}},
               ι::Real;
              ) where {T}
  res = Array{CoordinateVector{T}}(undef,size(e))
  Bfield!(res,Covariant(),e,ι)
  return res
end

function Bfield!(B::AbstractArray{CoordinateVector{T}},
                ::Contravariant,
                e::AbstractArray{BasisVectors{T}},
                ι::Real;
               ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in Bfield!"))
  @batch minbatch=16 for i in eachindex(B,e)
    B[i] = Bfield(Covariant(),e[i],ι)
  end
end

function Bnorm(x::MC,
               eq::ET;
              ) where {MC<:AbstractMagneticCoordinates,
                       ET<:AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "Bnorm with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

function Bnorm(x::AbstractArray{MC},
               eq::ET;
              ) where {MC <: AbstractMagneticCoordinates,
                       ET <: AbstractMagneticEquilibrium,
                      }
  T = typeof(getfield(first(x),1))
  res = Array{T}(undef,size(x))
  Bnorm!(res,x,eq)
  return res
end

function Bnorm!(B::AbstractArray{T},
                x::AbstractArray{MC},
                eq::ET;
               ) where {MC <: AbstractMagneticCoordinates,
                        ET <: AbstractMagneticEquilibrium,
                        T}
  size(B) == size(x) || throw(DimensionMismatch("Incompatible dimensions in Bnorm!"))
  @batch minbatch=16 for i in eachindex(x,B)
    B[i] = Bnorm(x[i],eq)
  end
end

function Bfield(x::MC,
                eq::ET,
               ) where {MC<:AbstractMagneticCoordinates,
                        ET<:AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "Bfield with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

function Bfield(x::AbstractArray{MC},
                eq::ET;
               ) where {MC <: AbstractMagneticCoordinates,
                        ET <: AbstractMagneticEquilibrium,
                       }
  T = typeof(getfield(first(x),1))
  res = Array{CoordinateVector{T}}(undef,size(x))
  Bfield!(res,x,eq)
  return res
end

function Bfield!(Bvec::AbstractArray{CoordinateVector},
                 x::AbstractArray{MC},
                 eq::ET;
                ) where {MC <: AbstractMagneticCoordinates,
                         ET <: AbstractMagneticEquilibrium,
                         T}
  size(Bvec) == size(x) || throw(DimensionMismatch("Incompatible dimensions in Bfield!"))
  @batch minbatch=16 for i in eachindex(Bvec,x) 
    Bvec[i] = Bfield(x[i],eq) 
  end
end

function gradB(x::MC,
               e::BasisVectors,
               eq::ET,
              ) where {MC<:AbstractMagneticCoordinates,
                       ET<:AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "gradB with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

function jacobian(x::MC,
                  eq::ET,
                 ) where { MC<:AbstractMagneticCoordinates,
                          ET<:AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "jacobian with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

"""
    curvatureProjection(e::BasisVectors,gradB::CoordinateVector)

Computes the projection of B × ∇B/B² onto the perpendicular coordinate vectors given by
∇X = `e[:,1]` and ∇Y = `e[:,2]`.
"""
function curvatureProjection(e::BasisVectors, gradB::CoordinateVector)
  #K1 = (B × ∇B)/B² ⋅ ∇X
  #K2 = (B × ∇B)/B² ⋅ ∇Y
  B = cross(e[:, 1], e[:, 2])
  Bmag = norm(B)
  K1 = dot(cross(B, gradB), e[:, 1]) / Bmag^2
  K2 = dot(cross(B, gradB), e[:, 2]) / Bmag^2
  return K1, K2
end
#∇B = dbdx ∇x + dbdy ∇y + dbdz ∇z
#b = bx ∇x + by ∇y + bz ∇z
#b × ∇B = dbdy*bx ∇x × ∇y + dbdz*bx ∇x × ∇z + dbdx*by ∇y × ∇x + dbdz*by ∇y × ∇z + dbdx*bz ∇z × ∇x + dbdy*bz ∇z × ∇y
#b × ∇B ⋅ ∇x = dbdz*by ∇x ⋅ ∇y × ∇x - dbdy*bz ∇x ⋅ ∇y × ∇z = 1/√g*(dbdz*by - dbdy bz)
#
function curvatureProjection(e::AbstractArray{BasisVectors{T}},
                             gradB::AbstractArray{CoordinateVector{T}};
                            ) where {T}
  size(e) == size(gradB) || throw(DimensionMismatch("Incompatible dimensions in curvatureProjection"))
  res = Array{Tuple{T,T}}(undef,size(e))
  @batch minbatch=16 for i in eachindex(e,gradB,res)
    res[i] = curvatureProjection(e[i],gradB[i])
  end
  return res
end

# ∇P = dP/dX ∇X for B = ∇X × ∇Y
function curvatureProjection(e::BasisVectors,gradB::CoordinateVector,gradP::T) where T
  K1, K2 = curvatureProjection(e,gradB)
  return K1, K2 + 4π*1e-7*norm(cross(e[:,1],e[:,2]))*gradP
end

"""
    normalCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector,gradY::CoordinateVector)

Computes the normal curvature component defined in straight fieldline coordinates (X,Y,Z) with basis vectors defined
by (∇X,∇Y,∇Z) with the relation κₙ = (B × ∇B) ⋅ ((∇X⋅∇X)∇Y - (∇X⋅∇Y)∇X)/(B³|∇X|)
"""
function normalCurvature(
  B::CoordinateVector,
  gradB::CoordinateVector,
  gradX::CoordinateVector,
  gradY::CoordinateVector,
)
  # κₙ = (B × ∇B) ⋅ ((∇ψ⋅∇ψ)∇α - (∇ψ⋅∇α)∇ψ)/(B³|∇ψ|)
  return dot(
    cross(B, gradB),
    gradY * dot(gradX, gradX) .- gradX * dot(gradX, gradY),
  ) / (norm(B)^3 * norm(gradX))
end

"""
    geodesicCurvature(b::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector)

Computes the geodesic curvature component defined in straight fieldline coordinates (X,Y,Z) with basis vectors defined
by (∇X,∇Y,∇Z) with the relation κ_g = -(B × ∇B) ⋅ ∇X/(B²|∇X|)
"""
function geodesicCurvature(
  B::CoordinateVector,
  gradB::CoordinateVector,
  gradX::CoordinateVector,
)
  # κ_g = -(B × ∇B) ⋅ ∇ψ/(B²|∇ψ|)
  return -dot(cross(B, gradB), gradX) / (norm(B)^2 * norm(gradX))
end

"""
    curvatureComponents(e::BasisVectors,gradB::CoordinateVector)

Computes the normal and geodesic curvature vectors for a stright field line coordinate system with basis vectors
∇X = `e[:,1]` and ∇Y = `e[:,2]`.

See also: [`normalCurvature`](@ref), [`geodesicCurvature`](@ref)
"""
function curvatureComponents(contravariantBasis::BasisVectors{T},
                             gradB::CoordinateVector{T};
                            ) where {T}
  B = cross(contravariantBasis[:, 1], contravariantBasis[:, 2])
  return normalCurvature(
    B,
    gradB,
    contravariantBasis[:, 1],
    contravariantBasis[:, 2],
  ),
  geodesicCurvature(B, gradB, contravariantBasis[:, 1])
end

function curvatureComponents(contravariantBasis::AbstractArray{BasisVectors{T}},
                             ∇B::AbstractArray{CoordinateVector{T}};
                            ) where {T}
  size(contravariantBasis) == size(∇B) || throw(
    DimensionMismatch("Basis vectors and ∇B arays must have the same size"),
  )
  res = Array{NTuple{2,T}}(undef, size(∇B))
  @batch minbatch=16 for i in eachindex(contravariantBasis, ∇B, res)
    res[i] = curvatureComponents(contravariantBasis[i], ∇B[i])
  end
  return res
end

"""
    metric2(e::BasisVectors)

Computes the metric tensor components `gᵘᵛ` or `gᵤᵥ` for a set of basis vectors `e` in a 6 element SVector: [g11 = `e[:,1]*e[:,1]`, g112 = `e[:,1]*e[:,2]`,…]
"""
function metric(e::BasisVectors, i::Integer, j::Integer)
  return dot(e[:, i], e[:, j])
end

function metric(e::BasisVectors)
  return SVector(
    dot(e[:, 1], e[:, 1]),
    dot(e[:, 1], e[:, 2]),
    dot(e[:, 2], e[:, 2]),
    dot(e[:, 1], e[:, 3]),
    dot(e[:, 2], e[:, 3]),
    dot(e[:, 3], e[:, 3]),
  )
end

function metric(e::AbstractArray{BasisVectors{T}}) where {T}
  res = Array{SVector{6,T}}(undef, size(e))
  @batch minbatch=16 for i in eachindex(e,res)
    res[i] = metric(e[i])
  end
  return res
end

function metric(
  e::AbstractArray{BasisVectors{T}},
  u::Integer,
  v::Integer,
) where {T}
  res = Array{T}(undef, size(e))
  @batch minbatch=16 for i in eachindex(res,e)
    res[i] = metric(e[i], u, v)
  end
  return res
end
