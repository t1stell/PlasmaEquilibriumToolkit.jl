
function B_norm(::Contravariant, 
                e::BasisVectors;
               )
  return norm(cross(e[:, 1], e[:, 2]), 2)
end

function B_norm(::Covariant,
                e::BasisVectors;
               )
  throw(ArgumentError("Rotational transform required to compute B_norm with covariant basis vectors!"))
end

function B_norm(::Covariant, 
                e::BasisVectors,
                ι::Real;
               )
  return norm(1.0 / jacobian(Covariant(), e) * (e[:, 3] + ι * e[:, 2]), 2)
end

function B_norm(e::BasisVectors)
  return B_norm(Contravariant(), e)
end


function B_norm(::Contravariant,
                e::AbstractArray{BasisVectors{T}};
               ) where {T}
  res = Array{T}(undef, size(e))
  B_norm!(res, Contravariant(), e)
  return res
end

function B_norm!(B::AbstractArray{T},
                 ::Contravariant,
                 e::AbstractArray{BasisVectors{T}},
                ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in B_norm!"))
  @batch minbatch = 10 for i in eachindex(B, e)
    B[i] = B_norm(Contravariant(), e[i])
  end
end

function B_norm(::Covariant,
                e::AbstractArray{BasisVectors{T}},
                ι::Real,
               ) where {T}
  res = Array{T}(undef, size(e))
  B_norm!(res, Covariant(), e, ι)
  return res
end

function B_norm!(B::AbstractArray{T},
                 ::Contravariant,
                 e::AbstractArray{BasisVectors{T}},
                 ι::Real,
                ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in B_norm!"))
  @batch minbatch = 16 for i in eachindex(B, e)
    B[i] = B_norm(Covariant(), e[i], ι)
  end
end

function B_field(::Contravariant,
  e::BasisVectors;
 )
return cross(e[:, 1], e[:, 2])
end

function B_field(::Covariant,
  e::BasisVectors, 
  ι::Real;
 )
return 1.0 / jacobian(Covariant(), e) * (e[:, 3] + ι * e[:, 2])
end

function B_field(e::BasisVectors)
return B_field(Contravariant(), e)
end

function B_field(::Contravariant,
                 e::AbstractArray{BasisVectors{T}};
                ) where {T}
  res = Array{CoordinateVector{T}}(undef, size(e))
  B_field!(res, Contravariant(), e)
  return res
end

function B_field!(B::AbstractArray{CoordinateVector{T}},
                  ::Contravariant,
                  e::AbstractArray{BasisVectors{T}},
                 ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in B_field!"))
  @batch minbatch = 16 for i in eachindex(B, e)
    B[i] = B_field(Contravariant(), e[i])
  end
end

function B_field(::Covariant,
                 e::AbstractArray{BasisVectors{T}},
                 ι::Real,
                ) where {T}
  res = Array{CoordinateVector{T}}(undef, size(e))
  B_field!(res, Covariant(), e, ι)
  return res
end

function B_field!(B::AbstractArray{CoordinateVector{T}},
                  ::Contravariant,
                  e::AbstractArray{BasisVectors{T}},
                  ι::Real,
                 ) where {T}
  size(B) == size(e) || throw(DimensionMismatch("Incompatible sizes in B_field!"))
  @batch minbatch = 16 for i in eachindex(B, e)
    B[i] = B_field(Covariant(), e[i], ι)
  end
end

function B_norm(x::C,
                eq::E,
               ) where {C <: AbstractMagneticCoordinates,
                        E <: AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "B_norm with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

function B_norm(x::AbstractArray,
                eq::E,
               ) where {E <: AbstractMagneticEquilibrium}
  T = typeof(getfield(first(x), 1))
  res = Array{T}(undef, size(x))
  B_norm!(res, x, eq)
  return res
end

function B_norm!(B::AbstractArray,
                 x::AbstractArray,
                 eq::E,
                ) where {E <: AbstractMagneticEquilibrium}
  size(B) == size(x) || throw(DimensionMismatch("Incompatible dimensions in B_norm!"))
  @batch minbatch = 16 for i in eachindex(x, B)
    B[i] = B_norm(x[i], eq)
  end
end

function B_field(x::C,
                 eq::E,
                ) where {C <: AbstractMagneticCoordinates,
                         E <: AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "B_field with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

function B_field(x::AbstractArray,
                 eq::ET,
                ) where {E <: AbstractMagneticEquilibrium}
  T = typeof(getfield(first(x), 1))
  res = Array{CoordinateVector{T}}(undef, size(x))
  B_field!(res, x, eq)
  return res
end

function B_field!(Bvec::AbstractArray{CoordinateVector},
                  x::AbstractArray,
                  eq::E,
                 ) where {E <: AbstractMagneticEquilibrium}
  size(Bvec) == size(x) ||
    throw(DimensionMismatch("Incompatible dimensions in B_field!"))
  @batch minbatch = 16 for i in eachindex(Bvec, x)
    Bvec[i] = B_field(x[i], eq)
  end
end

function grad_B(x::C,
                e::BasisVectors,
                eq::E,
               ) where {C <: AbstractMagneticCoordinates,
                       E <: AbstractMagneticEquilibrium}
  throw(ArgumentError(
      "gradB with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ))
end

function jacobian(x::C,
                  eq::E,
                 ) where {C <: AbstractMagneticCoordinates,
                          E <: AbstractMagneticEquilibrium}
  throw(
    ArgumentError(
      "jacobian with $(nameof(typeof(x))) for $(nameof(typeof(eq))) not yet implemented",
    ),
  )
end

"""
    gradBProjection(e::BasisVectors,∇B::CoordinateVector)

Computes the projection of B × ∇B/B² onto the perpendicular coordinate vectors given by
∇X = `e[:,1]` and ∇Y = `e[:,2]`.
"""
function grad_B_projection(e::BasisVectors,
                           ∇B::CoordinateVector;
                        )
  #K1 = (B × ∇B)/B² ⋅ ∇X
  #K2 = (B × ∇B)/B² ⋅ ∇Y
  B = cross(e[:, 1], e[:, 2])
  Bmag = norm(B)
  K1 = dot(cross(B, ∇B), e[:, 1]) / Bmag^2
  K2 = dot(cross(B, ∇B), e[:, 2]) / Bmag^2
  return K1, K2
end
#∇B = dbdx ∇x + dbdy ∇y + dbdz ∇z
#b = bx ∇x + by ∇y + bz ∇z
#b × ∇B = dbdy*bx ∇x × ∇y + dbdz*bx ∇x × ∇z + dbdx*by ∇y × ∇x + dbdz*by ∇y × ∇z + dbdx*bz ∇z × ∇x + dbdy*bz ∇z × ∇y
#b × ∇B ⋅ ∇x = dbdz*by ∇x ⋅ ∇y × ∇x - dbdy*bz ∇x ⋅ ∇y × ∇z = 1/√g*(dbdz*by - dbdy bz)
#

function grad_B_projection(e::AbstractArray{BasisVectors{T}},
                           gradB::AbstractArray{CoordinateVector{T}};
                          ) where {T}
  size(e) == size(gradB) || throw(DimensionMismatch("Incompatible dimensions in curvatureProjection"))
  res = Array{Tuple{T,T}}(undef,size(e))
  @batch minbatch=16 for i in eachindex(e,gradB,res)
    res[i] = grad_B_projection(e[i],gradB[i])
  end
  return res
end

# ∇P = dP/dX ∇X for B = ∇X × ∇Y
function grad_B_projection(e::BasisVectors,
                           ∇B::CoordinateVector,
                           ∇P::T,
                          ) where {T}
  K1, K2 = grad_B_projection(e, ∇B)
  return K1, K2 + 4π * 1e-7 * norm(cross(e[:, 1], e[:, 2])) * ∇P
end

"""
    normalCurvature(B::CoordinateVector,∇B::CoordinateVector,∇X::CoordinateVector,∇Y::CoordinateVector)

Computes the normal curvature component defined in straight fieldline coordinates (X,Y,Z) with basis vectors defined
by (∇X,∇Y,∇Z) with the relation κₙ = (B × ∇B) ⋅ ((∇X⋅∇X)∇Y - (∇X⋅∇Y)∇X)/(B³|∇X|)
"""
function normal_curvature(B::CoordinateVector{T},
                          ∇B::CoordinateVector{T},
                          ∇X::CoordinateVector{T},
                          ∇Y::CoordinateVector{T};
                         ) where {T}
  # κₙ = (B × ∇B) ⋅ ((∇ψ⋅∇ψ)∇α - (∇ψ⋅∇α)∇ψ)/(B³|∇ψ|)
  return dot(cross(B, ∇B), ∇Y * dot(∇X, ∇X) .- ∇X * dot(∇X, ∇Y)) / (norm(B)^3 * norm(∇X))
end

"""
    geodesic_curvature(b::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector)

Computes the geodesic curvature component defined in straight fieldline coordinates (X,Y,Z) with basis vectors defined
by (∇X,∇Y,∇Z) with the relation κ_g = -(B × ∇B) ⋅ ∇X/(B²|∇X|)
"""
function geodesic_curvature(B::CoordinateVector{T},
                            ∇B::CoordinateVector{T},
                            ∇X::CoordinateVector{T};
                           ) where {T}
  # κ_g = -(B × ∇B) ⋅ ∇ψ/(B²|∇ψ|)
  return -dot(cross(B, ∇B), ∇X) / (norm(B)^2 * norm(∇X))
end

"""
    curvature_components(e::BasisVectors,gradB::CoordinateVector)

Computes the normal and geodesic curvature vectors for a stright field line coordinate system with basis vectors
∇X = `e[:,1]` and ∇Y = `e[:,2]`.

See also: [`normalCurvature`](@ref), [`geodesicCurvature`](@ref)
"""
function curvature_components(∇X::BasisVectors{T},
                              ∇B::CoordinateVector{T},
                             ) where {T}
  B = cross(contravariantBasis[:, 1], contravariantBasis[:, 2])
  return normalCurvature(B, ∇B, ∇X[:, 1], ∇X[:, 2]),
         geodesicCurvature(B, ∇B,∇X[:, 1])
end

function curvature_components(∇X::AbstractArray{BasisVectors{T}},
                              ∇B::AbstractArray{CoordinateVector{T}},
                             ) where {T}
  size(contravariantBasis) == size(∇B) || throw(
    DimensionMismatch("Basis vectors and ∇B arrays must have the same size"),
  )
  res = Array{NTuple{2,T}}(undef, size(∇B))
  @batch minbatch = 16 for i in eachindex(∇X, ∇B, res)
    res[i] = curvature_components(∇X[i], ∇B[i])
  end
  return res
end

"""
    metric2(e::BasisVectors)

Computes the metric tensor components `gᵘᵛ` or `gᵤᵥ` for a set of basis vectors `e` in a 6 element SVector: [g11 = `e[:,1]*e[:,1]`, g112 = `e[:,1]*e[:,2]`,…]
Computes the metric tensor components `gᵘᵛ` or `gᵤᵥ` for a set of basis vectors `e` in a 6 element SVector: [g11 = `e[:,1]*e[:,1]`, g12 = `e[:,1]*e[:,2]`,…]
"""
function metric(e::BasisVectors,
                i::Integer,
                j::Integer;
               )
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
  @batch minbatch = 16 for i in eachindex(e, res)
    res[i] = metric(e[i])
  end
  return res
end

function metric(e::AbstractArray{BasisVectors{T}},
                u::Integer,
                v::Integer,
               ) where {T}
  res = Array{T}(undef, size(e))
  @batch minbatch = 16 for i in eachindex(res, e)
    res[i] = metric(e[i], u, v)
  end
  return res
end
