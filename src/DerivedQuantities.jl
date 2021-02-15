
function gradB
end

"""
    curvatureProjection(e::BasisVectors,gradB::CoordinateVector)

Computes the projection of B × ∇B/B² onto the perpendicular coordinate vectors given by
∇X = `e[:,1]` and ∇Y = `e[:,2]`.
"""
function curvatureProjection(e::BasisVectors,gradB::CoordinateVector)
  #K1 = (B × ∇B)/B² ⋅ ∇X
  #K2 = (B × ∇B)/B² ⋅ ∇Y
  B = cross(e[:,1],e[:,2])
  Bmag = norm(B)
  K1 = dot(cross(B,gradB),e[:,1])/Bmag^2
  K2 = dot(cross(B,gradB),e[:,2])/Bmag^2
  return K1, K2
end
#∇B = dbdx ∇x + dbdy ∇y + dbdz ∇z
#b = bx ∇x + by ∇y + bz ∇z
#b × ∇B = dbdy*bx ∇x × ∇y + dbdz*bx ∇x × ∇z + dbdx*by ∇y × ∇x + dbdz*by ∇y × ∇z + dbdx*bz ∇z × ∇x + dbdy*bz ∇z × ∇y
#b × ∇B ⋅ ∇x = dbdz*by ∇x ⋅ ∇y × ∇x - dbdy*bz ∇x ⋅ ∇y × ∇z = 1/√g*(dbdz*by - dbdy bz)

"""
    normalCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector,gradY::CoordinateVector)

Computes the normal curvature component defined in straight fieldline coordinates (X,Y,Z) with basis vectors defined
by (∇X,∇Y,∇Z) with the relation κₙ = (B × ∇B) ⋅ ((∇X⋅∇X)∇Y - (∇X⋅∇Y)∇X)/(B³|∇X|)
"""
function normalCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector,gradY::CoordinateVector)
  # κₙ = (B × ∇B) ⋅ ((∇ψ⋅∇ψ)∇α - (∇ψ⋅∇α)∇ψ)/(B³|∇ψ|)
  return dot(cross(B,gradB),gradY*dot(gradX,gradX) .- gradX*dot(gradX,gradY))/(norm(B)^3*norm(gradX))
end

"""
    geodesicCurvature(b::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector)
    
Computes the geodesic curvature component defined in straight fieldline coordinates (X,Y,Z) with basis vectors defined
by (∇X,∇Y,∇Z) with the relation κ_g = -(B × ∇B) ⋅ ∇X/(B²|∇X|)
"""
function geodesicCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector)
  # κ_g = -(B × ∇B) ⋅ ∇ψ/(B²|∇ψ|)
  return -dot(cross(B,gradB),gradX)/(norm(B)^2*norm(gradX))
end

"""
    curvatureComponents(e::BasisVectors,gradB::CoordinateVector)

Computes the normal and geodesic curvature vectors for a stright field line coordinate system with basis vectors
∇X = `e[:,1]` and ∇Y = `e[:,2]`.

See also: [`normalCurvature`](@ref), [`geodesicCurvature`](@ref)
"""
function curvatureComponents(contravariantBasis::BasisVectors,gradB::CoordinateVector)
  B = cross(contravariantBasis[:,1],contravariantBasis[:,2])
  return normalCurvature(B,gradB,contravariantBasis[:,1],contravariantBasis[:,2]), geodesicCurvature(B,gradB,contravariantBasis[:,1])
end

"""
    metric(e::BasisVectors)

Computes the metric tensor components `gᵘᵛ` or `gᵤᵥ` for a set of basis vectors `e` in 6 element tuple: (g11 = `e[:,1]*e[:,1]`, g112 = `e[:,1]*e[:,2]`,…)
"""
function metric(e::BasisVectors)
  return dot(e[:,1],e[:,1]), dot(e[:,1],e[:,2]), dot(e[:,2],e[:,2]), dot(e[:,1],e[:,3]), dot(e[:,2],e[:,3]), dot(e[:,3],e[:,3])
end
