function gradB
end

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
function normalCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector,gradY::CoordinateVector)
  # κₙ = (B × ∇B) ⋅ ((∇ψ⋅∇ψ)∇α - (∇ψ⋅∇α)∇ψ)/(B³|∇ψ|)
  return dot(cross(B,gradB),gradY*dot(gradX,gradX) .- gradX*dot(gradX,gradY))/(norm(B)^3*norm(gradX))
end

function geodesicCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector)
  # κ_g = -(B × ∇B) ⋅ ∇ψ/(B²|∇ψ|)
  return -dot(cross(B,gradB),gradX)/(norm(B)^2*norm(gradX))
end

function curvatureComponents(contravariantBasis::BasisVectors,gradB::CoordinateVector)
  B = cross(contravariantBasis[:,1],contravariantBasis[:,2])
  return normalCurvature(B,gradB,contravariantBasis[:,1],contravariantBasis[:,2]), geodesicCurvature(B,gradB,contravariantBasis[:,1])
end

function metric(e::BasisVectors)
  return dot(e[:,1],e[:,1]), dot(e[:,1],e[:,2]), dot(e[:,2],e[:,2]), dot(e[:,1],e[:,3]), dot(e[:,2],e[:,3]), dot(e[:,3],e[:,3])
end
