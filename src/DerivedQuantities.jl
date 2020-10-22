function gradB
end

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
