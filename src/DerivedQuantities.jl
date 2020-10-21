function normalCurvature(B::CoordinateVector,gradB::CoordinateVector,gradY::CoordinateVector)
  return dot(cross(B,gradB),gradY)/norm(B)^2
end

function geodesicCurvature(B::CoordinateVector,gradB::CoordinateVector,gradX::CoordinateVector)
  return -dot(cross(B,gradB),gradX)/(norm(B)^2*norm(gradX))
end

function curvatureComponents(contravariantBasis::BasisVectors,gradB::CoordinateVector)
  B = cross(contravariantBasis[:,1],contravariantBasis[:,2])
  return normalCurvature(B,gradB,contravariantBasis[:,2]), geodesicCurvature(B,gradB,contravariantBasis[:,1])
end
