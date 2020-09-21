function curvatureComponents(fieldline::Fieldline{T}) where T <: Real
  gradX = getfield(fieldline,:contraBasis_x)
  gradXMag = abs(gradX)
  gradXNorm = gradX / gradXMag 

  B = getfield(fieldline,:B)
  Bmag = abs(B)
  Bnorm = B / Bmag

  BCrossGradX = cross(B,gradX,getfield(fieldline,:jacobian)) 
  BCrossGradXMag = abs(BCrossGradX)
  BCrossGradXNorm = BCrossGradX / BCrossGradNorm

  curvature = getfield(fieldline,:curvature)
  basisX = getfield(fieldline,:coBasis_x)
  basisY = getfield(fieldline,:coBasis_y)
  basisZ = getfield(fieldline,:coBasis_z)
  metric_xyz = IdentityMetric(getfield(fieldline,:length))
  normalCurvature = dot(curvature,gradXNorm,metric_xyz)
  geodesicCurvature = dot(curvature,BCrossGradXNorm,metric_xyz)
  
  # κ = κ₁∇u¹ + κ₂∇u²
  
  return normalCurvature, geodesicCurvature

end

