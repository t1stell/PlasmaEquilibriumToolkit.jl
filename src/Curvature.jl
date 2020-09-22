function curvatureComponents(fieldline::Fieldline{T}) where T <: Real
  gradX = getfield(fieldline,:contraBasis_x)
  gradY = getfield(fieldline,:contraBasis_y)
  Bmag = abs(getfield(fieldline,:B))
  norm = abs(gradX).*abs(gradY)
  dBdx = Threads.@spawn mu0*getfield(fieldline,:dPdx)./(Bmag.*Bmag) .- 
         dot(getfield(fieldline,:B),cross(getfield(fieldline,:gradB),gradY)) ./ Bmag
  dBdy = Threads.@spawn -dot(getfield(fieldline,:B),cross(getfield(fieldline,:gradB),gradX)) ./ Bmag

  sinGradXGradY = Threads.@spawn abs(cross(gradX,gradY))./norm
  cosGradXGradY = Threads.@spawn abs.(dot(gradX,gradY))./norm
                  
  normalCurvature = fetch(dBdx) .* abs(gradX) .+ fetch(dBdy) .* abs(gradY) .* fetch(cosGradXGradY)
  geodesicCurvature = fetch(dBdy) .* abs(gradY) .* fetch(sinGradXGradY)
  
  return normalCurvature, geodesicCurvature

end

function curvature(fieldline::Fieldline{T}) where T <: Real
  #TODO Add pressure term using a spline
  metric = getfield(fieldline,:metric)
  Bmag = abs(getfield(fieldline,:B))
  dBdx = Threads.@spawn -dot(getfield(fieldline,:B),cross(getfield(fieldline,:gradB),getfield(fieldline,:contraBasis_y))) ./ Bmag
  dBdy = Threads.@spawn -dot(getfield(fieldline,:B),cross(getfield(fieldline,:gradB),getfield(fieldline,:contraBasis_x))) ./ Bmag

  # Define terms for computing magnetic curvature vector
  gamma1 = Threads.@spawn getfield(metric,:yy) .* getfield(metric,:xz) .- getfield(metric,:xy) .* getfield(metric,:yz)
  gamma2 = Threads.@spawn getfield(metric,:xx) .* getfield(metric,:yz) .- getfield(metric,:xy) .* getfield(metric,:xz)


  curvature = getfield(fieldline,:contraBasis_x)*
  (Bmag.*fetch(dBdx) .+  psiPrime * (fetch(gamma1) .* getfield(fieldline,:dBdz) ./ Bmag)) +
  getfield(fieldline,:contraBasis_y)*(Bmag.*fetch(tdBdy) .+ (fetch(gamma2) .* getfield(fieldline,:dBdz) ./ Bmag))

  return curvature
end
