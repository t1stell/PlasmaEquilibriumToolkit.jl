
"""
    gradB(vmecVectors::VmecCoordinates{D,T,N},vmec::VMEC.VmecData) where {D,T,N}

Computes the Cartesian components of the ∇B vector from a VMEC equilibrium at each point in `vmecVectors`

# Returns
- `gradB::VectorField{D,T,N}`
"""
function gradB(vmecVectors::VmecCoordinates{D,T,N},vmec::VMEC.VmecData) where {D,T,N}
  fluxFactor = typeof(vmecVectors) <: PestCoordinates ? 2*π*vmec.signgs/vmec.phi[vmec.ns] : 1.0
  s = map(i->i[1],getfield(vmecVectors,:data))[1]
  alpha = component(getfield(vmecVectors,:data),2)
  zeta = component(getfield(vmecVectors,:data),3)
  theta = VMEC.findThetaVmec(s,alpha.+VMEC.iota(s,vmec).*zeta,zeta,vmec)
  dBds = Threads.@spawn VMEC.inverseTransform(s,theta,zeta,vmec,:bmnc,:bmns;ds=true)
  dBdtv = Threads.@spawn VMEC.inverseTransform(s,theta,zeta,vmec,:bmnc,:bmns;dpoloidal=true)
  dBdz = Threads.@spawn VMEC.inverseTransform(s,theta,zeta,vmec,:bmnc,:bmns;dtoroidal=true)
  return gradX1(vmecVectors) * fetch(dBds) + gradX2(vmecVectors) * fetch(dBdtv) + gradX3(vmecVectors) * fetch(dBdz)
end

"""
    curvatureComponents(basisVectors::AbstractCoordinateField{D,T,N},vmecVectors::VmecCoordinates{D,T,N},
                        vmec::VMEC.VmecData) where {D,T,N}

Computes the normal and geodesic curvature components with respect to the input coordinate basis vectors

# Returns
- `normalCurvature::Array{T,N}`
- `geodesicCurvature::Array{T,N}`

"""
function curvatureComponents(basisVectors::AbstractCoordinateField{D,T,N},vmecVectors::VmecCoordinates{D,T,N},
                             vmec::VMEC.VmecData) where {D,T,N}
  gradX = gradX1(basisVectors)
  gradY = gradX2(basisVectors)
  B = cross(gradX,gradY)
  grad_B = gradB(vmecVectors,vmec)
  Bmag = abs(B)
  norm = abs(gradX).*abs(gradY)
  #dBdx = Threads.@spawn mu0*VMEC.dPdS(s,vmec)./(Bmag.*Bmag) .- 
  #       dot(B,cross(gradB,gradY)) ./ Bmag
  dBdX = Threads.@spawn dot(cross(B,grad_B),gradY) ./ (Bmag)
  dBdY = Threads.@spawn dot(cross(B,grad_B),gradX) ./ (Bmag)

  sinGradXGradY = Threads.@spawn abs(cross(gradX,gradY))./norm
  cosGradXGradY = Threads.@spawn abs.(dot(gradX,gradY))./norm
                  
  #normalCurvature = fetch(dBdx) .* abs(gradX) .+ fetch(dBdy) .* abs(gradY) .* fetch(cosGradXGradY)
  #geodesicCurvature = fetch(dBdy) .* abs(gradY) .* fetch(sinGradXGradY)
  
  return fetch(dBdX),vmec.signgs*fetch(dBdY)

end

#=
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
=#
