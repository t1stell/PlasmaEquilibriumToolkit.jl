import VMEC

"""
    computeVmecDerivatives(::VMEC.VmecData,s::Float64,α::Float64,ζ::Vector{Float64})

Computes the R and Z values, as well as derivatives w.r.t to the VMEC coordinates and the scalar Jacobian
at each point along a field line on a magnetic surface `s`, labeled by `α`, parameterized by the points in the vector `ζ`.

# Returns
- `basisS_vmec`: VectorField with components specified by (∂R/∂s,∂Z/∂s,∂Λ/∂s)
- `basisThetaVmec_vmec`: VectorField with components specified by (∂R/∂θᵥ,∂Z/∂θᵥ,∂Λ/∂θᵥ)
- `basisZeta_vmec`: VectorField with components specified by (∂R/∂ζ,∂Z/∂ζ,∂Λ/∂ζ)
- `R`: Array with values of the cylindrical coordinate R
- `Z`: Array with values of the cylindrical coordinate Z
- `J`: Array with values of the scalar jacobian
- `coords`: TupleField with the coordinates of the field line
"""
function computeVmecDerivatives(vmec::VMEC.VmecData,surface::Float64,alpha::Float64,zeta::Vector{Float64})
  iota, dIotads = VMEC.iotaShearPair(surface,vmec)
  edgeFlux2Pi = vmec.phi[vmec.ns]*vmec.signgs/(2*π) 
  psiPrime = vmec.phi[vmec.ns]
  nz = length(zeta)
  thetaVmec = VMEC.findThetaVmec(surface,alpha .+ iota.*zeta,zeta,vmec)

  R = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns)
  Z = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc)
  J = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:gmnc,:gmns)

  #tBs = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:bmnc,:bmns;ds=true)
  #tBtv = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:bmnc,:bmns;dpoloidal=true)
  #tBz = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:bmnc,:bmns;dtoroidal=true)
  
  trs = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;ds=true)
  tzs = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;ds=true)
  tls = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,ds=true)

  trp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;dpoloidal=true)
  tzp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;dpoloidal=true)
  tlp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,dpoloidal=true)

  trt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;dtoroidal=true)
  tzt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;dtoroidal=true)
  tlt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,dtoroidal=true)

  dRZLdS_vmec = VectorField(fetch(trs),fetch(tzs),fetch(tls))
  dRZLdThetaVmec_vmec = VectorField(fetch(trp),fetch(tzp),fetch(tlp))
  dRZLdZeta_vmec = VectorField(fetch(trt),fetch(tzt),fetch(tlt))
  
  return dRZLdS_vmec, dRZLdThetaVmec_vmec, dRZLdZeta_vmec, fetch(R), fetch(Z), fetch(J), TupleField(fill(surface,length(zeta)),fill(alpha,length(zeta)),zeta)
end

"""
    computeVmecVectors(basisS_vmec::VectorField{D,T,N},basisThetaVmec_vmec::VectorField{D,T,N},
                       basisZeta_vmec::VectorField{D,T,N},R::Array{T,N},
                       Z::Array{T,N},J::Array{T,N},coords::TupleField{D,T,N})

Computes the covariant and contravariant basis vectors w.r.t to the VMEC coordinates
at each point along a field line on a magnetic surface `s`, labeled by `α`, parameterized by the points in the vector `ζ`.

# Returns
- `VmecCoordinates`

See also: [`computeVmecBasis`](@ref), [`VmecCoordinates`](@ref)
"""
function computeVmecVectors(basisS_vmec::VectorField{D,T,N},basisThetaVmec_vmec::VectorField{D,T,N},
                            basisZeta_vmec::VectorField{D,T,N},R::Array{T,N},
                            Z::Array{T,N},J::Array{T,N},coords::TupleField{D,T,N}) where {D,T,N}
  zeta = component(coords,3) 
  cosZeta = cos.(zeta)
  sinZeta = sin.(zeta)

  basisS_xyz = VectorField(component(basisS_vmec,1).*cosZeta,component(basisS_vmec,1).*sinZeta,
                           component(basisS_vmec,2))
  basisThetaVmec_xyz = VectorField(component(basisThetaVmec_vmec,1).*cosZeta,component(basisThetaVmec_vmec,1).*sinZeta,
                                         component(basisThetaVmec_vmec,2))
  basisZeta_xyz = VectorField(component(basisZeta_vmec,1).*cosZeta .- R.*sinZeta,
                              component(basisZeta_vmec,1).*sinZeta .+ R.*cosZeta,
                              component(basisZeta_vmec,2))

  # VMEC contravariant basis vectors 
  gradS_xyz = cross(basisThetaVmec_xyz,basisZeta_xyz) / J
  gradThetaVmec_xyz = cross(basisZeta_xyz,basisS_xyz) / J
  gradZeta_xyz = cross(basisS_xyz,basisThetaVmec_xyz) / J

  return VmecCoordinates(getfield(coords,:data),
                         basisS_xyz,basisThetaVmec_xyz,basisZeta_xyz,
                         gradS_xyz,gradThetaVmec_xyz,gradZeta_xyz)
end

function computeVmecVectors(vmec::VMEC.VmecData,surface::Float64,alpha::Float64,zeta::Vector{Float64})
  vmecVectorArgs = computeVmecDerivatives(vmec,surface,alpha,zeta)
  return computeVmecVectors(vmecVectorArgs...)
end

"""
    computePestVectors(::VMEC.VmecData,s::Float64,α::Float64,ζ::Float64)

Computes the covariant and contravariant basis vectors w.r.t to PEST coordinates
at each point along a field line on a magnetic surface `s`, labeled by `α`, parameterized by the points in the vector `ζ`.

# Returns
- `PestCoordinates`
- `SThetaZetaCoordinates`

See also: [`computeVmecVectors`](@ref), [`PestCoordinates`](@ref), [`SThetaZetaCoordinates`](@ref)
"""
function computePestVectors(vmec::VMEC.VmecData,surface::Float64,alpha::Float64,zeta::Vector{Float64})
  vmecVectorArgs = basisS_vmec, basisThetaVmec_vmec, basisZeta_vmec, R, Z, J, coords = computeVmecBasis(vmec,surface,alpha,zeta) 
  vmecVectors = computeVmecVectors(vmecVectorArgs...)

  edgeFlux2Pi = vmec.phi[vmec.ns]*vmec.signgs/(2*π) 
  iota, dIotads = VMEC.iotaShearPair(surface,vmec)
  zeta = component(coords,3)

  lambdaSFactor = component(basisS_vmec,3) .- dIotads .* zeta
  lambdaThetaVmecFactor = component(basisThetaVmec_vmec,3) .+ 1.0
  lambdaZetaFactor = component(basisZeta_vmec,3) .- iota

  basisPsi_xyz = eX1(vmecVectors)*edgeFlux2Pi
  gradPsi_xyz = gradX1(vmecVectors)*edgeFlux2Pi
  gradAlpha_xyz = ((gradX1(vmecVectors) * lambdaSFactor) + (gradX2(vmecVectors) * lambdaThetaVmecFactor) + 
                   (gradX3(vmecVectors) * lambdaZetaFactor))
  gradTheta_xyz = ((gradX1(vmecVectors) * component(basisS_vmec,3)) + (gradX2(vmecVectors) * lambdaThetaVmecFactor) + 
                   (gradX3(vmecVectors) * component(basisZeta_vmec,3)))
  basisTheta_xyz = cross(gradX3(vmecVectors),gradX1(vmecVectors)) * J

  #gradB_xyz = ((gradS_xyz * component(BVmecDerivatives,1)) + (gradThetaVmec_xyz * component(BVmecDerivatives,2)) + 
  #            (gradZeta_xyz * component(BVmecDerivatives,3))) 

  #B_xyz = (((basisZeta_xyz * lambdaThetaVmecFactor) - (basisThetaVmec_xyz * lambdaZetaFactor)) / J)*edgeFlux2Pi

  return (PestCoordinates(getfield(coords,:data),
                          basisPsi_xyz,basisTheta_xyz,eX1(vmecVectors),
                          gradPsi_xyz,gradAlpha_xyz,gradX3(vmecVectors)),
          SThetaZetaCoordinates(getfield(TupleField(component(coords,1),component(coords,2) .+ iota.*zeta,zeta),:data),
                                eX1(vmecVectors),basisTheta_xyz,eX1(vmecVectors),
                                gradX1(vmecVectors),gradTheta_xyz,gradX3(vmecVectors)))
end
