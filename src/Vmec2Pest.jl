import VMEC

function computePestVectors(vmec::VMEC.VmecData,surface::Float64,alpha::Float64,zeta::Vector{Float64})
  iota, dIotads = VMEC.iotaShearPair(surface,vmec)
  edgeFlux2Pi = vmec.phi[vmec.ns]*vmec.signgs/(2*π) 
  psiPrime = vmec.phi[vmec.ns]
  nz = length(zeta)
  thetaVmec = VMEC.findThetaVmec(surface,alpha .+ iota.*zeta,zeta,vmec)

  # Compute R, Z and scalar Jacobian at each point along the field line
  tR = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns)
  tZ = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc)
  tJ = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:gmnc,:gmns)

  tBs = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:bmnc,:bmns;ds=true)
  tBtv = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:bmnc,:bmns;dpoloidal=true)
  tBz = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:bmnc,:bmns;dtoroidal=true)
  
  trs = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;ds=true)
  tzs = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;ds=true)
  tls = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,ds=true)

  trp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;dpoloidal=true)
  tzp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;dpoloidal=true)
  tlp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,dpoloidal=true)

  trt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;dtoroidal=true)
  tzt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;dtoroidal=true)
  tlt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,dtoroidal=true)

  BVmecDerivatives = VectorField(fetch(tBs),fetch(tBtv),fetch(tBz))
  basisS_vmec = VectorField(fetch(trs),fetch(tzs),fetch(tls))
  basisThetaVmec_vmec = VectorField(fetch(trp),fetch(tzp),fetch(tlp))
  basisZeta_vmec = VectorField(fetch(trt),fetch(tzt),fetch(tlt))
   
  R = fetch(tR)
  Z = fetch(tZ)
  J = fetch(tJ)

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
 
  lambdaSFactor = component(basisS_vmec,3) .- dIotads .* zeta
  lambdaThetaVmecFactor = component(basisThetaVmec_vmec,3) .+ 1.0
  lambdaZetaFactor = component(basisZeta_vmec,3) .- iota

  basisPsi_xyz = basisS_xyz*edgeFlux2Pi
  gradPsi_xyz = gradS_xyz*edgeFlux2Pi
  gradAlpha_xyz = ((gradS_xyz * lambdaSFactor) + (gradThetaVmec_xyz * lambdaThetaVmecFactor) + 
                  (gradZeta_xyz * lambdaZetaFactor))
  gradTheta_xyz = ((gradS_xyz * component(basisS_vmec,3)) + (gradThetaVmec_xyz * lambdaThetaVmecFactor) + 
                  (gradZeta_xyz * component(basisZeta_vmec,3)))
  basisTheta_xyz = cross(gradZeta_xyz,gradS_xyz) * J

  #=
  gradB_xyz = ((gradS_xyz * getfield(BVmecDerivatives,:x)) + (gradThetaVmec_xyz * getfield(BVmecDerivatives,:y)) + 
              (gradZeta_xyz * getfield(BVmecDerivatives,:z))) 

  =#
  B_xyz = (((basisZeta_xyz * lambdaThetaVmecFactor) - (basisThetaVmec_xyz * lambdaZetaFactor)) / J)*edgeFlux2Pi

  return (PestCoordinates(getfield(TupleField(fill(surface,length(zeta)),fill(alpha,length(zeta)),zeta),:data),
                          basisPsi_xyz,basisTheta_xyz,basisZeta_xyz,
                          gradPsi_xyz,gradAlpha_xyz,gradZeta_xyz),
          SThetaZetaCoordinates(getfield(TupleField(fill(surface,length(zeta)),alpha .+ iota.*zeta,zeta),:data),
                                basisS_xyz,basisTheta_xyz,basisZeta_xyz,
                                gradS_xyz,gradTheta_xyz,gradZeta_xyz),
         B_xyz)
end
