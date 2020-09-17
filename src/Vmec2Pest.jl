import VMEC

function vmec2pest(vmec::VMEC.VmecData,fieldline::PET.Fieldline{Float64})
  iota, dIotads = VMEC.iotaShearPair(fieldline.s,vmec)
  edgeFlux2Pi = vmec.phi[vmec.ns]*vmec.signgs/(2*π) 
  zeta = fieldline.zeta
  nz = fieldline.length
  thetaVmec = VMEC.findThetaVmec(fieldline.s,fieldline.alpha .+ iota.*zeta,zeta,vmec)

  tBs = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;ds=true)
  tBtv = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;dpoloidal=true)
  tBz = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;dtoroidal=true)
  
  trs = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;ds=true)
  tzs = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;ds=true)
  tls = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,ds=true)

  trp = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;dpoloidal=true)
  tzp = Threads.@spawn  VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;dpoloidal=true)
  tlp = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,dpoloidal=true)

  trt = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;dtoroidal=true)
  tzt = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;dtoroidal=true)
  tlt = Threads.@spawn VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,dtoroidal=true)

  BVectorVmec = Vector3D{Float64}(fetch(tBs),fetch(tBtv),fetch(tBz),:co)
  basisS_vmec = Vector3D{Float64}(fetch(trs),fetch(tzs),fetch(tls),:co)
  basisThetaVmec_vmec = Vector3D{Float64}(fetch(trp),fetch(tzp),fetch(tlp),:co)
  basisZeta_vmec = Vector3D{Float64}(fetch(trt),fetch(tzt),fetch(tlt),:co)

  #=
  BVectorVmec = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;ds=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;dpoloidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns,dtoroidal=true),:co)
  basisS_vmec = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;ds=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;ds=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,ds=true),:co)
  basisThetaVmec_vmec = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;dpoloidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;dpoloidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,dpoloidal=true),:co)
  basisZeta_vmec = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;dtoroidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;dtoroidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,dtoroidal=true),:co)
=#
  R = VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns)
  Z = VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc)

  # Jacobian at each point along the field line
  J = VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:gmnc,:gmns)

  cosZeta = cos.(zeta)
  sinZeta = sin.(zeta)
  X = R.*cosZeta
  Y = R.*sinZeta

  basisS_xyz = Vector3D{Float64}(getfield(basisS_vmec,:x).*cosZeta,getfield(basisS_vmec,:x).*sinZeta,
                                 getfield(basisS_vmec,:y),:co)
  basisThetaVmec_xyz = Vector3D{Float64}(getfield(basisThetaVmec_vmec,:x).*cosZeta,getfield(basisThetaVmec_vmec,:x).*sinZeta,
                                  getfield(basisThetaVmec_vmec,:y),:co)
  basisZeta_xyz = Vector3D{Float64}(getfield(basisZeta_vmec,:x).*cosZeta .- R.*sinZeta,
                                    getfield(basisZeta_vmec,:x).*sinZeta .+ R.*cosZeta,
                                    getfield(basisZeta_vmec,:y),:co)

  metric_xyz = MetricTensor{Float64,1}(fill(1.0,nz),fill(0.0,nz),fill(1.0,nz),fill(0.0,nz),fill(0.0,nz),fill(1.0,nz),tuple(nz))
  jac_xyz = fill(1.0,nz)
  # VMEC contravariant basis vectors 
  gradS_xyz = crossLoop(basisThetaVmec_xyz,basisZeta_xyz,J)
  gradThetaVmec_xyz = crossLoop(basisZeta_xyz,basisS_xyz,J)
  gradZeta_xyz = crossLoop(basisS_xyz,basisThetaVmec_xyz,J)
 
  lambdaSFactor = Vector{Float64}(getfield(basisS_vmec,:z) .- dIotads .* zeta)
  lambdaThetaVmecFactor = Vector{Float64}(getfield(basisThetaVmec_vmec,:z) .+ 1.0)
  lambdaZetaFactor = Vector{Float64}(getfield(basisZeta_vmec,:z) .- iota)

  gradPsi_xyz = gradS_xyz*edgeFlux2Pi
  gradAlpha_xyz = ((gradS_xyz * lambdaSFactor) + (gradThetaVmec_xyz * lambdaThetaVmecFactor) + 
                  (gradZeta_xyz * lambdaZetaFactor))
  gradTheta_xyz = ((gradS_xyz * getfield(basisS_vmec,:z)) + (gradThetaVmec_xyz * lambdaThetaVmecFactor) + 
                  (gradZeta_xyz * getfield(basisZeta_vmec,:z)))
  basisTheta_xyz = crossLoop(gradZeta_xyz,gradS_xyz,J)

  gradB_xyz = ((gradS_xyz * getfield(BVectorVmec,:x)) + (gradThetaVmec_xyz * getfield(BVectorVmec,:y)) + 
              (gradZeta_xyz * getfield(BVectorVmec,:z))) 
  B_xyz = (((basisZeta_xyz * lambdaThetaVmecFactor) - (basisThetaVmec_xyz * lambdaZetaFactor)) / J)*edgeFlux2Pi

  BCrossGradBDotGradPsi_xyz = dot(B_xyz,crossLoop(gradB_xyz,gradPsi_xyz,jac_xyz),metric_xyz) 
  BCrossGradBDotGradAlpha_xyz = dot(B_xyz,crossLoop(gradB_xyz,gradAlpha_xyz,jac_xyz),metric_xyz)
  BCrossGradPsiDotGradAlpha_xyz = dot(B_xyz,crossLoop(gradPsi_xyz,gradAlpha_xyz,jac_xyz),metric_xyz)


end
