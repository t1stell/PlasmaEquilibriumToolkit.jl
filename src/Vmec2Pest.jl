import VMEC

function vmec2pest(vmec::VMEC.VmecData,surface::Float64,alpha::Float64,zeta::Vector{Float64})
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
  tzp = Threads.@spawn  VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;dpoloidal=true)
  tlp = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,dpoloidal=true)

  trt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:rmnc,:rmns;dtoroidal=true)
  tzt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:zmns,:zmnc;dtoroidal=true)
  tlt = Threads.@spawn VMEC.inverseTransform(surface,thetaVmec,zeta,vmec,:lmns,:lmnc,dtoroidal=true)

  BVmecDerivatives = Vector3D{Float64}(fetch(tBs),fetch(tBtv),fetch(tBz))
  basisS_vmec = Vector3D{Float64}(fetch(trs),fetch(tzs),fetch(tls))
  basisThetaVmec_vmec = Vector3D{Float64}(fetch(trp),fetch(tzp),fetch(tlp))
  basisZeta_vmec = Vector3D{Float64}(fetch(trt),fetch(tzt),fetch(tlt))

  R = fetch(tR)
  Z = fetch(tZ)
  J = fetch(tJ)

  cosZeta = cos.(zeta)
  sinZeta = sin.(zeta)

  basisS_xyz = Vector3D{Float64}(getfield(basisS_vmec,:x).*cosZeta,getfield(basisS_vmec,:x).*sinZeta,
                                 getfield(basisS_vmec,:y))
  basisThetaVmec_xyz = Vector3D{Float64}(getfield(basisThetaVmec_vmec,:x).*cosZeta,getfield(basisThetaVmec_vmec,:x).*sinZeta,
                                  getfield(basisThetaVmec_vmec,:y))
  basisZeta_xyz = Vector3D{Float64}(getfield(basisZeta_vmec,:x).*cosZeta .- R.*sinZeta,
                                    getfield(basisZeta_vmec,:x).*sinZeta .+ R.*cosZeta,
                                    getfield(basisZeta_vmec,:y))

  # VMEC contravariant basis vectors 
  gradS_xyz = cross(basisThetaVmec_xyz,basisZeta_xyz) / J
  gradThetaVmec_xyz = cross(basisZeta_xyz,basisS_xyz) / J
  gradZeta_xyz = cross(basisS_xyz,basisThetaVmec_xyz) / J
 
  lambdaSFactor = Vector{Float64}(getfield(basisS_vmec,:z) .- dIotads .* zeta)
  lambdaThetaVmecFactor = Vector{Float64}(getfield(basisThetaVmec_vmec,:z) .+ 1.0)
  lambdaZetaFactor = Vector{Float64}(getfield(basisZeta_vmec,:z) .- iota)

  basisPsi_xyz = basisS_xyz*edgeFlux2Pi
  gradPsi_xyz = gradS_xyz*edgeFlux2Pi
  gradAlpha_xyz = ((gradS_xyz * lambdaSFactor) + (gradThetaVmec_xyz * lambdaThetaVmecFactor) + 
                  (gradZeta_xyz * lambdaZetaFactor))
  gradTheta_xyz = ((gradS_xyz * getfield(basisS_vmec,:z)) + (gradThetaVmec_xyz * lambdaThetaVmecFactor) + 
                  (gradZeta_xyz * getfield(basisZeta_vmec,:z)))
  basisTheta_xyz = cross(gradZeta_xyz,gradS_xyz) * J

  gradB_xyz = ((gradS_xyz * getfield(BVmecDerivatives,:x)) + (gradThetaVmec_xyz * getfield(BVmecDerivatives,:y)) + 
              (gradZeta_xyz * getfield(BVmecDerivatives,:z))) 

  B_xyz = (((basisZeta_xyz * lambdaThetaVmecFactor) - (basisThetaVmec_xyz * lambdaZetaFactor)) / J)*edgeFlux2Pi

  # Compute the different components of the metric tensor
  t11 = Threads.@spawn dot(gradPsi_xyz,gradPsi_xyz)
  t12 = Threads.@spawn dot(gradPsi_xyz,gradAlpha_xyz)
  t22 = Threads.@spawn dot(gradAlpha_xyz,gradAlpha_xyz)
  t13 = Threads.@spawn dot(gradPsi_xyz,gradZeta_xyz)
  t23 = Threads.@spawn dot(gradAlpha_xyz,gradZeta_xyz)
  t33 = Threads.@spawn dot(gradZeta_xyz,gradZeta_xyz)
  metric = 




  #=
  for i = 1:nz
    println(zeta[i])
    println(R[i],' ',X[i],' ',Y[i],' ',Z[i])
    println(getfield(basisS_xyz,:x)[i],' ',getfield(basisS_xyz,:y)[i],' ',getfield(basisS_xyz,:z)[i])
    println(getfield(basisThetaVmec_xyz,:x)[i],' ',getfield(basisThetaVmec_xyz,:y)[i],' ',getfield(basisThetaVmec_xyz,:z)[i])
    println(getfield(basisZeta_xyz,:x)[i],' ',getfield(basisZeta_xyz,:y)[i],' ',getfield(basisZeta_xyz,:z)[i])
    println(getfield(gradPsi_xyz,:x)[i],' ',getfield(gradPsi_xyz,:y)[i],' ',getfield(gradPsi_xyz,:z)[i])
    println(getfield(gradThetaVmec_xyz,:x)[i],' ',getfield(gradThetaVmec_xyz,:y)[i],' ',getfield(gradThetaVmec_xyz,:z)[i])
    println(getfield(gradZeta_xyz,:x)[i],' ',getfield(gradZeta_xyz,:y)[i],' ',getfield(gradZeta_xyz,:z)[i])
    println(getfield(gradAlpha_xyz,:x)[i],' ',getfield(gradAlpha_xyz,:y)[i],' ',getfield(gradAlpha_xyz,:z)[i])
    println(getfield(gradTheta_xyz,:x)[i],' ',getfield(gradTheta_xyz,:y)[i],' ',getfield(gradTheta_xyz,:z)[i])
    println(getfield(gradB_xyz,:x)[i],' ',getfield(gradB_xyz,:y)[i],' ',getfield(gradB_xyz,:z)[i])
    println(getfield(B_xyz,:x)[i],' ',getfield(B_xyz,:y)[i],' ',getfield(B_xyz,:z)[i])
    println(dBdAlpha_xyz[i],' ',sinGradPsiGradAlpha[i],' ',curvGeo[i],' ',BCrossGradBDotGradAlpha_xyz[i])
    println("====")
  end
  =#
  
  pestFieldline = Fieldline{Float64}()
  pestFieldline.length = nz
  setfield!(pestFieldline,:x,fill(surface,nz))
  setfield!(pestFieldline,:y,fill(alpha,nz))
  setfield!(pestFieldline,:z,zeta)
  setfield!(pestFieldline,:B,B_xyz)
  setfield!(pestFieldline,:dBdz,getfield(BVmecDerivatives,:z))
  setfield!(pestFieldline,:coBasis_x,basisPsi_xyz)
  setfield!(pestFieldline,:coBasis_y,basisTheta_xyz)
  setfield!(pestFieldline,:coBasis_z,basisZeta_xyz)
  setfield!(pestFieldline,:contraBasis_x,gradPsi_xyz)
  setfield!(pestFieldline,:contraBasis_y,gradAlpha_xyz)
  setfield!(pestFieldline,:contraBasis_z,gradZeta_xyz)
  setfield!(pestFieldline,:metric,MetricTensor(fetch(t11),fetch(t12),fetch(t22),fetch(t13),fetch(t23),fetch(t33)))
  setfield!(pestFieldline,:jacobian,abs.(J))
  setfield!(pestFieldline,:rotationalTransform,iota)
  return pestFieldline

end
