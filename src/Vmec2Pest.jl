import VMEC

function vmec2pest(vmec::VMEC.VmecData,fieldline::PET.Fieldline{Float64})
  iota, dIotads = VMEC.iotaShearPair(fieldline.s,vmec)
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
  basis_VMEC_s = Vector3D{Float64}(fetch(trs),fetch(tzs),fetch(tls),:co)
  basis_VMEC_tv = Vector3D{Float64}(fetch(trp),fetch(tzp),fetch(tlp),:co)
  basis_VMEC_zeta = Vector3D{Float64}(fetch(trt),fetch(tzt),fetch(tlt),:co)

  #=
  BVectorVmec = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;ds=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns;dpoloidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:bmnc,:bmns,dtoroidal=true),:co)
  basis_VMEC_s = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;ds=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;ds=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,ds=true),:co)
  basis_VMEC_tv = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;dpoloidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:zmns,:zmnc;dpoloidal=true),
                             VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:lmns,:lmnc,dpoloidal=true),:co)
  basis_VMEC_zeta = Vector3D{Float64}(VMEC.inverseTransform(fieldline.s,thetaVmec,zeta,vmec,:rmnc,:rmns;dtoroidal=true),
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

  basis_XYZ_s = Vector3D{Float64}(getfield(basis_VMEC_s,:x).*cosZeta,getfield(basis_VMEC_s,:x).*sinZeta,
                                 getfield(basis_VMEC_s,:y),:co)
  basis_XYZ_tv = Vector3D{Float64}(getfield(basis_VMEC_tv,:x).*cosZeta,getfield(basis_VMEC_tv,:x).*sinZeta,
                                  getfield(basis_VMEC_tv,:y),:co)
  basis_XYZ_zeta = Vector3D{Float64}(getfield(basis_VMEC_zeta,:x).*cosZeta .- R.*sinZeta,
                                    getfield(basis_VMEC_zeta,:x).*sinZeta .+ R.*cosZeta,
                                    getfield(basis_VMEC_zeta,:y),:co)

  # Contravariant basis vectors
  gradS_XYZ = crossLoop(basis_XYZ_zeta,basis_XYZ_tv,J)
  gradThetaVmec_XYZ = crossLoop(basis_XYZ_zeta,basis_XYZ_s,J)
  gradZeta_XYZ = crossLoop(basis_XYZ_s,basis_XYZ_tv,J)
 
  lambdaSFactor = Vector{Float64}(getfield(basis_VMEC_s,:z) .- dIotads .* zeta)
  lambdaThetaVmecFactor = Vector{Float64}(getfield(basis_VMEC_tv,:z) .+ 1.0)
  lambdaZetaFactor = Vector{Float64}(getfield(basis_VMEC_zeta,:z) .- iota)

  gradAlpha_XYZ = ((gradS_XYZ * lambdaSFactor) + (gradThetaVmec_XYZ * lambdaThetaVmecFactor) + 
                  (gradZeta_XYZ * lambdaZetaFactor))
  gradTheta_XYZ = ((gradS_XYZ * getfield(basis_VMEC_s,:z)) + (gradThetaVmec_XYZ * lambdaThetaVmecFactor) + 
                  (gradZeta_XYZ * getfield(basis_VMEC_zeta,:z)))
  basisTheta_XYZ = crossLoop(gradZeta_XYZ,gradS_XYZ,J)

  gradB_XYZ = ((gradS_XYZ * getfield(BVectorVmec,:x)) + (gradThetaVmec_XYZ * getfield(BVectorVmec,:y)) + 
              (gradZeta_XYZ * getfield(BVectorVmec,:z))) 
  B_XYZ = ((basis_XYZ_zeta * lambdaThetaVmecFactor) - (basis_XYZ_tv * lambdaZetaFactor)) / J

  for i = 1:nz
    println(zeta[i])
    println(R[i],' ',X[i],' ',Y[i],' ',Z[i])
    println(getfield(basis_XYZ_s,:x)[i],' ',getfield(basis_XYZ_s,:y)[i],' ',getfield(basis_XYZ_s,:z)[i])
    println(getfield(basis_XYZ_tv,:x)[i],' ',getfield(basis_XYZ_tv,:y)[i],' ',getfield(basis_XYZ_tv,:z)[i])
    println(getfield(basis_XYZ_zeta,:x)[i],' ',getfield(basis_XYZ_zeta,:y)[i],' ',getfield(basis_XYZ_zeta,:z)[i])
    println(getfield(gradS_XYZ,:x)[i],' ',getfield(gradS_XYZ,:y)[i],' ',getfield(gradS_XYZ,:z)[i])
    println(getfield(gradThetaVmec_XYZ,:x)[i],' ',getfield(gradThetaVmec_XYZ,:y)[i],' ',getfield(gradThetaVmec_XYZ,:z)[i])
    println(getfield(gradZeta_XYZ,:x)[i],' ',getfield(gradZeta_XYZ,:y)[i],' ',getfield(gradZeta_XYZ,:z)[i])
    println("====")
  end

end
