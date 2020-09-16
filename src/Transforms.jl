import VMEC

function computeB!(vmec::VMEC.VmecData,fieldline::PET.Fieldline{Float64})
  iota = VMEC.iota(fieldline.s,vmec)
  thetaVmec = Vector{Float64}(undef,fieldline.length)
  map!(zeta->VMEC.findThetaVmec(fieldline.s,fieldline.alpha + iota*zeta,zeta),thetaVmec,fieldline.zeta)
  B = VMEC.inverseTransform(fieldline.s,thetaVmec,fieldline.zeta,vmec,:bmnc,:bmns) 
end

