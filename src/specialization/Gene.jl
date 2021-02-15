using Printf
export GeneFromPest, GeneFromFlux, geneGeometryCoefficients

struct GeneFromPest <: Transformation; end
struct GeneFromFlux <: Transformation; end

function writeGeneGeometry(filename::String,coords::AbstractVector{PestCoordinates},eq::ME,
                           g::AbstractVector{NTuple{6,Float64}},
                           modB::AbstractVector{Float64},jac::AbstractVector{Float64},K1::AbstractVector{Float64},
                           K2::AbstractVector{Float64},dBdθ::AbstractVector{Float64}) where ME <: MagneticEquilibrium
  geneFile = filename*".dat"
  io = open(geneFile,"w")
  # Write the parameters namelist
  s0 =getfield(first(coords),1)*2π/eq.phi[end]*eq.signgs
  α0 = getfield(first(coords),2)
  coordString = "!PEST coordinates\n"
  q0 = 1.0/eq.iota[1]
  shat = -2*s0/q0*eq.iota[2]*q0^2
  lengthString = "!major, minor radius[m] = "*string(eq.Rmajor_p)*" "*string(eq.Aminor_p)*"\n"
  Bstring = "!Bref = "*string(eq.phi[end]/(π*eq.Aminor_p^2))*"\n"
  gridString = "gridpoints = "*string(length(coords))*"\n"
  npolString = "n_pol = "*string(abs(round(Int,getfield(last(coords),3)*eq.iota[1]/π)))*"\n"

  write(io,"&parameters\n")
  write(io,coordString)
  write(io,"!s0, alpha0 = "*string(s0)*" "*string(α0)*"\n")
  write(io,lengthString)
  write(io,Bstring)
  write(io,"my_dpdx = "*string(zero(Float64))*"\n")
  write(io,"q0 = "*string(q0)*"\n")
  write(io,"shat = "*string(shat)*"\n")
  write(io,gridString)
  write(io,npolString)
  write(io,"sign_Ip_CW = "*string(convert(Int,sign(getfield(first(coords),1))))*"\n")
  write(io,"sign_Bt_CW = "*string(convert(Int,sign(getfield(first(coords),1))))*"\n")
  write(io,"/\n")

  for i = 1:length(coords)
    s = @sprintf "%23.16f %23.16f %23.16f %23.16f %23.16f %23.16f %23.16f %23.16f %23.16f\n" g[i][1] g[i][2] g[i][3] modB[i] jac[i] K2[i] K1[i] dBdθ[i] 0.0
    write(io,s)
  end
  close(io)
end
