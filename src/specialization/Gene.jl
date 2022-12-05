using Printf
export GeneFromPest, GeneFromFlux, geneGeometryCoefficients, GeneCoordinates

struct GeneFromPest <: Transformation; end
struct GeneFromFlux <: Transformation; end

struct GeneCoordinates{T, A}
    x::T
    y::A
    z::A
end

function GeneCoordinates(x, y, z)
    x2, y2, z2 = promote(x, y, z)
    return GeneCoordinates{typeof(x2),typeof(y2)}(x2, y2, z2)
end

function writeGeneGeometry(filename::String,
                           coords,
                           eq::MG,
                           g::AbstractVector{SVector{6,T}},
                           modB::AbstractVector{T},
                           jac::AbstractVector{T},
                           K1::AbstractVector{T},
                           K2::AbstractVector{T},
                           dBdθ::AbstractVector{T}) where {T, MG <: AbstractGeometry}
  Ba = abs(eq.phi[end]/(π*eq.Aminor_p^2))
  geneFile = filename*".dat"
  io = open(geneFile,"w")
  # Write the parameters namelist
  s0 = getfield(first(coords),1)*2π/eq.phi[end]*eq.signgs
  #Assumes coords variable is PEST (ψ,α=θ-ιζ,ζ)
  α0 = -getfield(first(coords),2)/eq.iota[1]
  coordString = "!PEST coordinates\n"
  q0 = 1.0/eq.iota[1]
  shat = -2*s0/q0*eq.iota[2]*q0^2
  lengthString = "!major, minor radius[m] = "*string(eq.Rmajor_p)*" "*string(eq.Aminor_p)*"\n"
  Bstring = "!Bref = "*string(Ba)*"\n"
  gridString = "gridpoints = "*string(length(coords))*"\n"
  npolString = "n_pol = "*string(abs(round(Int,getfield(last(coords),3)*eq.iota[1]/π)))*"\n"
  dpdx_string = "my_dpdx = "*string(-4.0*sqrt(eq.s)*eq.pres[2]/(Ba^2)*4π*1e-7)*"\n"

  write(io,"&parameters\n")
  write(io,coordString)
  write(io,"!s0, alpha0 = "*string(s0)*" "*string(α0)*"\n")
  write(io,lengthString)
  write(io,Bstring)
  write(io,dpdx_string)
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
