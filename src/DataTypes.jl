using StaticArrays

"""
    CoordinateVector{T} <: SVector{3,T}

Convenience type for defining a 3-element StaticVector.
"""
const CoordinateVector{T} = SVector{3,T} where T

"""
    BasisVectors{T} <: SArray{Tuple{3,3},T,2,9}

Convenience type for defining the Cartesian coordinate representation of 3D curvilinear vector fields.
For a vector defined by inverse map ``R(x,y,z) = (U(x,y,z),V(x,y,z),W(x,y,z))``, the components are
represented by the 3×3 SArray:

`R = [Ux Uy Uz;Vx Vy Vz;Wx Wy Wz]`

where `Ux` is the `x`-component of `U`.
# Examples
```julia-repl
julia> using StaticArrays
julia> typeof(@SArray(ones(3,3))) <: BasisVectors{Float64}
true
```
"""
const BasisVectors{T} = SArray{Tuple{3,3},T,2,9} where T

#=
function eq_typeof(eq::MagneticEquilibrium)
  if occursin("Vmec",string(typeof(eq)))
    return :vmec
  elseif occursin("Spec",string(typeof(eq)))
    return :spec
  end
end

struct MagneticSurface
  eqType::Symbol
  coordType::Symbol
  label
  coordinates::AbstractMatrix
end

struct MagneticFieldline
  eqType::Symbol
  coordType::Symbol
  label
  coordinates::AbstractVector
end

struct SpaceCurve
  eqType::Symbol
  coordType::Symbol
  label
  coordaintes::AbstractVector
end

function MagneticSurface(surfaceLabel,eq::MagneticEquilibrium,coords::AbstractMatrix)
  eqType = eq_typeof(eq)
  coordType = !isprimitivetype(eltype(coords)) ? Symbol(eltype(coords)) : :Cartesian
  label = surfaceLabel
  coordinates = coords
  return MagneticSurface(eq,coordType,label,coordinates)
end

function MagneticFieldline(surfaceLabel,fieldlineLabel,eq::MagneticEquilibrium,coords::AbstractVector)
  eqType = eq_typeof(eq)
  coordType = !isprimitivetype(eltype(coords)) ? Symbol(eltype(coords)) : :Cartesian
  label = (surfaceLabel,fieldlineLabel)
  coordinates = coords
  return MagneticFieldline(eq,coordType,label,coordinates)
end
=#
