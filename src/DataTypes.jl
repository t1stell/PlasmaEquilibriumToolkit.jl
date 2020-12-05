using StaticArrays

const CoordinateVector{T} = SVector{3,T} where T
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
