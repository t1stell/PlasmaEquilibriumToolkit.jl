using StaticArrays

const CoordinateVector{T} = SVector{3,T} where T
const BasisVectors{T} = SArray{Tuple{3,3},T,2,9} where T

abstract type AbstractMagneticGeometry end;
abstract type MagneticGeometry <: AbstractMagneticGeometry end;

struct AbstractMagneticSurface <: AbstractMagneticGeometry
  eqType::Type
  coord::Type
  surfaceLabel::Float64
end

struct MagneticSurface <: MagneticGeometry
  eq::MagneticEquilibrium
  coords::AbstractArray{MagneticCoordinates}
  MagneticSurface() = new()
end

struct AbstractMagneticFieldline <: AbstractMagneticGeometry
  eqType::Type
  coordinates::Type
  surfaceLabel::Float64
  fieldlineLabel::Float64
end

struct MagneticFieldline <: MagneticGeometry
  eq::MagneticEquilibrium
  coordinates::AbstractVector{MagneticCoordinates}
end

function MagneticSurface(eqType::Type,coordType::Type,surface)
  @assert isstructtype(eqType) && eqType <: MagneticEquilibrium "$(eqType) is not a MagneticEquilibrium type!"
  @assert isstructtype(coordType) && coordType <: MagneticCoordinates "$(coordType) is not a MagneticCoordinates type!"
  @assert convert(Float64,surface) >= 0.0 "The surface label must be >= 0.0"
  return AbstractMagneticSurface(eqType,coordType,convert(Float64,surface))
end

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
