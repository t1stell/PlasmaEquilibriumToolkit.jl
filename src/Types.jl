
"""
    AbstractGeometry

Abstract supertype for different geometries.
"""
abstract type AbstractGeometry end;


"""
    AbstractMagneticGeometry

Abstract subtype of AbstractGeometry for different geometries including magnetic structures
"""
abstract type AbstractMagneticGeometry <: AbstractGeometry end;

"""
   MagneticGeometry

Subtype of AbstractMagneticGeometry. Possibly should be a struct with some 
general features of magnetic geometry (like field information)  
Right now it is empty
"""
abstract type MagneticGeometry <: AbstractMagneticGeometry end;


"""
    AbstractMagneticGeometry

Abstract subtype of AbstractMagneticGeometry for different equilibria representations (VMEC, SPEC...)

This layer of abstraction is probably not necessary
"""
abstract type AbstractMagneticEquilibrium <: AbstractMagneticGeometry end;

"""
    AbstractMagneticCoordinates

Abstract supertype for different magnetic coordinates.
"""
abstract type AbstractMagneticCoordinates end;

"""
   AbstractSurface

Abstract subtype of AbstractGeometry for a surface, not necessarily a magnetic surface
"""
abstract type AbstractSurface <: AbstractGeometry end;


"""
   AbstractMagneticSurface

Abstract subtype of AbstracSurface.  These are magnetic surfaces generally 
derived from equilibria of different types
"""
abstract type AbstractMagneticSurface <: AbstractSurface end;

"""
    NullEquilibrium()

Empty subtype of MagneticEquilibrium to represent no equilibrium
"""
struct NullEquilibrium <: AbstractMagneticEquilibrium end

"""
    CoordinateVector{T} <: SVector{3,T}
Convenience type for defining a 3-element StaticVector.
"""
const CoordinateVector{T} = SVector{3,T} where {T}

"""
    BasisVectors{T} <: SArray{Tuple{3,3},T,2,9}

Convenience type for defining the Cartesian coordinate representation of 3D curvilinear vector fields.
For a vector defined by inverse map ``R(x,y,z) = (U(x,y,z),V(x,y,z),W(x,y,z))``, the components are
represented by the 3×3 SArray:

`R = [Ux Vx Wx;Uy Vy Wy;Uz Vz Wz]`

where `Ux` is the `x`-component of `U`.
# Examples
```julia-repl
julia> using StaticArrays
julia> typeof(@SArray(ones(3,3))) <: BasisVectors{Float64}
true
```
"""
const BasisVectors{T} = SArray{Tuple{3,3},T,2,9} where {T}

"""
    SurfaceFourierData{T} where T <: AbstractFloat

Composite type representation for a single set of `cos` and `sin` coefficients and
deriatives  w.r.t `s` for a given poloidal and toroidal mode number

# Fields
- `m::T` : Poloidal mode number
- `n::T` : Toroidal mode number multiplied by number of field periods
- `cos::T` : The cosine coefficient
- `sin::T` : The sine coefficient
- `dcosds::T` : Derivative of the cosine coefficient w.r.t. `s`
- `dsinds::T` : Derivative of the sine coefficient w.r.t. `s`
"""
struct SurfaceFourierData{T}
  m::T
  n::T
  cos::T
  sin::T
  dcos_ds::T
  dsin_ds::T
end

const SurfaceFourierArray{T} = StructArray{SurfaceFourierData{T}} where T


abstract type BasisType end

#Define the singleton types
struct Covariant <: BasisType end
struct Contravariant <: BasisType end


struct MagneticSurface <: AbstractMagneticSurface
  eqType::Type
  surfaceLabel::AbstractFloat
end

struct MagneticFieldline <: AbstractMagneticSurface
  eqType::Type
  surfaceLabel::Float64
  fieldlineLabel::Float64
  toroidalAngle::Float64
end
#=
struct MagneticSurface <: MagneticGeometry
  eq::MagneticEquilibrium
  coords::AbstractArray{AbstractMagneticCoordinates}
  covariantBasis::AbstractArray
  contravariantBasis::AbstractArray
  MagneticSurface() = new()
  MagneticSurface(e::eqType,c::Union{coordType,AbstractArray{coordType}}) where {eqType <: MagneticEquilibrium, coordType <: AbstractMagneticCoordinates} = new(e,c)
end
=#
#=

struct MagneticFieldline <: MagneticGeometry
  eq::MagneticEquilibrium
  coordinates::AbstractVector{AbstractMagneticCoordinates}
  MagneticFieldline() = new()
  MagneticFieldline(e::eqType,c::Union{coordType,AbstractArray{coordType}}) where {eqType <: MagneticEquilibrium, coordType <: AbstractMagneticCoordinates} = new(e,c)
end

function MagneticSurface(eqType::Type,surface)
  @assert isstructtype(eqType) && eqType <: MagneticEquilibrium "$(eqType) is not a MagneticEquilibrium type!"
  @assert convert(Float64,surface) >= 0.0 "The surface label must be >= 0.0"
  return AbstractMagneticSurface(eqType,convert(Float64,surface))
end

function MagneticFieldline(eqType::Type,surface,label,toroidalAngle=0.0)
  @assert isstructtype(eqType) && eqType <: MagneticEquilibrium "$(eqType) is not a MagneticEquilibrium type!"
  @assert convert(Float64,surface) >= 0.0 "The surface label must be >= 0.0"
  @assert convert(Float64,abs(label)) <= 2π "The fieldline label at zero toroidal angle must be <= 2π"
  @assert convert(Float64,abs(toroidalAngle)) <= 2π "The initial toroidal angle must be <= 2π"
  return AbstractMagneticFieldline(eqType,convert(Float64,surface),convert(Float64,label),convert(Float64,toroidalAngle))
end
=#
#=
function MagneticCoordinateArray(C::Type{MC},α::Union{Real,AbstractVector{Real}},
                             β::Union{Real,AbstractVector{Real}},
                             η::Union{Real,AbstractVector{Real}};
                             grid=true,fastdim=3) where MC <: AbstractMagneticCoordinates
  T = eltype(promote(first(α),first(β),first(η))
  scalar = prod(isempty.((axes(α),axes(β),axes(η))))
  if scalar
    return C(α,β,η)
  else
    dims, indices = buildDimsAndIndices(axes(α),axes(β),axes(η),firstDim=firstDim,secondDim=secondDim)
    coordinates = Array{C{T}}(undef,dims)
    @inbounds @simd for (i,index) in indices
      coordinates[i] = C{T}(promote(α[index[1]],β[index[2]],η[index[3]]))
    end
    return coordinates
  end
end
=#
#=
struct MagneticFieldElement
  coordType::Type
  coords::AbstractMagneticCoordinates
  basis::BasisVectors{Float64}
  B::Float64
  ∇B::CoordinateVector{Float64}
end

struct MagneticFieldline
  coordType::Type
  coords::AbstractVector{MagneticFieldElement}
  ι::Float64
end

function Base.getindex(x::MagneticFieldline, I)
  return Base.getindex(Base.getfield(x, :coords), I)
end

function Base.setindex!(
  x::MagneticFieldline,
  val::MC,
  I,
) where {MC<:AbstractMagneticCoordinates}
  Base.setindex!(getfield(x, :coords), val, I)
end

function Base.size(x::MagneticFieldline, d::Integer = 0)
  return d > 0 ? Base.size(getfield(x, :coords), d) :
         Base.size(getfield(x, :coords))
end

function Base.length(x::MagneticFieldline)
  return Base.length(getfield(x, :coords))
end

function Base.eltype(x::MagneticFieldline)
  return Base.eltype(getfield(x, :coords))
end

function Base.iterate(x::MagneticFieldline, state = 1)
  return Base.iterate(getfield(x, :coords), state)
end
=#

