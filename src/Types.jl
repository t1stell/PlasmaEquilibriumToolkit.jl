
"""
    AbstractMagneticEquilibrium

Abstract supertype for different magnetic equilibrium representations (VMEC, SPEC,...).
"""
abstract type AbstractMagneticEquilibrium end;

"""
    AbstractMagneticCoordinates

Abstract supertype for different magnetic coordinates.
"""
abstract type AbstractMagneticCoordinates end;

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

abstract type BasisType end

#Define the singleton types
struct Covariant <: BasisType end
struct Contravariant <: BasisType end

abstract type AbstractMagneticGeometry end;
abstract type MagneticGeometry <: AbstractMagneticGeometry end;

struct AbstractMagneticSurface <: AbstractMagneticGeometry
  eqType::Type
  surfaceLabel::AbstractFloat
end

macro interpolate_array(coord_type, args...)
  names = fieldnames(eval(coord_type))
  kwargs = Expr[]
  for (name, arg) in zip(names, args)
    push!(kwargs, Expr(:(kw), name, arg))
  end
  SA_expr = Expr(:(curly), :StructArray, coord_type)
  #return :(StructArray{$coord_type}($kwargs...))
  return esc(Expr(:call, SA_expr, kwargs...))
end

#=
function MagneticCoordinateGrid(
  C::CT,
  x::AbstractVector,
  y::AbstractVector,
  z::AbstractVector;
  interpolate = true,
) where {CT<:AbstractMagneticCoordinates}
  if !interpolate
    (size(x) == size(y) && size(y) == size(z)) ||
      throw(DimensionMismatch("Dimensions of input arrays do not match"))
    coordType = typeof(C(first(x), first(y), first(z)))
    grid = StructArray{coordType}(undef, size(x))
    for i = 1:length(grid)
      grid[i] = C(x[i], y[i], z[i])
    end
    return grid
  else
    return @interpolate_array(C, x, y, z)
  end
end
=#
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
struct AbstractMagneticFieldline <: AbstractMagneticGeometry
  eqType::Type
  surfaceLabel::Float64
  fieldlineLabel::Float64
  toroidalAngle::Float64
end

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

function MagneticCoordinateGrid(
  C::Type{MC},
  α::Real,
  β::Real,
  η::AbstractVector{VT},
) where {VT} where {MC<:AbstractMagneticCoordinates}
  T = typeof(C(α, β, first(η)))
  coords = StructVector{T}(undef, length(η))
  @inbounds for i = 1:length(η)
    coords[i] = C(α, β, η[i])
  end
  return coords
end

function MagneticCoordinateGrid(
  C::Type{MC},
  α::Real,
  β::AbstractVector{VT},
  η::Real,
) where {VT} where {MC<:AbstractMagneticCoordinates}
  T = typeof(C(α, first(β), η))
  coords = StructVector{T}(undef, length(β))
  @inbounds for i = 1:length(β)
    coords[i] = C(α, β[i], η)
  end
  return coords
end

function MagneticCoordinateGrid(
  C::Type{MC},
  α::AbstractVector{VT},
  β::Real,
  η::Real,
) where {VT} where {MC<:AbstractMagneticCoordinates}
  T = typeof(C(first(α), β, η))
  coords = StructVector{T}(undef, length(α))
  @inbounds for i = 1:length(α)
    coords[i] = C(α[i], β, η)
  end
  return coords
end

function MagneticCoordinateGrid(
  C::Type{MC},
  α::AbstractArray{VT1,2},
  β::AbstractArray{VT2,2},
  η::AbstractArray{VT3,2};
  keep_order = false,
  dim_order::Union{Nothing,Vector{Int},Tuple{Int,Int}} = nothing,
) where {VT1} where {VT2} where {VT3} where {MC<:AbstractMagneticCoordinates}
  size(α) == size(β) && size(β) == size(η) ||
    throw(DimensionMismatch("Dimensions of the input arrays must match"))
  T = typeof(C(first(α), first(β), first(η)))
  if !keep_order
    dimperm = isnothing(dim_order) ? SVector{2,Int}(reverse(sortperm([size(β)...]))) : SVector{2,Int}(dim_order)
  else
    dimperm = SVector(1, 2)
  end
  coords = StructArray{T}(undef, size(α)[dimperm])
  for j = 1:size(coords, 2)
    @inbounds for i = 1:size(coords, 1)
      ij_entry = (i, j)[dimperm]
      coords[i, j] = C(α[ij_entry...], β[ij_entry...], η[ij_entry...])
    end
  end
  return coords
end

function MagneticCoordinateGrid(
  C::Type{MC},
  α::Real,
  β::AbstractVector{VT1},
  η::AbstractVector{VT2};
  grid::Bool = true,
  keep_order::Bool = false,
  dim_order::Union{Nothing,Vector{Int}} = nothing,
) where {VT1} where {VT2} where {MC<:AbstractMagneticCoordinates}
  T = typeof(C(first(α), first(β), first(η)))
  if grid
    if !keep_order
      dimperm =
        isnothing(dim_order) ? reverse(sortperm([length(β), length(η)])) :
        dim_order
    else
      dimperm = [1, 2]
    end
    firstvec = dimperm[1] == 2 ? η : β
    secondvec = dimperm[2] == 1 ? β : η
    gridSize = (length(firstvec), length(secondvec))
    coords = StructArray{T}(undef, gridSize)
    for j = 1:length(secondvec)
      @inbounds for i = 1:length(firstvec)
        entry = (α, (firstvec[i], secondvec[j])[dimperm]...)
        coords[i, j] = C(entry...)
      end
    end
    return coords
  else
    length(β) == length(η) || throw(
      DimensionMismatch(
        "Dimensions of the vectors for the angle-like variables must match for non-gridded use",
      ),
    )
    coords = StructVector{T}(undef, length(β))
    @inbounds for i = 1:length(β)
      coords[i] = C(α, β[i], η[i])
    end
    return coords
  end
end

#=
function MagneticCoordinateArray(
  C::Type{MC},
  α::AbstractVector{VT1},
  β::Real,
  η::AbstractVector{VT2};
  grid = true,
  fastArg::Int = 3,
) where {VT1} where {VT2} where {MC<:AbstractMagneticCoordinates}
  T = typeof(C(first(α), first(β), first(η)))
  if grid
    fastvec = fastArg == 3 ? η : α
    slowvec = fastArg == 3 ? α : η
    gridSize = (length(fastvec), length(slowvec))
    coords = Matrix{T}(undef, gridSize)
    for j = 1:length(slowvec)
      @simd for i = 1:length(fastvec)
        entry =
          fastArg == 3 ? (slowvec[j], β, fastvec[i]) :
          (fastvec[i], β, slowvec[j])
        @inbounds coords[i, j] = C(entry...)
      end
    end
    return coords
  else
    length(α) == length(η) || throw(
      DimensionMismatch(
        "Dimensions of the vectors for the angle-like variables must match for non-gridded use",
      ),
    )
    coords = Vector{T}(undef, length(α))
    @inbounds @simd for i = 1:length(α)
      coords[i] = C(α[i], β, η[i])
    end
    return coords
  end
end
=#

function MagneticCoordinateArray(
  C::Type{MC},
  α::AbstractVector{VT1},
  β::AbstractVector{VT2},
  η::Real;
  grid = true,
  fastArg::Int = 2,
) where {VT1} where {VT2} where {MC<:AbstractMagneticCoordinates}
  T = typeof(C(first(α), first(β), first(η)))
  if grid
    fastvec = fastArg == 2 ? β : α
    slowvec = fastArg == 2 ? α : β
    gridSize = (length(fastvec), length(slowvec))
    coords = Matrix{T}(undef, gridSize)
    for j = 1:length(slowvec)
      for i = 1:length(fastvec)
        entry =
          fastArg == 3 ? (slowvec[j], fastec[i], η) :
          (fastvec[i], slowvec[j], η)
        @inbounds coords[i, j] = C(entry...)
      end
    end
    return coords
  else
    length(α) == length(β) || throw(
      DimensionMismatch(
        "Dimensions of the vectors for the angle-like variables must match for non-gridded use",
      ),
    )
    coords = Vector{T}(undef, length(α))
    @inbounds @simd for i = 1:length(α)
      coords[i] = C(α[i], β[i], η)
    end
    return coords
  end
end
