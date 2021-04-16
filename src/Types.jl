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

`R = [Ux Vx Wx;Uy Vy Wy;Uz Vz Wz]`

where `Ux` is the `x`-component of `U`.
# Examples
```julia-repl
julia> using StaticArrays
julia> typeof(@SArray(ones(3,3))) <: BasisVectors{Float64}
true
```
"""
const BasisVectors{T} = SArray{Tuple{3,3},T,2,9} where T

abstract type BasisType end

#Define the singleton types
struct Covariant <: BasisType end
struct Contravariant <: BasisType end

abstract type AbstractMagneticGeometry end;
abstract type MagneticGeometry <: AbstractMagneticGeometry end;

struct AbstractMagneticSurface <: AbstractMagneticGeometry
  eqType::Type
  surfaceLabel::Float64
end

#=
struct MagneticSurface <: MagneticGeometry
  eq::MagneticEquilibrium
  coords::AbstractArray{AbstractMagneticCoordinates}
  MagneticSurface() = new()
  MagneticSurface(e::eqType,c::Union{coordType,AbstractArray{coordType}}) where {eqType <: MagneticEquilibrium, coordType <: AbstractMagneticCoordinates} = new(e,c)
end

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

function Base.getindex(x::MagneticFieldline,I)
  return Base.getindex(Base.getfield(x,:coords),I)
end

function Base.setindex!(x::MagneticFieldline,val::MC,I) where MC <: AbstractMagneticCoordinates
  Base.setindex!(getfield(x,:coords),val,I)
end

function Base.size(x::MagneticFieldline,d::Integer=0)
  return d > 0 ? Base.size(getfield(x,:coords),d) : Base.size(getfield(x,:coords))
end

function Base.length(x::MagneticFieldline)
  return Base.length(getfield(x,:coords))
end

function Base.eltype(x::MagneticFieldline)
  return Base.eltype(getfield(x,:coords))
end

function Base.iterate(x::MagneticFieldline, state=1)
  return Base.iterate(getfield(x,:coords),state)
end

function MagneticCoordinateArray(C::Type{MC},α::Real,β::Real,η::AbstractVector{VT}) where VT where MC <: AbstractMagneticCoordinates
  T = typeof(C(α,β,first(η)))
  coords = Vector{T}(undef,length(η))
  @inbounds @simd for i = 1:length(η)
    coords[i] = C(α,β,η[i])
  end
  return coords
end

function MagneticCoordinateArray(C::Type{MC},α::Real,β::AbstractVector{VT},η::Real) where VT where MC <: AbstractMagneticCoordinates
  T = typeof(C(α,first(β),η))
  coords = Vector{T}(undef,length(β))
  @inbounds @simd for i = 1:length(β)
    coords[i] = C(α,β[i],η)
  end
  return coords
end

function MagneticCoordinateArray(C::Type{MC},α::AbstractVector{VT},β::Real,η::Real) where VT where MC <: AbstractMagneticCoordinates
  T = typeof(C(first(α),β,η))
  coords = Vector{T}(undef,length(α))
  @inbounds @simd for i = 1:length(α)
    coords[i] = C(α[i],β,η)
  end
  return coords
end

function MagneticCoordinateArray(C::Type{MC},α::Real,β::AbstractVector{VT1},η::AbstractVector{VT2};
                             grid=true,fastArg::Int=3) where VT1 where VT2 where MC <: AbstractMagneticCoordinates
  T = typeof(C(first(α),first(β),first(η)))
  if grid
    fastvec = fastArg == 3 ? η : β
    slowvec = fastArg == 3 ? β : η
    gridSize = (length(fastvec),length(slowvec))
    coords = Matrix{T}(undef,gridSize)
    for j = 1:length(slowvec)
      @inbounds @simd for i = 1:length(fastvec)
        entry = fastArg == 3 ? (α,slowvec[j],fastvec[i]) : (α,fastvec[i],slowvec[j])
        coords[i,j] = C(entry...)
      end
    end
    return coords
  else
    length(β) == length(η) || throw(DimensionMismatch("Dimensions of the vectors for the angle-like variables must match for non-gridded use"))
    coords = Vector{T}(undef,length(β))
    @inbounds @simd for i = 1:length(β)
      coords[i] = C(α,β[i],η[i])
    end
    return coords
  end
end

function MagneticCoordinateArray(C::Type{MC},α::AbstractVector{VT1},β::Real,η::AbstractVector{VT2};
                             grid=true,fastArg::Int=3) where VT1 where VT2 where MC <: AbstractMagneticCoordinates
  T = typeof(C(first(α),first(β),first(η)))
  if grid
    fastvec = fastArg == 3 ? η : α
    slowvec = fastArg == 3 ? α : η
    gridSize = (length(fastvec),length(slowvec))
    coords = Matrix{T}(undef,gridSize)
    for j = 1:length(slowvec)
      @simd for i = 1:length(fastvec)
        entry = fastArg == 3 ? (slowvec[j],β,fastvec[i]) : (fastvec[i],β,slowvec[j])
        @inbounds coords[i,j] = C(entry...)
      end
    end
    return coords
  else
    length(α) == length(η) || throw(DimensionMismatch("Dimensions of the vectors for the angle-like variables must match for non-gridded use"))
    coords = Vector{T}(undef,length(α))
    @inbounds @simd for i = 1:length(α)
      coords[i] = C(α[i],β,η[i])
    end
    return coords
  end
end

function MagneticCoordinateArray(C::Type{MC},α::AbstractVector{VT1},β::AbstractVector{VT2},η::Real;
                             grid=true,fastArg::Int=2) where VT1 where VT2 where MC <: AbstractMagneticCoordinates
  T = typeof(C(first(α),first(β),first(η)))
  if grid
    fastvec = fastArg == 2 ? β : α
    slowvec = fastArg == 2 ? α : β
    gridSize = (length(fastvec),length(slowvec))
    coords = Matrix{T}(undef,gridSize)
    for j = 1:length(slowvec)
      @simd for i = 1:length(fastvec)
        entry = fastArg == 3 ? (sj,fi,η) : (fi,sj,η)
        @inbounds coords[i,j] = C(entry...)
      end
    end
    return coords
  else
    length(α) == length(β) || throw(DimensionMismatch("Dimensions of the vectors for the angle-like variables must match for non-gridded use"))
    coords = Vector{T}(undef,length(α))
    @inbounds @simd for i = 1:length(α)
      coords[i] = C(α[i],β[i],η)
    end
    return coords
  end
end

function MagneticCoordinateArray(C::Type{MC},α::AbstractVector{VT1},β::AbstractVector{VT2},
                             η::AbstractVector{VT3};grid=true,fastArg::Int=3) where VT1 where VT2 where VT3 where MC <: AbstractMagneticCoordinates
  T = typeof(C(first(α),first(β),first(η)))
  if grid
    fastvec = fastArg == 3 ? η : β
    slowvec = fastArg == 3 ? β : η
    gridSize = (length(fastvec),length(slowvec),length(α))
    coords = Array{T}(undef,gridSize)
    for k = 1:length(α)
      for j = 1:length(slowvec)
        @simd for i = 1:length(fastvec)
          entry = fastArg == 3 ? (α[k],slowvec[j],fastvec[i]) : (α[k],fastvec[i],slowvec[j])
          @inbounds coords[i,j,k] = C(entry...)
        end
      end
    end
    return coords
  else
    length(β) == length(η) || throw(DimensionMismatch("Dimensions of the vectors for the angle-like variables must match for non-gridded use"))
    length(α) == length(β) || throw(DimensionMismatch("Dimensions for flux (radial) variable must be the same as angle-like variables"))
    coords = Vector{T}(undef,length(α))
    for i = 1:length(α)
      coords[i] = C(α[i],β[i],η[i])
    end
    return coords
  end
end

function MagneticCoordinateArray(C::Type{MC},α::Real,β::AbstractArray{VT1,2},η::AbstractArray{VT2,2}) where VT1 where VT2 where MC <: AbstractMagneticCoordinates
  size(β) == size(η) || throw(DimensionMismatch("Dimensions of the angle-like arguments must match for 2D input arrays"))
  T = typeof(C(α,first(β),first(η)))
  coords = Matrix{T}(undef,size(β))
  coordAxes = axes(β)
  @inbounds @simd for i in CartesianIndices(axes(β))
    coords[i] = C(α,β[i],η[i])
  end
  return coords
end

function MagneticCoordinateArray(C::Type{MC},α::AbstractVector{VT1},β::AbstractArray{VT2,2},η::AbstractArray{VT3,2}) where VT1 where VT2 where VT3 where MC <: AbstractMagneticCoordinates
  size(β) == size(η) || throw(DimensionMismatch("Dimensions of the angle-like arguments must match for 2D input arrays"))
  T = typeof(C(α,first(β),first(η)))
  dimperm = reverse(sortperm([size(β)...]))
  coords = Array{T}(undef,(size(β)[dimperm],length(α)))
  for k = 1:size(coords,3)
    for j = 1:size(coords,2)
      @inbounds @simd for i = 1:size(coords,1)
        ij_entry = (i,j)[dimperm]
        coords[i,j,k] = C(α[k],β[ij_entry...],η[ij_entry...])
      end
    end
  end
  return coords
end

function MagneticCoordinateArray(C::Type{MC},α::AbstractArray{VT1,2},β::AbstractArray{VT2,2},η::AbstractArray{VT3,2}) where VT1 where VT2 where VT3 where MC <: AbstractMagneticCoordinates
  size(α) == size(β) && size(β) == size(η) || throw(DimensionMismatch("Dimensions of the input arrays must match"))
  T = typeof(C(first(α),first(β),first(η)))
  dimperm = reverse(sortperm([size(α)...]))
  coords = Array{T}(undef,size(α)[dimperm])
  for j = 1:size(coords,2)
    @inbounds @simd for i = 1:size(coords,1)
      ij_entry = (i,j)[dimperm]
      coords[i,j,k] = C(α[ih_entry...],β[ij_entry...],η[ij_entry...])
    end
  end
  return coords
end
#=
function MagneticCoordinateArray(C::Type{MC},α::Union{Real,AbstractVector{Real}},
                             β::AbstractVector{Real},η::AbstractVector{Real};
                             grid=true,fastdim=3) where MC <: AbstractMagneticCoordinates
  T = typeof(C(first(α),first(β),first(η)))
  α_scalar = isempty(axes(α))
  if grid
    α_scalar = !α_scalar ? length(α) == 1 ? true : false : false
    fastvec = fastdim == 3 ? η : β
    slowvec = fastdim == 3 ? β : η
    slowdim = fastdim == 3 ? 2 : 3
    gridDims = α_scalar ? (length(fastvec),length(slowvec)) : () : (length(fastvec),length(slowvec),length(α))
    for k = 1:length(α)
      for (j,sj) = enumerate(slowvec)
        @simd for (i,fi) in enumerate(fastvec)
          entry = fastdim == 3 ? (α[k],sj,fi) : (α[k],fi,sj)
          coords[i,j,k] = C(entry...)
        end
      end
    end
  else
    length(β) == length(η) || throw(DimensionMismatch("Dimensions of the vectors for the angle-like variables must match for non-gridded use"))
    !α_scalar && length(α) == 1 ? α_scalar = true :
      length(α) == length(β) || throw(DimensionMismatch("Dimension of the surface label vector and angle-like vectors must match for non-gridded use"))
    coords = Vector{T}(undef,length(β))
    α_vec = α_scalar ? repeat([first(α)],length(β)) : α
    @inbounds @simd for i = 1:length(β)
      coords[i] = C(α_vec[i],β[i],ζ[i])
    end
    return (coords)
  end


=#
#=
function buildDimsAndIndices(dims::Vararg{Tuple{Vararg{Base.OneTo{Int64}}}},
                             firstDim::Int=3,secondDim::Int=2,
                             firstDim::Union{Missing,Int}=missing)
  # This function assumes the first argument is the most important for the memory
  # layout and the last is the least important
  # First do some checks

  scalarDims = isempty.(dims)
  nonScalarDims = length(scalarDims) - length(findall(scalarDims))
  scalarDimAxis = Base.OneTo(1)
  finalDims = Array{Tuple{Vararg{Base.OneTo{Int64}}}}(undef,length(nonScalarDims))
  for d in 1:nonScalarDims
    finalDims[i] = ()
  finalDims = (scalarDims[firstDim] ? scalarDimAxis : first(dims[firstDim]),
               scalarDims[secondDim] ? scalarDimAxis : first(dims[secondDim]),
               scalarDims[thirdDim] ? scalarDimAxis : first(dims[thirdDim]))
  indices = Vector{SVector{Int,3}}(undef,prod(i->i.stop,finalDims))
  entry = Vector{Int}(undef,length(dims))
  for k in finalDims[3]
    for j in finalDims[2]
      @inbounds for i in finalDims[1]
        entry[dimensionPerm] = [i,j,k]
        indices[((k-1)*finalDims[2].stop+(j-1))*finalDims[1].stop+i] = @SVector(entry)
      end
    end
  end
  return finalDims, indices
end

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
  coordType = !isprimitivetype(eltypecoords)) ? Symbol(eltypecoords)) : :Cartesian
  label = surfaceLabel
  coordinates = coords
  return MagneticSurface(eq,coordType,label,coordinates)
end

function MagneticFieldline(surfaceLabel,fieldlineLabel,eq::MagneticEquilibrium,coords::AbstractVector)
  eqType = eq_typeof(eq)
  coordType = !isprimitivetype(eltypecoords)) ? Symbol(eltypecoords)) : :Cartesian
  label = (surfaceLabel,fieldlineLabel)
  coordinates = coords
  return MagneticFieldline(eq,coordType,label,coordinates)
end
=#
