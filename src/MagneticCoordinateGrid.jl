function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractArray{VT1,3},
                                 β::AbstractArray{VT2,3},
                                 η::AbstractArray{VT3,3};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  size(α) == size(β) && size(β) == size(η) || throw(DimensionMismatch("Incommensurate grid sizes for MagneticCoordinateGrid"))
  T = typeof(C(first(α),first(β),first(η)))
  return StructArray{T}((α,β,η))
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::Real,
                                 β::AbstractMatrix{VT1},
                                 η::AbstractMatrix{VT2};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
  return MagneticCoordinateGrid(C,fill(α,size(β)),β,η)
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractMatrix{VT1},
                                 β::Real,
                                 η::AbstractMatrix{VT2};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
  return MagneticCoordinateGrid(C,α,fill(β,size(η)),η)
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractMatrix{VT1},
                                 β::AbstractMatrix{VT2},
                                 η::Real;
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
  return MagneticCoordinateGrid(C,α,β,fill(η,size(β)))
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractMatrix{VT1},
                                 β::AbstractMatrix{VT2},
                                 η::AbstractMatrix{VT3};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  size(α) == size(β) && size(β) == size(η) || throw(DimensionMismatch("Incommensurate grid sizes for MagneticCoordinateGrid"))
  T = typeof(C(first(α),first(β),first(η)))
  return StructArray{T}((α,β,η))
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractVector{VT1},
                                 β::AbstractMatrix{VT2},
                                 η::AbstractMatrix{VT3};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  size(β) == size(η) || throw(DimensionMismatch("Incommensurate grid sizes for $β and $η"))
  T = typeof(C(first(α),first(β),first(η)))
  fullSize = (size(η)...,length(α))
  α_grid = reshape(repeat(α,inner=length(β)),fullSize)
  β_grid = repeat(β,outer=(1,1,length(α)))
  η_grid = repeat(η,outer=(1,1,length(α)))
  return StructArray{T}((α_grid,β_grid,η_grid))
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractMatrix{VT1},
                                 β::AbstractVector{VT2},
                                 η::AbstractMatrix{VT3};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  size(α) == size(η) || throw(DimensionMismatch("Incommensurate grid sizes for $α and $η"))
  T = typeof(C(first(α),first(β),first(η)))
  fullSize = (size(η,1),length(β),size(η,2))
  α_grid = reshape(repeat(α,inner=(1,length(β))),fullSize)
  β_grid = reshape(repeat(β,inner=size(η,1),outer=size(η,2)),fullSize)
  η_grid = reshape(repeat(η,outer=(length(β),1)),fullSize)
  return StructArray{T}((α_grid,β_grid,η_grid))
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractMatrix{VT1},
                                 β::AbstractMatrix{VT2},
                                 η::AbstractVector{VT3};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  size(α) == size(β) || throw(DimensionMismatch("Incommensurate grid sizes for $α and $β"))
  T = typeof(C(first(α),first(β),first(η)))
  fullSize = (length(η),size(α)...)
  α_grid = reshape(repeat(α,inner=(1,length(η))),fullSize)
  β_grid = reshape(repeat(β,inner=(length(η),1)),fullSize)
  η_grid = reshape(repeat(η,outer=(1,prod(size(β)))),fullSize)
  return StructArray{T}((α_grid,β_grid,η_grid))
end

function MagneticCoordinateGrid(C::Type{CT},
                                 α::AbstractVector{VT1},
                                 β::AbstractVector{VT2},
                                 η::AbstractVector{VT3};
                                 grid=true,
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  if !grid
    return MagneticCoordinateCurve(C,α,β,η)
  else
    T = typeof(C(first(α),first(β),first(η)))
    fullSize = (length(η),length(β),length(α))
    α_grid = reshape(repeat(α,inner=length(β)*length(η)),fullSize)
    β_grid = reshape(repeat(β,inner=length(η),outer=length(α)),fullSize)
    η_grid = reshape(repeat(η,outer=length(β)*length(α)),fullSize)
    return StructArray{T}((α_grid,β_grid,η_grid))
  end
end

function MagneticCoordinateGrid(C::Type{CT},
                                α::Real,
                                β::AbstractVector{VT1},
                                η::AbstractVector{VT2};
                                grid=true,
                               ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
  if !grid
    return MagneticCoordinateCurve(C,α,β,η)
  else
    T = typeof(C(first(α),first(β),first(η)))
    fullSize = (length(η),length(β))
    α_grid = fill(α,fullSize)
    β_grid = reshape(repeat(β,inner=length(η)),fullSize)
    η_grid = reshape(repeat(η,outer=length(β)),fullSize)
    return StructArray{T}((α_grid,β_grid,η_grid))
  end
end

function MagneticCoordinateGrid(C::Type{CT},
                                α::AbstractVector{VT1},
                                β::Real,
                                η::AbstractVector{VT2};
                                grid=true,
                               ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
  if !grid
    return MagneticCoordinateCurve(C,α,β,η)
  else
    T = typeof(C(first(α),first(β),first(η)))
    fullSize = (length(η),length(α))
    α_grid = reshape(repeat(α,inner=length(η)),fullSize)
    β_grid = fill(β,fullSize)
    η_grid = reshape(repeat(η,outer=length(α)),fullSize)
    return StructArray{T}((α_grid,β_grid,η_grid))
  end
end

function MagneticCoordinateGrid(C::Type{CT},
                                α::AbstractVector{VT1},
                                β::AbstractVector{VT2},
                                η::Real;
                                grid=true,
                               ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
  if !grid
    return MagneticCoordinateCurve(C,α,β,η)
  else
    T = typeof(C(first(α),first(β),first(η)))
    fullSize = (length(β),length(α))
    α_grid = reshape(repeat(α,inner=length(β)),fullSize)
    β_grid = reshape(repeat(β,outer=length(α)),fullSize)
    η_grid = fill(η,fullSize)
    return StructArray{T}((α_grid,β_grid,η_grid))
  end
end

function MagneticCoordinateCurve(C::Type{CT},
                                 α::Real,
                                 β::Real,
                                 η::AbstractVector{VT1};
                                ) where {CT <: AbstractMagneticCoordinates, VT1}
  MagneticCoordinateCurve(C,fill(α,size(η)),fill(β,size(η)),η)
end

function MagneticCoordinateCurve(C::Type{CT},
                                 α::Real,
                                 β::AbstractVector{VT1},
                                 η::Real;
                                ) where {CT <: AbstractMagneticCoordinates, VT1}
  MagneticCoordinateCurve(C,fill(α,size(β)),β,fill(η,size(β)))
end

function MagneticCoordinateCurve(C::Type{CT},
                                 α::AbstractVector{VT1},
                                 β::Real,
                                 η::Real;
                                ) where {CT <: AbstractMagneticCoordinates, VT1}
  MagneticCoordinateCurve(C,α,fill(β,size(α)),fill(η,size(α)))
end

function MagneticCoordinateCurve(C::Type{CT},
                                 α::Real,
                                 β::AbstractVector{VT1},
                                 η::AbstractVector{VT2};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
    MagneticCoordinateCurve(C,fill(α,size(β)),β,η)
end

function MagneticCoordinateCurve(C::Type{CT},
                                 α::AbstractVector{VT1},
                                 β::Real,
                                 η::AbstractVector{VT2};
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
    MagneticCoordinateCurve(C,α,fill(β,size(η)),η)
end

function MagneticCoordinateCurve(C::Type{CT},
                                 α::AbstractVector{VT1},
                                 β::AbstractVector{VT2},
                                 η::Real;
                                ) where {CT <: AbstractMagneticCoordinates, VT1, VT2}
    MagneticCoordinateCurve(C,α,β,fill(η,size(β)))
end

function MagneticCoordinateCurve(C::Type{CT},
                                α::AbstractVector{VT1},
                                β::AbstractVector{VT2},
                                η::AbstractVector{VT3};
                               ) where {CT <: AbstractMagneticCoordinates, VT1, VT2, VT3}
  size(α) == size(β) && size(β) == size(η) || throw(DimensionMismatch("Incommensurate grid sizes to create MagneticCoordinateCurve"))
  T = typeof(C(first(α),first(β),first(η)))
  StructVector{T}((α,β,η))
end
