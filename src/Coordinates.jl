struct PestCoordinates{D,T,N} <: AbstractCoordinateField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
  ePsi::VectorField{D,T,N}
  eAlpha::VectorField{D,T,N}
  eZeta::VectorField{D,T,N}
  gradPsi::VectorField{D,T,N}
  gradAlpha::VectorField{D,T,N}
  gradZeta::VectorField{D,T,N}
end

struct SThetaZetaCoordinates{D,T,N} <: AbstractCoordinateField{D,T,N}
  data::AbstractArray{NTuple{D,T},N}
  ePsi::VectorField{D,T,N}
  eTheta::VectorField{D,T,N}
  eZeta::VectorField{D,T,N}
  gradPsi::VectorField{D,T,N}
  gradTheta::VectorField{D,T,N}
  gradZeta::VectorField{D,T,N}
end

function eX1(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,2)
end

function eX1(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,2),index)
end

function eX2(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,3)
end

function eX2(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,3),index)
end

function eX3(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,4)
end

function eX3(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,4),index)
end

function gradX1(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,5)
end

function gradX1(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,5),index)
end

function gradX2(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,6)
end

function gradX2(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,6),index)
end

function gradX3(C::AbstractCoordinateField{D,T,N}) where {D,T,N}
  return getfield(C,7)
end

function gradX3(C::AbstractCoordinateField{D,T,N},index::I) where {D,T,N} where I <: Integer
  return component(getfield(C,7),index)
end

