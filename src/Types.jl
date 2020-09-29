
struct MetricTensor{D,T,N} <: AbstractDataField{T,N}
  data::AbstractArray{NTuple{D,T},N}
end

#Define the MetricTensor constructors for different instances
function MetricTensor(dataFields::Vararg{AbstractArray{T,N},D}) where {D,T,N}
  @assert D >= 3 "Metric requires at least three components!"
  for i = 1:D-1
    @assert size(dataFields[i]) == size(dataFields[i+1]) "Incompatible array sizes"
  end
  data = Array{NTuple{D,T},N}(undef,size(dataFields[1]))
  map!(inds::Vararg{T,D}->inds,data,dataFields...)
  return MetricTensor{D,T,N}(data)
end

function MetricTensor(u::VectorField{D,T,N}) where {D,T,N}
  data = Array{NTuple{sum(1:D),T},N}(undef,size(u))
  map!(computeMetric,data,getfield(u,:data))
  return MetricTensor(data)
end

function IdentityMetric(D::I,T::Type,size::SizeT) where SizeT <: Union{Int,Tuple} where I <: Integer
  entries = zeros(T,sum(1:D))
  for i = 1:D
    entries[sum(1:i)] = one(T)
  end
  data = fill(tuple(entries...),size)
  return MetricTensor(data)
end

function computeMetric(t::NTuple{D,T}) where {D,T}
  entries = zeros(T,sum(1:D))
  for i = 1:D 
    for j = sum(1:i)-i+1:sum(1:i)  
      entries[j] = t[i]*t[j-sum(1:i-1)]
    end
  end
  return tuple(entries...)
end

# Define the constructors
"""
  Fieldline{T}

  Defines a struct that contains all the relevant information for a magnetic field line, including
  the vector B, the covariant and contravariant components, the metric tensor and the jacobian
  along fieldline
"""
mutable struct Fieldline{T<:AbstractFloat}
  x::Vector{T}
  y::Vector{T}
  z::Vector{T}
  B::VectorField{3,T,1}
  coBasis_x::VectorField{3,T,1}
  coBasis_y::VectorField{3,T,1}
  coBasis_z::VectorField{3,T,1}
  contraBasis_x::VectorField{3,T,1}
  contraBasis_y::VectorField{3,T,1}
  contraBasis_z::VectorField{3,T,1}
  metric::MetricTensor{T,1}
  jacobian::Vector{T}
  curvature::VectorField{T}
  length::Int
  rotationalTransform::Float64
  Fieldline{T}() where T <: Real = new()
end
