import Base.length, Base.+, Base.-, Base.*, Base./, Base.abs
import LinearAlgebra

function length(v::Vector3D{T}) where T <: AbstractFloat
  return getfield(v,:length)
end

function cross(u::Vector3D{T},v::Vector3D{T}) where  T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors do not have same number of elements"
  tx = Threads.@spawn getfield(u,:y) .* getfield(v,:z) .- getfield(u,:z) .* getfield(v,:y) 
  ty = Threads.@spawn getfield(u,:z) .* getfield(v,:x) .- getfield(u,:x) .* getfield(v,:z) 
  tz = Threads.@spawn getfield(u,:x) .* getfield(v,:y) .- getfield(u,:y) .* getfield(v,:x)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function dot(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors do not have same number of elements"

  txx = Threads.@spawn getfield(u,:x) .* getfield(v,:x)
  tyy = Threads.@spawn getfield(u,:y) .* getfield(v,:y)
  tzz = Threads.@spawn getfield(u,:z) .* getfield(v,:z)
  ni = getfield(u,:length)
  res = Vector{T}(undef,ni)
  xx = fetch(txx)
  yy = fetch(tyy)
  zz = fetch(tzz)
  @inbounds @simd for i = 1:ni
    res[i] = xx[i] + yy[i] + zz[i]
  end
  return res
end

# Element wise addition of two Vector3D instances
function +(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn getfield(u,:x) .+ getfield(v,:x)
  ty = Threads.@spawn getfield(u,:y) .+ getfield(v,:y)
  tz = Threads.@spawn getfield(u,:z) .+ getfield(v,:z)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

# Element wise subtraction of two Vector3D instances
function -(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn getfield(u,:x) .- getfield(v,:x)
  ty = Threads.@spawn getfield(u,:y) .- getfield(v,:y)
  tz = Threads.@spawn getfield(u,:z) .- getfield(v,:z)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

# Element wise multiplication of two Vector3D instances
function *(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn getfield(u,:x) .* getfield(v,:x)
  ty = Threads.@spawn getfield(u,:y) .* getfield(v,:y)
  tz = Threads.@spawn getfield(u,:z) .* getfield(v,:z)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))

end

# Element wise multiplication of each component by a vector
function *(u::Vector3D{T},v::Vector{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == Base.length(v) "Vectors have incompatible length"
  tx = Threads.@spawn getfield(u,:x) .* v
  ty = Threads.@spawn getfield(u,:y) .* v
  tz = Threads.@spawn getfield(u,:z) .* v
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function *(v::Vector{T},u::Vector3D{T}) where T <: AbstractFloat
  return u * v
end

# Scalar multiplication
function *(u::Vector3D{T},a::T) where T <: AbstractFloat
  return Vector3D{T}(getfield(u,:x)*a,getfield(u,:y)*a,getfield(u,:z)*a)
end

# Element wise division of two Vector3D instances
function /(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn getfield(u,:x) ./ getfield(v,:x)
  ty = Threads.@spawn getfield(u,:y) ./ getfield(v,:y)
  tz = Threads.@spawn getfield(u,:z) ./ getfield(v,:z)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

# Element wise division of each component by a vector
function /(u::Vector3D{T},v::Vector{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == Base.length(v) "Vectors have incompatible length"
  tx = Threads.@spawn getfield(u,:x) ./ v
  ty = Threads.@spawn getfield(u,:y) ./ v
  tz = Threads.@spawn getfield(u,:z) ./ v
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function /(v::Vector{T},u::Vector3D{T}) where T <: AbstractFloat
  return u / v
end

# Scalar multiplication
function /(u::Vector3D{T},a::T) where T <: AbstractFloat
  return Vector3D{T}(getfield(u,:x)/a,getfield(u,:y)/a,getfield(u,:z)/a)
end

function abs(u::Vector3D{T}) where T <: AbstractFloat
  return sqrt.(dot(u,u))
end


# Vector3DSpline operations
function (u::Vector3DSpline{T})(p::PT) where T <: AbstractFloat where PT <: AbstractFloat
  return getfield(u,:xSpline)(p), getfield(u,:ySpline)(p), getfield(u,:zSpline)(p)
end

