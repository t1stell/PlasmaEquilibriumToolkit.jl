import Base.length, Base.+, Base.-, Base.*, Base./, Base.abs

function length(v::Vector3D{T}) where T <: AbstractFloat
  return getfield(v,:length)
end

function cross(u::Vector3D{T},v::Vector3D{T}) where  T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors do not have same number of elements"
  tx = Threads.@spawn crossComponent(getfield(u,:y),getfield(u,:z),getfield(v,:y),getfield(v,:z)) 
  ty = Threads.@spawn crossComponent(getfield(u,:z),getfield(u,:x),getfield(v,:z),getfield(v,:x)) 
  tz = Threads.@spawn crossComponent(getfield(u,:x),getfield(u,:y),getfield(v,:x),getfield(v,:y)) 
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function crossComponent(a::Vector{T},b::Vector{T},c::Vector{T},d::Vector{T}) where T <: AbstractFloat
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]*d[i] - c[i]*b[i]
  end
  return res
end


function dot(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors do not have same number of elements"

  txx = Threads.@spawn dotComponent(getfield(u,:x),getfield(v,:x))
  tyy = Threads.@spawn dotComponent(getfield(u,:y),getfield(v,:y))
  tzz = Threads.@spawn dotComponent(getfield(u,:z),getfield(v,:z))
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

function dotComponent(a::Vector{T},b::Vector{T}) where T <: AbstractFloat
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]*b[i]
  end
  return res
end

# Element wise addition of two Vector3D instances
function +(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn loopAdd(getfield(u,:x),getfield(v,:x))
  ty = Threads.@spawn loopAdd(getfield(u,:y),getfield(v,:y))
  tz = Threads.@spawn loopAdd(getfield(u,:z),getfield(v,:z))
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function loopAdd(a::Vector{T},b::Vector{T}) where T <: AbstractFloat
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]+b[i]
  end
  return res
end

# Element wise subtraction of two Vector3D instances
function -(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn loopSub(getfield(u,:x),getfield(v,:x))
  ty = Threads.@spawn loopSub(getfield(u,:y),getfield(v,:y))
  tz = Threads.@spawn loopSub(getfield(u,:z),getfield(v,:z))
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function loopSub(a::Vector{T},b::Vector{T}) where T <: AbstractFloat
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]-b[i]
  end
  return res
end

# Element wise multiplication of two Vector3D instances
function *(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn loopMult(getfield(u,:x),getfield(v,:x))
  ty = Threads.@spawn loopMult(getfield(u,:y),getfield(v,:y))
  tz = Threads.@spawn loopMult(getfield(u,:z),getfield(v,:z))
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))

end

# Element wise multiplication of each component by a vector
function *(u::Vector3D{T},v::Vector{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == Base.length(v) "Vectors have incompatible length"
  tx = Threads.@spawn loopMult(getfield(u,:x),v)
  ty = Threads.@spawn loopMult(getfield(u,:y),v)
  tz = Threads.@spawn loopMult(getfield(u,:z),v)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function *(v::Vector{T},u::Vector3D{T}) where T <: AbstractFloat
  return u * v
end

# Scalar multiplication
function *(u::Vector3D{T},a::T) where T <: AbstractFloat
  return Vector3D{T}(getfield(u,:x)*a,getfield(u,:y)*a,getfield(u,:z)*a)
end

function loopMult(a::Vector{T},b::Vector{T}) where T <: AbstractFloat
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]*b[i]
  end
  return res
end

# Element wise division of two Vector3D instances
function /(u::Vector3D{T},v::Vector3D{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == getfield(v,:length) "Vectors have incompatible length"
  tx = Threads.@spawn loopDiv(getfield(u,:x),getfield(v,:x))
  ty = Threads.@spawn loopDiv(getfield(u,:y),getfield(v,:y))
  tz = Threads.@spawn loopDiv(getfield(u,:z),getfield(v,:z))
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

# Element wise division of each component by a vector
function /(u::Vector3D{T},v::Vector{T}) where T <: AbstractFloat
  @assert getfield(u,:length) == Base.length(v) "Vectors have incompatible length"
  tx = Threads.@spawn loopDiv(getfield(u,:x),v)
  ty = Threads.@spawn loopDiv(getfield(u,:y),v)
  tz = Threads.@spawn loopDiv(getfield(u,:z),v)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz))
end

function /(v::Vector{T},u::Vector3D{T}) where T <: AbstractFloat
  return u / v
end

# Scalar multiplication
function /(u::Vector3D{T},a::T) where T <: AbstractFloat
  return Vector3D{T}(getfield(u,:x)/a,getfield(u,:y)/a,getfield(u,:z)/a)
end

function loopDiv(a::Vector{T},b::Vector{T}) where T <: AbstractFloat
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]/b[i]
  end
  return res
end

function abs(u::Vector3D{T}) where T <: AbstractFloat
  return Base.abs.(getfield(u,:x)) .+ Base.abs.(getfield(u,:y)) .+ Base.abs.(getfield(u,:z))
end


# Vector3DSpline operations
function (u::Vector3DSpline{T})(p::PT) where T <: AbstractFloat where PT <: AbstractFloat
  return getfield(u,:xSpline)(p), getfield(u,:ySpline)(p), getfield(u,:zSpline)(p)
end

