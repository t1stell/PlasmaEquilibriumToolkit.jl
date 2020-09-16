import Base.length, Base.+, Base.-, Base.*, Base./

function length(v::Vector3D{T}) where T <: Real
  return v.length
end

function basisType(v::Vector3D{T}) where T <: Real
  return v.basisType
end

function crossVec(v::Vector3D{T},u::Vector3D{T},jacobian::Vector{T}) where T <: Real
  @assert u.basisType === v.basisType "Vectors do not have same basis"
  @assert u.length == v.length "Vectors do not have same number of elements"

  basisType = v.basisType == :co ? :contra : :co
  jac = v.basisType == :co ? 1.0./jacobian : jacobian
  x = Vector{T}(undef,v.length)
  y = Vector{T}(undef,v.length)
  z = Vector{T}(undef,v.length)
  tx = Threads.@spawn crossVecComponent(u.y,u.z,v.y,v.z,jac) 
  ty = Threads.@spawn crossVecComponent(u.z,u.x,v.z,v.x,jac) 
  tz = Threads.@spawn crossVecComponent(u.x,u.y,v.x,v.y,jac) 
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tx),basisType)
end

function crossVecComponent(a::Vector{T},b::Vector{T},c::Vector{T},d::Vector{T},jacobian::Vector{T}) where T <: Real
  return jacobian .* (a .* d .- c .* b)
end

function crossLoop(v::Vector3D{T},u::Vector3D{T},jacobian::Vector{T}) where T <: Real

  @assert u.basisType === v.basisType "Vectors do not have same basis"
  @assert u.length == v.length "Vectors do not have same number of elements"

  basisType = v.basisType == :co ? :contra : :co
  jac = v.basisType == :co ? 1.0./jacobian : jacobian
  tx = Threads.@spawn crossLoopComponent(u.y,u.z,v.y,v.z,jac) 
  ty = Threads.@spawn crossLoopComponent(u.z,u.x,v.z,v.x,jac) 
  tz = Threads.@spawn crossLoopComponent(u.x,u.y,v.x,v.y,jac) 
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz),basisType)
end

function crossLoopComponent(a::Vector{T},b::Vector{T},c::Vector{T},d::Vector{T},jacobian::Vector{T}) where T <: Real
  ni = Base.length(jacobian)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = jacobian[i]*(a[i]*d[i] - c[i]*b[i])
  end
  return res
end


function dot(u::Vector3D{T},v::Vector3D{T},jacobian::Vector{T}) where T <: Real
  return u.x .* v.x .+ u.y .* v.y .+ u.z .* v.z
end

# Element wise addition of two Vector3D instances
function +(u::Vector3D{T},v::Vector3D{T}) where T <: Real
  @assert getfield(u,:basisType) === getfield(v,:basisType) "Cannot add vectors with different basis types"
  return Vector3D{T}(getfield(u,:x).+getfield(v,:x),
                     getfield(u,:y).+getfield(v,:y),
                     getfield(u,:z).+getfield(v,:z),getfield(u,:basisType))
end

# Element wise subtraction of two Vector3D instances
function -(u::Vector3D{T},v::Vector3D{T}) where T <: Real
  @assert getfield(u,:basisType) === getfield(v,:basisType) "Cannot add vectors with different basis types"
  return Vector3D{T}(getfield(u,:x).-getfield(v,:x),
                     getfield(u,:y).-getfield(v,:y),
                     getfield(u,:z).-getfield(v,:z),getfield(u,:basisType))
end

# Element wise multiplication of two Vector3D instances
function *(u::Vector3D{T},v::Vector3D{T}) where T <: Real
  @assert getfield(u,:basisType) === getfield(v,:basisType) "Cannot add vectors with different basis types"
  tx = Threads.@spawn loopMult(getfield(u,:x),getfield(v,:x))
  ty = Threads.@spawn loopMult(getfield(u,:y),getfield(v,:y))
  tz = Threads.@spawn loopMult(getfield(u,:z),getfield(v,:z))
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz),getfield(u,:basisType))

end

# Element wise multiplication of each component by a vector
function *(u::Vector3D{T},v::Vector{T}) where T <: Real
  tx = Threads.@spawn loopMult(getfield(u,:x),v)
  ty = Threads.@spawn loopMult(getfield(u,:y),v)
  tz = Threads.@spawn loopMult(getfield(u,:z),v)
  return Vector3D{T}(fetch(tx),fetch(ty),fetch(tz),getfield(u,:basisType))
end

# Scalar multiplication
function *(u::Vector3D{T},a::T) where T <: Real
  return Vector3D{T}(getfield(u,:x)*a,getfield(u,:y)*a,getfield(u,:z)*a,getfield(u,:basisType))
end

function loopMult(a::Vector{T},b::Vector{T}) where T <: Real
  ni = Base.length(a)
  res = Vector{T}(undef,ni)
  @inbounds @simd for i = 1:ni
    res[i] = a[i]*b[i]
  end
  return res
end

# Element wise division of two Vector3D instances
function /(u::Vector3D{T},v::Vector3D{T}) where T <: Real
  @assert getfield(u,:basisType) === getfield(v,:basisType) "Cannot add vectors with different basis types"
  return Vector3D{T}(getfield(u,:x)./getfield(v,:x),
                     getfield(u,:y)./getfield(v,:y),
                     getfield(u,:z)./getfield(v,:z),getfield(u,:basisType))

end

# Element wise division of each component by a vector
function /(u::Vector3D{T},v::Vector{T}) where T <: Real
  return Vector3D{T}(getfield(u,:x)./v,getfield(u,:y)./v,getfield(u,:z)./v,getfield(u,:basisType))
end

