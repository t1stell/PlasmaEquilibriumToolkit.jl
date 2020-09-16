function size(s::Surface3D{T}) where T <: Real
  return s.size
end

function cross(v::Surface3D{T},u::Surface3D{T},jacobian::Array{T,2}) where T <: Real
  if (v.basisType != u.basisType) 
    throw(TypeError("Vectors do not have same basis vectors!"))
  end
  w = Surface3D{T}(v.size)

  w.basisType = v.basisType == :co ? :contra : :co
  jac .= v.basisType == :co ? 1.0./jacobian : jacobian
  w.x .= jac .* (u.y .* v.z .- v.y .* u.z)
  w.y .= jac .* (u.z .* v.x .- v.z .* u.x)
  w.z .= jac .* (u.x .* v.y .- v.x .* u.y) 
  return w
end

function dot(u::Surface3D{T},v::Surface3D{T},jacobian::Array{T,2}) where T <: Real
  return u.x .* v.x .+ u.y .* v.y .+ u.z .* v.z
end
