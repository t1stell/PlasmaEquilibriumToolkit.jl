function inverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.cos*cos(v.m*θ-v.n*ζ) + v.sin*sin(v.m*θ-v.n*ζ)
end

function dsInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.dcos_ds*cos(v.m*θ-v.n*ζ) + v.dsin_ds*sin(v.m*θ-v.n*ζ)
end

function dθInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.m*(v.sin*cos(v.m*θ-v.n*ζ) - v.cos*sin(v.m*θ-v.n*ζ))
end

function dζInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.n*(v.cos*sin(v.m*θ-v.n*ζ) - v.sin*cos(v.m*θ-v.n*ζ))
end

#placeholder functions for now
function dsdθInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.m*(v.dsin_ds*cos(v.m*θ-v.n*ζ) - v.dcos_ds*sin(v.m*θ-v.n*ζ))
end

function dsdζInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.n*(v.dcos_ds*sin(v.m*θ-v.n*ζ) - v.dsin_ds*cos(v.m*θ-v.n*ζ))
end

function dθdθInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.m*v.m*(-1)*inverseKernel(θ, ζ, v)
end

function dθdζInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.m*v.n*inverseKernel(θ, ζ, v)
end

function dζdζInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  return v.n*v.n*(-1)*inverseKernel(θ, ζ, v)
end

function dsdsInverseKernel(θ::A,ζ::A,v::SurfaceFourierData{T}) where {A, T}
  println("ds^2 derivative not implemented")
  return 0
end

function cosineKernel(x::AbstractMagneticCoordinates,m::Int,n::Int,A::T) where T
  return A*cos(m*getfield(x,:2)-n*getfield(x,:3))
end

function sineKernel(x::AbstractMagneticCoordinates,m::Int,n::Int,A::T) where T
  return A*sin(m*getfield(x,:2)-n*getfield(x,:3))
end

function cosineTransform(m::Int,n::Int,x::AbstractArray{C},A::AbstractArray) where C <: AbstractMagneticCoordinates
  res = Array{Float64, ndims(x)}(undef, size(x))
  map!((y, b)->cosineKernel(y, m, n, b), res, x, A)
  return sum(res)
end

function sineTransform(m::Int,n::Int,x::AbstractArray{C},A::AbstractArray) where C <: AbstractMagneticCoordinates
  res = Array{Float64, ndims(x)}(undef, size(x))
  map!((y, b)->sineKernel(y, m, n, b), res, x, A)
  return sum(res)
end

"""
    inverseTransform(x::AbstractMagneticCoordinates,data::SurfaceFourierData{T};deriv::Symbol=:none)

Compute the inverse Fourier transform of VMEC spectral data with coefficients
defined in `data` at the points defined by the AbstractMagneticCoordinates `x`.
The inverse transform of the derivative of `data` can be specified as `:none`,`:ds`, `:dθ`, or `dζ`.

# Examples
"""
function inverseTransform(x::C,
                          data::AbstractVector{SurfaceFourierData{T}};
                          deriv::Symbol=:none,
                         ) where {T, C <: AbstractMagneticCoordinates}
  res = Array{T,1}(undef, length(data))
  θ = getfield(x, :2)
  ζ = getfield(x, :3)
  kernel = (deriv === :none ? inverseKernel : 
            deriv === :ds ? dsInverseKernel : 
            deriv === :dθ ? dθInverseKernel : 
            deriv === :dζ ? dζInverseKernel :
            deriv === :dθdθ ? dθdθInverseKernel :
            deriv === :dζdζ ? dζdζInverseKernel :
            deriv === :dsdθ ? dsdθInverseKernel :
            deriv === :dsdζ ? dsdζInverseKernel : 
            deriv === :dθdζ ? dθdζInverseKernel : dsdsInverseKernel)
  map!(data_mn->kernel(θ, ζ, data_mn), res,data)
  return sum(res)
end

function inverseTransform(x::AbstractArray{C},
                          data::AbstractVector{SurfaceFourierData{T}};
                          deriv::Symbol=:none,
                         ) where {T, C <: AbstractMagneticCoordinates}
  res = Array{T}(undef, size(x))
  inverseTransform!(res, x, data, deriv=deriv)
  return res
end

function inverseTransform!(res::AbstractArray{T},
                           x::AbstractArray{C},
                           data::AbstractVector{SurfaceFourierData{T}};
                           deriv::Symbol=:none,
                          ) where {T, C <: AbstractMagneticCoordinates}
  size(res) == size(x) || throw(DimensionMismatch("Incompatible coordinate and results arrays in inverseTransform"))
  @batch for i in eachindex(x, res)
    res[i] = inverseTransform(x[i], data, deriv=deriv)
  end
end
