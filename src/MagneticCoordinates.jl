using CoordinateTransformations

abstract type MagneticEquilibrium end;
abstract type MagneticCoordinates end;

"""
    FluxCoordinates{T,A}(s::T,θ::A,ζ::A)

Coordinates on a magnetic flux surface, where `s` is the surface label and
`θ` and `ζ` are angle-like variables
"""
struct FluxCoordinates{T <: Real,A <: Real} <: MagneticCoordinates
  s::T
  θ::A
  ζ::A
  FluxCoordinates{T,A}(s::T,θ::A,ζ::A) where {T,A} = new(s,θ,ζ)
end

function FluxCoordinates(s,θ,ζ)
  s2, θ2, ζ2 = promote(s,θ,ζ)
  return FluxCoordinates{typeof(s2),typeof(θ2)}(s2,θ2,ζ2)
end

function FluxCoordinates(s,θ,ζ::AbstractVector)
  coords = Vector{FluxCoordinates}(undef,length(ζ))
  map!(z->FluxCoordinates(s,θ,z),coords,ζ)
  return coords
end

function FluxCoordinates(s,θ::AbstractVector,ζ::AbstractVector)
  coords = Matrix{FluxCoordinates}(undef,length(ζ),length(θ))
  for t = 1:length(θ)
    for z = 1:length(ζ)
      @inbounds coords[z,t] = FluxCoordinates(s,θ[t],ζ[z])
    end
  end
  return coords
end

function FluxCoordinates(s,θ::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(θ) == size(ζ) "Incompatible sizes"
  coords = Matrix{FluxCoordinates}(undef,size(θ))
  map!((t,z)->FluxCoordinates(s,t,z),coords,θ,ζ)
  return coords
end

function FluxCoordinates(s::AbstractVector,θ::AbstractVector,ζ::AbstractVector)
  coords = Array{FluxCoordinates,3}(undef,length(ζ),length(θ),length(s))
  for k = 1:length(s)
    for t = 1:length(θ)
      for z = 1:length(ζ)
        @inbounds coords[z,t,k] = FluxCoordinates(s[k],θ[t],ζ[z])
      end
    end
  end
  return coords
end

function FluxCoordinates(s::AbstractVector,θ::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(θ) == size(ζ) "Incompatible sizes"
  coords = Array{FluxCoordinates,3}(undef,(size(θ)...,length(s)))
  for k = 1:length(s)
    for j = 1:size(θ,2)
      for i = 1:size(θ,1)
        coords[i,j,k] = FluxCoordinates(s[k],θ[i,j],ζ[i,j])
      end
    end
  end
  return coords
end

function FluxCoordinates(s::AbstractArray{T,3},θ::AbstractArray{T,3},ζ::AbstractArray{T,3}) where T
  @assert size(s) == size(θ) == size(ζ) "Incompatible sizes"
  coords = Array{FluxCoordinates,3}(undef,size(s))
  map!((s,t,z)->FluxCoordinates(s,t,z),coords,s,θ,ζ)
  return coords
end

Base.show(io::IO, x::FluxCoordinates) = print(io, "FluxCoordinates(s=$(x.s), θ=$(x.θ), ζ=$(x.ζ))")
Base.isapprox(x1::FluxCoordinates, x2::FluxCoordinates; kwargs...) = isapprox(x1.s,x2.s;kwargs...) && isapprox(x1.θ,x2.θ;kwargs...) && isapprox(x1.ζ,x2.ζ;kwargs...)

"""
    PestCoordinates{T,A}(s::T,θ::A,ζ::A)

Coordinates on a magnetic flux surface, where `s` is the surface label and
`θ` and `ζ` are angle-like variables
"""
struct PestCoordinates{T <: Real,A <: Real} <: MagneticCoordinates
  ψ::T
  α::A
  ζ::A
  PestCoordinates{T,A}(ψ::T,α::A,ζ::A) where {T,A} = new(ψ,α,ζ)
end

function PestCoordinates(ψ,α,ζ)
  ψ2, α2, ζ2 = promote(ψ,α,ζ)
  return PestCoordinates{typeof(ψ2),typeof(α2)}(ψ2,α2,ζ2)
end

function PestCoordinates(ψ,α,ζ::AbstractVector)
  coords = Vector{PestCoordinates}(undef,length(ζ))
  map!(z->PestCoordinates(ψ,α,z),coords,ζ)
  return coords
end

function PestCoordinates(ψ,α::AbstractVector,ζ::AbstractVector)
  coords = Matrix{PestCoordinates}(undef,length(ζ),length(α))
  for a = 1:length(α)
    for z = 1:length(ζ)
      @inbounds coords[z,a] = PestCoordinates(ψ,α[a],ζ[z])
    end
  end
  return coords
end

function PestCoordinates(ψ,α::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(α) == size(ζ) "Incompatible sizes"
  coords = Matrix{PestCoordinates}(undef,size(α))
  map!((a,z)->PestCoordinates(ψ,a,z),coords,α,ζ)
  return coords
end

function PestCoordinates(ψ::AbstractVector,α::AbstractVector,ζ::AbstractVector)
  coords = Array{PestCoordinates,3}(undef,length(ζ),length(α),length(ψ))
  for s = 1:length(ψ)
    for a = 1:length(α)
      for z = 1:length(ζ)
        @inbounds coords[z,a,s] = PestCoordinates(ψ[s],α[a],ζ[z])
      end
    end
  end
  return coords
end

function PestCoordinates(ψ::AbstractVector,α::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(α) == size(ζ) "Incompatible sizes"
  coords = Array{PestCoordinates,3}(undef,(size(α)...,length(ψ)))
  for s = 1:length(ψ)
    for j = 1:size(α,2)
      for i = 1:size(α,1)
        coords[i,j,s] = PestCoordinates(ψ[s],α[i,j],ζ[i,j])
      end
    end
  end
  return coords
end

function PestCoordinates(ψ::AbstractArray{T,3},α::AbstractArray{T,3},ζ::AbstractArray{T,3}) where T
  @assert size(ψ) == size(α) == size(ζ) "Incompatible sizes"
  coords = Array{PestCoordinates,3}(undef,size(ψ))
  map!((s,a,z)->PestCoordinates(s,a,z),coords,ψ,α,ζ)
  return coords
end

Base.show(io::IO, x::PestCoordinates) = print(io, "PestCoordinates(s=$(x.ψ), α=$(x.α), ζ=$(x.ζ))")
Base.isapprox(x1::PestCoordinates, x2::PestCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.α,x2.α;kwargs...) && isapprox(x1.ζ,x2.ζ;kwargs...)

"""
    BoozerCoordinates{T,A}(ψ::T,χ::A,ϕ::A)

Coordinates on a magnetic flux surface, where `ψ` is the flux label and
`χ` and `ϕ` are angle-like variables
"""
struct BoozerCoordinates{T <: Real,A <: Real} <: MagneticCoordinates
  ψ::T
  χ::A
  ϕ::A
  BoozerCoordinates{T,A}(ψ::T,χ::A,ϕ::A) where {T,A} = new(ψ,χ,ϕ)
end

function BoozerCoordinates(ψ,χ,ϕ)
  ψ2, χ2, ϕ2 = promote(ψ,χ,ϕ)
  return BoozerCoordinates{typeof(ψ2),typeof(χ2)}(ψ2,χ2,ϕ2)
end

function BoozerCoordinates(ψ,χ,ϕ::T) where T <: AbstractVector
  coords = Vector{BoozerCoordinates}(undef,length(ϕ))
  map!(z->BoozerCoordinates(ψ,χ,z),coords,ϕ)
  return coords
end

function BoozerCoordinates(ψ,χ::AbstractVector,ϕ::AbstractVector)
  coords = Matrix{BoozerCoordinates}(undef,length(ϕ),length(χ))
  for x = 1:length(χ)
    for z = 1:length(ϕ)
      @inbounds coords[z,x] = BoozerCoordinates(ψ,χ[x],ϕ[z])
    end
  end
  return coords
end

function BoozerCoordinates(ψ,χ::AbstractMatrix,ϕ::AbstractMatrix)
  @assert size(χ) == size(ϕ) "Incompatible sizes"
  coords = Matrix{BoozerCoordinates}(undef,size(χ))
  map!((x,z)->BoozerCoordinates(ψ,x,z),coords,χ,ϕ)
  return coords
end

function BoozerCoordinates(ψ::AbstractVector,χ::AbstractVector,ϕ::AbstractVector)
  coords = Array{BoozerCoordinates,3}(undef,length(ϕ),length(χ),length(ψ))
  for s = 1:length(ψ)
    for x = 1:length(χ)
      for z = 1:length(ϕ)
        @inbounds coords[z,x,s] = BoozerCoordinates(ψ[s],χ[x],ϕ[z])
      end
    end
  end
  return coords
end

function BoozerCoordinates(ψ::AbstractVector,χ::AbstractMatrix,ϕ::AbstractMatrix)
  @assert size(χ) == size(ϕ) "Incompatible sizes"
  coords = Array{BoozerCoordinates,3}(undef,(size(χ)...,length(ψ)))
  for s = 1:length(ψ)
    for j = 1:size(χ,2)
      for i = 1:size(χ,1)
        coords[i,j,s] = BoozerCoordinates(ψ[s],χ[i,j],ϕ[i,j])
      end
    end
  end
  return coords
end

function BoozerCoordinates(ψ::AbstractArray{T,3},χ::AbstractArray{T,3},ϕ::AbstractArray{T,3}) where T
  @assert size(ψ) == size(χ) == size(ϕ) "Incompatible sizes"
  coords = Array{BoozerCoordinates,3}(undef,size(ψ))
  map!((s,x,z)->BoozerCoordinates(s,x,z),coords,ψ,χ,ϕ)
  return coords
end

Base.show(io::IO, x::BoozerCoordinates) = print(io, "BoozerCoordinates(s=$(x.ψ), χ=$(x.χ), ϕ=$(x.ϕ))")
Base.isapprox(x1::BoozerCoordinates, x2::BoozerCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.χ,x2.χ;kwargs...) && isapprox(x1.ϕ,x2.ϕ;kwargs...)
