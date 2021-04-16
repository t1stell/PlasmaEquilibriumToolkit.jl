"""
    AbstractMagneticEquilibrium

Abstract supertype for different magnetic equilibrium representations (VMEC, SPEC,...).
"""
abstract type AbstractMagneticEquilibrium end;

"""
    AbstractMagneticCoordinates

Abstract supertype for different magnetic coordinates.
"""
abstract type AbstractMagneticCoordinates end;

"""
    NullEquilibrium()

Empty subtype of MagneticEquilibrium to represent no equilibrium
"""
struct NullEquilibrium <: AbstractMagneticEquilibrium end

"""
    ClebschCoordinates(α,β,η)

Coordintes `(α,β,η)` representing a divergence-free magnetic field ``B = ∇α×∇β``
with a Jacobian defined by ``√g = 1/(∇η ⋅ ∇α × ∇β)``.
"""
struct ClebschCoordinates{T <: Real} <: AbstractMagneticCoordinates
  α::T
  β::T
  η::T
  ClebschCoordinates{T}(α::T,β::T,η::T) where {T} = new(α,β,η)
end

function ClebschCoordinates(α,β,η)
  α2,β2,ζ2 = promote(α,β,η)
  return ClebschCoordinates{typeof(α2)}(α2,β2,η2)
end

"""
    FluxCoordinates{T,A}(ψ::T,θ::A,ζ::A) <: MagneticCoordinates

Coordinates on a magnetic flux surface, where `ψ` is the physical toroidal flux
divided by 2π and `θ` and `ζ` are angle-like variables
"""
struct FluxCoordinates{T <: Real,A <: Real} <: AbstractMagneticCoordinates
  ψ::T
  θ::A
  ζ::A
  FluxCoordinates{T,A}(ψ::T,θ::A,ζ::A) where {T,A} = new(ψ,θ,ζ)
end

function FluxCoordinates(ψ,θ,ζ)
  ψ2, θ2, ζ2 = promote(ψ,θ,ζ)
  return FluxCoordinates{typeof(ψ2),typeof(θ2)}(ψ2,θ2,ζ2)
end
#=
"""
    FluxCoordinates(ψ,θ,ζ::AbstractVector)

Generate a `Vector{FluxCoordinates}` given by `(s,θ,ζⱼ)` where `ζⱼ` is the `j`-th entry of `ζ`.
"""
function FluxCoordinates(ψ,θ,ζ::AbstractVector)
  coords = Vector{FluxCoordinates}(undef,length(ζ))
  map!(z->FluxCoordinates(ψ,θ,z),coords,ζ)
  return coords
end

"""
    FluxCoordinates(ψ,θ::AbstractVector,ζ::AbstractVector)

Generate a grid of `FluxCoordinates` given by `(ψ,θᵢ,ζⱼ)` is the `i`-th entry of `θ` and
`ζⱼ` is the `j`-th entry of `ζ`.
"""
function FluxCoordinates(ψ,θ::AbstractVector,ζ::AbstractVector)
  coords = Matrix{FluxCoordinates}(undef,length(ζ),length(θ))
  for t = 1:length(θ)
    for z = 1:length(ζ)
      @inbounds coords[z,t] = FluxCoordinates(ψ,θ[t],ζ[z])
    end
  end
  return coords
end


"""
    FluxCoordinates(ψ,θ::AbstractMatrix,ζ::AbstractMatrix)

Generate a grid of `FluxCoordinates` given by `(ψ,θᵢⱼ,ζᵢⱼ)` where `θ(ζ)ᵢⱼ` is the `ij`-entry of the `θ(ζ)` matrix.
"""
function FluxCoordinates(ψ,θ::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(θ) == size(ζ) "Incompatible sizes"
  coords = Matrix{FluxCoordinates}(undef,size(θ))
  map!((t,z)->FluxCoordinates(ψ,t,z),coords,θ,ζ)
  return coords
end

"""
    FluxCoordinates(ψ::AbstractVector,θ::AbstractVector,ζ::AbstractVector)

Generate a 3D grid of `FluxCoordinates` given by `(ψᵢ,θⱼ,ζₖ)` where `ψᵢ` is the `i`-th entry of `ψ`,
`θⱼ` is the `j`-th entry of `θ` and `ζₖ` is the `k`-th entry of `ζ`.
"""
function FluxCoordinates(s::AbstractVector,θ::AbstractVector,ζ::AbstractVector)
  coords = Array{FluxCoordinates,3}(undef,length(ζ),length(θ),length(ψ))
  for k = 1:length(ψ)
    for t = 1:length(θ)
      for z = 1:length(ζ)
        @inbounds coords[z,t,k] = FluxCoordinates(ψ[k],θ[t],ζ[z])
      end
    end
  end
  return coords
end

"""
    FluxCoordinates(ψ::AbstractVector,θ::AbstractMatrix,ζ::AbstractMatrix)

Generate a 3D grid of `FluxCoordinates` given by `(ψᵢ,θⱼₖ,ζⱼₖ)` where `ψᵢ` is the `i`-th entry of `s`,
`θⱼₖ(ζⱼₖ)` is the `jk`-th entry of `θ(ζ)`.
"""
function FluxCoordinates(ψ::AbstractVector,θ::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(θ) == size(ζ) "Incompatible sizes"
  coords = Array{FluxCoordinates,3}(undef,(size(θ)...,length(ψ)))
  for k = 1:length(ψ)
    for j = 1:size(θ,2)
      for i = 1:size(θ,1)
        coords[i,j,k] = FluxCoordinates(ψ[k],θ[i,j],ζ[i,j])
      end
    end
  end
  return coords
end

"""
    FluxCoordinates(ψ::AbstractArray{T,3},θ::AbstractArray{T,3},ζ::AbstractArray{T,3})

Generate a 3D grid of `FluxCoordinates` given by `(ψᵢⱼₖ,θᵢⱼₖ,ζᵢⱼₖ)` where `ψᵢⱼₖ` is the `ijk`-th entry of `s`,
`θᵢⱼₖ` is the `ijk`-th entry of `θ` and  `ζᵢⱼₖ` is the `ijk`-th entry of `ζ`.
"""
function FluxCoordinates(ψ::AbstractArray{T,3},θ::AbstractArray{T,3},ζ::AbstractArray{T,3}) where T
  @assert size(ψ) == size(θ) == size(ζ) "Incompatible sizes"
  coords = Array{FluxCoordinates,3}(undef,size(ψ))
  map!((s,t,z)->FluxCoordinates(s,t,z),coords,ψ,θ,ζ)
  return coords
end
=#
Base.show(io::IO, x::FluxCoordinates) = print(io, "FluxCoordinates(ψ=$(x.ψ), θ=$(x.θ), ζ=$(x.ζ))")
Base.isapprox(x1::FluxCoordinates, x2::FluxCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.θ,x2.θ;kwargs...) && isapprox(x1.ζ,x2.ζ;kwargs...)

"""
    PestCoordinates{T,A}(ψ::T,α::A,ζ::A) <: MagneticCoordinates

Coordinates on a magnetic flux surface, where `ψ` is the toroidal flux divided by 2π
in the clockwise direction, `α` is the field line label, and `ζ` is the geometric toroidal
angle advancing the clockwise direction.
"""
struct PestCoordinates{T <: Real,A <: Real} <: AbstractMagneticCoordinates
  ψ::T
  α::A
  ζ::A
  PestCoordinates{T,A}(ψ::T,α::A,ζ::A) where {T,A} = new(ψ,α,ζ)
end

function PestCoordinates(ψ,α,ζ)
  ψ2, α2, ζ2 = promote(ψ,α,ζ)
  return PestCoordinates{typeof(ψ2),typeof(α2)}(ψ2,α2,ζ2)
end
#=
"""
    PestCoordinates(ψ,α,ζ::AbstractVector)

Generate a `Vector{PestCoordinates}` given by `(ψ,α,ζⱼ)` where `ζⱼ` is the `j`-th entry of `ζ`.
"""
function PestCoordinates(ψ,α,ζ::AbstractVector)
  coords = Vector{PestCoordinates}(undef,length(ζ))
  map!(z->PestCoordinates(ψ,α,z),coords,ζ)
  return coords
end

"""
    PestCoordinates(ψ,α::AbstractVector,ζ::AbstractVector)

Generate a grid of `PestCoordinates` given by `(s,αᵢ,ζⱼ)` is the `i`-th entry of `α` and
`ζⱼ` is the `j`-th entry of `ζ`.
"""
function PestCoordinates(ψ,α::AbstractVector,ζ::AbstractVector)
  coords = Matrix{PestCoordinates}(undef,length(ζ),length(α))
  for a = 1:length(α)
    for z = 1:length(ζ)
      @inbounds coords[z,a] = PestCoordinates(ψ,α[a],ζ[z])
    end
  end
  return coords
end

"""
    PestCoordinates(ψ,α::AbstractMatrix,ζ::AbstractMatrix)

Generate a grid of `PestCoordinates` given by `(ψ,αᵢⱼ,ζᵢⱼ)` where `α(ζ)ᵢⱼ` is the `ij`-entry of the `α(ζ)` matrix.
"""
function PestCoordinates(ψ,α::AbstractMatrix,ζ::AbstractMatrix)
  @assert size(α) == size(ζ) "Incompatible sizes"
  coords = Matrix{PestCoordinates}(undef,size(α))
  map!((a,z)->PestCoordinates(ψ,a,z),coords,α,ζ)
  return coords
end

"""
    PestCoordinates(ψ::AbstractVector,α::AbstractVector,ζ::AbstractVector)

Generate a 3D grid of `PestCoordinates` given by `(ψᵢ,αⱼ,ζₖ)` where `ψᵢ` is the `i`-th entry of `ψ`,
`αⱼ` is the `j`-th entry of `α` and `ζₖ` is the `k`-th entry of `ζ`.
"""
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

"""
    PestCoordinates(ψ::AbstractVector,α::AbstractMatrix,ζ::AbstractMatrix)

Generate a 3D grid of `PestCoordinates` given by `(ψᵢ,αⱼₖ,ζⱼₖ)` where `ψᵢ` is the `i`-th entry of `ψ`,
`αⱼₖ(ζⱼₖ)` is the `jk`-th entry of `α(ζ)`.
"""
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

"""
    PestCoordinates(ψ::AbstractArray{T,3},α::AbstractArray{T,3},ζ::AbstractArray{T,3})

Generate a 3D grid of `PestCoordinates` given by `(ψᵢⱼₖ,αᵢⱼₖ,ζᵢⱼₖ)` where `ψᵢⱼₖ` is the `ijk`-th entry of `ψ`,
`αᵢⱼₖ` is the `ijk`-th entry of `α` and  `ζᵢⱼₖ` is the `ijk`-th entry of `ζ`.
"""
function PestCoordinates(ψ::AbstractArray{T,3},α::AbstractArray{T,3},ζ::AbstractArray{T,3}) where T
  @assert size(ψ) == size(α) == size(ζ) "Incompatible sizes"
  coords = Array{PestCoordinates,3}(undef,size(ψ))
  map!((s,a,z)->PestCoordinates(s,a,z),coords,ψ,α,ζ)
  return coords
end
=#
Base.show(io::IO, x::PestCoordinates) = print(io, "PestCoordinates(ψ=$(x.ψ), α=$(x.α), ζ=$(x.ζ))")
Base.isapprox(x1::PestCoordinates, x2::PestCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.α,x2.α;kwargs...) && isapprox(x1.ζ,x2.ζ;kwargs...)

"""
    BoozerCoordinates{T,A}(ψ::T,χ::A,ϕ::A) <: MagneticCoordinates

Coordinates on a magnetic flux surface, where `ψ` is the flux label and
`χ` and `ϕ` are angle-like variables
"""
struct BoozerCoordinates{T <: Real,A <: Real} <: AbstractMagneticCoordinates
  ψ::T
  χ::A
  ϕ::A
  BoozerCoordinates{T,A}(ψ::T,χ::A,ϕ::A) where {T,A} = new(ψ,χ,ϕ)
end

function BoozerCoordinates(ψ,χ,ϕ)
  ψ2, χ2, ϕ2 = promote(ψ,χ,ϕ)
  return BoozerCoordinates{typeof(ψ2),typeof(χ2)}(ψ2,χ2,ϕ2)
end
#=
"""
    BoozerCoordinates(ψ,χ,ϕ::AbstractVector)

Generate a `Vector{BoozerCoordinates}` given by `(ψ,χ,ϕⱼ)` where `ϕⱼ` is the `j`-th entry of `ϕ`.
"""
function BoozerCoordinates(ψ,χ,ϕ::AbstractVector)
  coords = Vector{BoozerCoordinates}(undef,length(ϕ))
  map!(z->BoozerCoordinates(ψ,χ,z),coords,ϕ)
  return coords
end

"""
    BoozerCoordinates(ψ,χ::AbstractVector,ϕ::AbstractVector)

Generate a grid of `BoozerCoordinates` given by `(s,χᵢ,ϕⱼ)` is the `i`-th entry of `χ` and
`ϕⱼ` is the `j`-th entry of `ϕ`.
"""
function BoozerCoordinates(ψ,χ::AbstractVector,ϕ::AbstractVector)
  coords = Matrix{BoozerCoordinates}(undef,length(ϕ),length(χ))
  for x = 1:length(χ)
    for z = 1:length(ϕ)
      @inbounds coords[z,x] = BoozerCoordinates(ψ,χ[x],ϕ[z])
    end
  end
  return coords
end

"""
    BoozerCoordinates(ψ,χ::AbstractMatrix,ϕ::AbstractMatrix)

Generate a grid of `BoozerCoordinates` given by `(ψ,χᵢⱼ,ϕᵢⱼ)` where `χ(ϕ)ᵢⱼ` is the `ij`-entry of the `χ(ϕ)` matrix.
"""
function BoozerCoordinates(ψ,χ::AbstractMatrix,ϕ::AbstractMatrix)
  @assert size(χ) == size(ϕ) "Incompatible sizes"
  coords = Matrix{BoozerCoordinates}(undef,size(χ))
  map!((x,z)->BoozerCoordinates(ψ,x,z),coords,χ,ϕ)
  return coords
end

"""
    BoozerCoordinates(ψ::AbstractVector,χ::AbstractVector,ϕ::AbstractVector)

Generate a 3D grid of `BoozerCoordinates` given by `(ψᵢ,χⱼ,ϕₖ)` where `ψᵢ` is the `i`-th entry of `ψ`,
`χⱼ` is the `j`-th entry of `χ` and `ϕₖ` is the `k`-th entry of `ϕ`.
"""
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

"""
    BoozerCoordinates(ψ::AbstractVector,χ::AbstractMatrix,ϕ::AbstractMatrix)

Generate a 3D grid of `BoozerCoordinates` given by `(ψᵢ,χⱼₖ,ϕⱼₖ)` where `ψᵢ` is the `i`-th entry of `ψ`,
`χⱼₖ(ϕⱼₖ)` is the `jk`-th entry of `χ(ϕ)`.
"""
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

"""
    BoozerCoordinates(ψ::AbstractArray{T,3},χ::AbstractArray{T,3},ϕ::AbstractArray{T,3})

Generate a 3D grid of `BoozerCoordinates` given by `(ψᵢⱼₖ,χᵢⱼₖ,ϕᵢⱼₖ)` where `ψᵢⱼₖ` is the `ijk`-th entry of `ψ`,
`χᵢⱼₖ` is the `ijk`-th entry of `χ` and  `ϕᵢⱼₖ` is the `ijk`-th entry of `ϕ`.
"""
function BoozerCoordinates(ψ::AbstractArray{T,3},χ::AbstractArray{T,3},ϕ::AbstractArray{T,3}) where T
  @assert size(ψ) == size(χ) == size(ϕ) "Incompatible sizes"
  coords = Array{BoozerCoordinates,3}(undef,size(ψ))
  map!((s,x,z)->BoozerCoordinates(s,x,z),coords,ψ,χ,ϕ)
  return coords
end
=#
Base.show(io::IO, x::BoozerCoordinates) = print(io, "BoozerCoordinates(ψ=$(x.ψ), χ=$(x.χ), ϕ=$(x.ϕ))")
Base.isapprox(x1::BoozerCoordinates, x2::BoozerCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.χ,x2.χ;kwargs...) && isapprox(x1.ϕ,x2.ϕ;kwargs...)
