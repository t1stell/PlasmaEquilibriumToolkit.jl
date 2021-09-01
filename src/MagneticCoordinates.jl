
"""
    ClebschCoordinates(α,β,η)

Coordintes `(α,β,η)` representing a divergence-free magnetic field ``B = ∇α×∇β``
with a Jacobian defined by ``√g = 1/(∇η ⋅ ∇α × ∇β)``.
"""
struct ClebschCoordinates{T<:Real,A<:Real} <: AbstractMagneticCoordinates
  α::T
  β::A
  η::A
  ClebschCoordinates{T,A}(α::T, β::A, η::A) where {T,A} = new(α, β, η)
end

function ClebschCoordinates(α, β, η)
  α2, β2, η2 = promote(α, β, η)
  return ClebschCoordinates{typeof(α2),typeof(β2)}(α2, β2, η2)
end

"""
    FluxCoordinates{T,A}(ψ::T,θ::A,ζ::A) <: MagneticCoordinates

Coordinates on a magnetic flux surface, where `ψ` is the physical toroidal flux
divided by 2π and `θ` and `ζ` are angle-like variables
"""
struct FluxCoordinates{T<:Real,A<:Real} <: AbstractMagneticCoordinates
  ψ::T
  θ::A
  ζ::A
  FluxCoordinates{T,A}(ψ::T, θ::A, ζ::A) where {T,A} = new(ψ, θ, ζ)
end

function FluxCoordinates(ψ, θ, ζ)
  ψ2, θ2, ζ2 = promote(ψ, θ, ζ)
  return FluxCoordinates{typeof(ψ2),typeof(θ2)}(ψ2, θ2, ζ2)
end

Base.show(io::IO, x::FluxCoordinates) =
  print(io, "FluxCoordinates(ψ=$(x.ψ), θ=$(x.θ), ζ=$(x.ζ))")
Base.isapprox(x1::FluxCoordinates, x2::FluxCoordinates; kwargs...) =
  isapprox(x1.ψ, x2.ψ; kwargs...) &&
  isapprox(x1.θ, x2.θ; kwargs...) &&
  isapprox(x1.ζ, x2.ζ; kwargs...)

"""
    PestCoordinates{T,A}(ψ::T,α::A,ζ::A) <: MagneticCoordinates

Coordinates on a magnetic flux surface, where `ψ` is the toroidal flux divided by 2π
in the clockwise direction, `α` is the field line label, and `ζ` is the geometric toroidal
angle advancing the clockwise direction.
"""
struct PestCoordinates{T<:Real,A<:Real} <: AbstractMagneticCoordinates
  ψ::T
  α::A
  ζ::A
  PestCoordinates{T,A}(ψ::T, α::A, ζ::A) where {T,A} = new(ψ, α, ζ)
end

function PestCoordinates(ψ, α, ζ)
  ψ2, α2, ζ2 = promote(ψ, α, ζ)
  return PestCoordinates{typeof(ψ2),typeof(α2)}(ψ2, α2, ζ2)
end

Base.show(io::IO, x::PestCoordinates) =
  print(io, "PestCoordinates(ψ=$(x.ψ), α=$(x.α), ζ=$(x.ζ))")
Base.isapprox(x1::PestCoordinates, x2::PestCoordinates; kwargs...) =
  isapprox(x1.ψ, x2.ψ; kwargs...) &&
  isapprox(x1.α, x2.α; kwargs...) &&
  isapprox(x1.ζ, x2.ζ; kwargs...)

"""
    BoozerCoordinates{T,A}(ψ::T,χ::A,ϕ::A) <: MagneticCoordinates

Coordinates on a magnetic flux surface, where `ψ` is the flux label and
`χ` and `ϕ` are angle-like variables
"""
struct BoozerCoordinates{T<:Real,A<:Real} <: AbstractMagneticCoordinates
  ψ::T
  χ::A
  ϕ::A
  BoozerCoordinates{T,A}(ψ::T, χ::A, ϕ::A) where {T,A} = new(ψ, χ, ϕ)
end

function BoozerCoordinates(ψ, χ, ϕ)
  ψ2, χ2, ϕ2 = promote(ψ, χ, ϕ)
  return BoozerCoordinates{typeof(ψ2),typeof(χ2)}(ψ2, χ2, ϕ2)
end

Base.show(io::IO, x::BoozerCoordinates) =
  print(io, "BoozerCoordinates(ψ=$(x.ψ), χ=$(x.χ), ϕ=$(x.ϕ))")
Base.isapprox(x1::BoozerCoordinates, x2::BoozerCoordinates; kwargs...) =
  isapprox(x1.ψ, x2.ψ; kwargs...) &&
  isapprox(x1.χ, x2.χ; kwargs...) &&
  isapprox(x1.ϕ, x2.ϕ; kwargs...)
