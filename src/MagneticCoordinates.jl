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

function PestCoordinates(s,α,ζ)
  ψ2, α2, ζ2 = promote(s,α,ζ)
  return PestCoordinates{typeof(ψ2),typeof(α2)}(ψ2,α2,ζ2)
end

Base.show(io::IO, x::PestCoordinates) = print(io, "PestCoordinates(s=$(x.ψ), α=$(x.α), ζ=$(x.ζ))")
Base.isapprox(x1::PestCoordinates, x2::PestCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.α,x2.α;kwargs...) && isapprox(x1.ζ,x2.ζ;kwargs...)

"""
    BoozerCoordinates{T,A}(s::T,θ::A,ζ::A)

Coordinates on a magnetic flux surface, where `s` is the surface label and
`θ` and `ζ` are angle-like variables
"""
struct BoozerCoordinates{T <: Real,A <: Real} <: MagneticCoordinates
  ψ::T
  χ::A
  ζ::A
  BoozerCoordinates{T,A}(ψ::T,χ::A,ζ::A) where {T,A} = new(ψ,χ,ζ)
end

function BoozerCoordinates(s,χ,ζ)
  ψ2, χ2, ζ2 = promote(s,χ,ζ)
  return BoozerCoordinates{typeof(ψ2),typeof(χ2)}(ψ2,χ2,ζ2)
end

Base.show(io::IO, x::BoozerCoordinates) = print(io, "PestCoordinates(s=$(x.ψ), α=$(x.χ), ζ=$(x.ζ))")
Base.isapprox(x1::BoozerCoordinates, x2::BoozerCoordinates; kwargs...) = isapprox(x1.ψ,x2.ψ;kwargs...) && isapprox(x1.χ,x2.χ;kwargs...) && isapprox(x1.ζ,x2.ζ;kwargs...)
