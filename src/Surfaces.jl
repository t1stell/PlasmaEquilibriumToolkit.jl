function make_surface_interpolation(valmn,
                                    nfp::Int,
                                    θres::Int,
                                    ζres::Int)

  θs = range(0, length = θres, step = 2π/θres)
  ζs = range(0, length = ζres, step = 2π/nfp/ζres)
  knots = (θs, ζs)
  itp_types = (BSpline(Cubic(Periodic(OnCell()))),
                 BSpline(Cubic(Periodic(OnCell()))))
  itp = (f) -> scale(interpolate(f, itp_types), knots...)
  extp = (f) -> extrapolate(itp(f), (Periodic(), Periodic()))
  T = typeof(inverseTransform(FourierCoordinates(0.0, 0.0, 0.0), valmn))
  field = Array{T, 2}(undef, (length(θs), length(ζs)))
  deriv = similar(field)
  for (j, ζ) in enumerate(ζs)
    for (i, θ) in enumerate(θs)
      field[i, j] = inverseTransform(FourierCoordinates(0.0, θ, ζ), valmn)
      deriv[i, j] = inverseTransform(FourierCoordinates(0.0, θ, ζ), valmn, deriv=:ds)
    end
  end
  return (extp(field), extp(deriv))
end

function magnetic_surface(s::T, eq::E; opt::O) where {T, E <: AbstractMagneticEquilibrium, O}
  println("No surface implemented for eq of type: ",typeof(eq))
end

function surface_get(x::C,
                     surf::S,
                     quantity::Symbol;
                     minθres = 128,
                     minζres = 128,
                     deriv::Symbol=:none) where {C <: AbstractMagneticCoordinates,
                      S <: AbstractSurface}
  #note this function requires the non-mn versions, probably should put this in
  #documentation somehow
  q = nothing
  try
    q = getfield(surf, quantity)
  catch e
    quantity_string = string(quantity)
    println("vmecsurf has no quantity, "*quantity_string)
    return nothing
  end

  function set_surf_field()
    quantity_string = string(quantity)
    quantity_ds = Symbol("d"*quantity_string*"ds")
    quantity_mn = Symbol(quantity_string*"mn")
    qmn = getfield(surf, quantity_mn)
    (q, dqds) = make_surface_interpolation(qmn, surf.nfp, minθres, minζres)
    setfield!(surf, quantity, q)
    setfield!(surf, quantity_ds, dqds)
  end

  if q == nothing
    set_surf_field()
    q = getfield(surf, quantity)
  end

  #check if size is too small
  if size(q.itp)[1] < minθres || size(q.itp)[2] < minζres
    set_surf_field()
    q = getfield(surf, quantity)
  end

  if deriv == :none
    return q(x.θ, x.ζ)
  elseif deriv == :ds
    quantity_ds = Symbol("d"*string(quantity)*"ds")
    dqds = getfield(surf, quantity_ds)
    return dqds(x.θ, x.ζ)
  elseif deriv == :dθ
    return Interpolations.gradient(q, x.θ, x.ζ)[1]
  elseif deriv == :dζ
    return Interpolations.gradient(q, x.θ, x.ζ)[2]
  end
end

function surface_get_exact(x::C,
                           surf::S,
                           quantity::Symbol;
                           deriv::Symbol=:none) where {C <: AbstractMagneticCoordinates,
                           S <: AbstractSurface}
  return inverseTransform(x, getfield(surf, quantity), deriv=deriv)
end

function normal_vector(x::C,
        surf::S;
        ) where {C <: AbstractMagneticCoordinates, S <: AbstractSurface}

  dRdθ = surface_get(x, surf, :r, deriv=:dθ)
  dZdθ = surface_get(x, surf, :z; deriv=:dθ)
  dϕdθ = zero(typeof(x.θ))

  dRdζ = surface_get(x, surf, :r; deriv=:dζ)
  dZdζ = surface_get(x, surf, :z; deriv=:dζ)
  dϕdζ = one(typeof(x.θ))

  a = @SVector [dRdθ, dZdθ, dϕdθ]
  b = @SVector [dRdζ, dZdζ, dϕdζ]

  return cross(a,b)
end

function get_2d_boundary(surf::S, ζ::Float64; res=100
          ) where {S <: AbstractSurface}
  boundary_curve = [(surface_get(FourierCoordinates(0.0, θ, ζ), surf, :r), 
                     surface_get(FourierCoordinates(0.0, θ, ζ), surf, :z))
                     for θ in range(0,2π,res)]
  #enforce equality at boundary just in case small errors
  boundary_curve[end] = boundary_curve[1]
  return boundary_curve

end

"""
  in_surface(cc, surf; res=100)
  in_surface(xyz, surf; res=100)

Calculate where a value is inside a surface given a surface and cylindrical or
cartesian coordinates. Uses the PolygonOps package

"""
function in_surface(point::Union{SVector{2, F}, Tuple{F}},
                    boundary_curve::Vector{Tuple{F,F}};
                    inverse::Bool=false) where {F <: AbstractFloat}
  #helper version to allow calling with other surfaces
  if !inverse
    return inpolygon(point, boundary_curve, in=true, on=true, out=false)
  else
    return inpolygon(point, boundary_curve, in=false, on=false, out=true)
  end
end

function in_surface(cc::Cylindrical, surf::S; res=100
                   ) where {S <: AbstractSurface}
  ζ = cc.θ
  boundary_curve = get_2d_boundary(surf, ζ, res=res)
  point = @SVector [cc.r, cc.z]
  return in_surface(point, boundary_curve)
  
end

function in_surface(xyz::SVector, surf::S; res=100
                   ) where {S <: AbstractSurface}
  cc = CylindricalFromCartesian()(xyz)
  return in_surface(cc, surf, res=res)
end


#cribbed version of "inpolygon" from PolygonOps but without
#the obnoxious input format
function in_surface(x::T, 
                    y::T,
                    xwall::Vector{T},
                    ywall::Vector{T};
                    inverse::Bool=false) where {T <: Real}

    if abs(xwall[1] - xwall[end]) > 1.0E-12
        println("Warning: x wall dimensions do not match, aborting calc")
        return !inverse
    elseif abs(ywall[1] - ywall[end]) > 1.0E-12
        println("Warning: y wall dimensions do not match, aborting calc")
        return !inverse
    end

    k=0
    for i in UnitRange(firstindex(xwall), lastindex(xwall) - 1)
        v1 = ywall[i] - y
        v2 = ywall[i+1] - y

        if v1 < zero(T) && v2 < zero(T) || v1 > zero(T) && v2 > zero(T)
            continue
        end

        u1 = xwall[i] - x
        u2 = xwall[i+1] - x

        f = (u1 * v2) - (u2 * v1)

        if v2 > zero(T) && v1 <= zero(T) 
            if f > zero(T)
                k += 1
            elseif iszero(f)
                return !inverse
            end
        elseif v1 > zero(T) && v2 <= zero(T)
            if f < zero(T)
                k += 1
            elseif iszero(f)
                return !inverse
            end
        elseif iszero(v2) && v1 < zero(T)
            iszero(f) && return !inverse
        elseif iszero(v1) && v2 < zero(T)
            iszero(f) && return !inverse
        elseif iszero(v1) && iszero(v2)
            if u2 <= zero(T) && u1 >= zero(T)
                return !inverse
            elseif u1 <= zero(T) && u2 >= zero(T)
                return !inverse
            end
        end
    end

    iszero(k%2) && return inverse
    return !inverse
end
