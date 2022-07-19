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
  field = [inverseTransform(VmecCoordinates(0.0, θ, ζ), valmn)
           for ζ in ζs for θ in θs]
  field = reshape(field, (θres, ζres))
  deriv = [inverseTransform(VmecCoordinates(0.0, θ, ζ), valmn, deriv=:ds)
           for ζ in ζs for θ in θs]
  deriv = reshape(deriv, (θres, ζres))
  return (extp(field), extp(deriv))
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

function surface_get_exact(x::AbstractMagneticCoordinates,
                           surf::AbstractSurface,
                           quantity::Symbol;
                           deriv::Symbol=:none) where {T}
  return inverseTransform(x, getfield(vmecsurf, quantity), deriv=deriv)
end



function CoordinateTransformations.transform_deriv(::CylindricalFromFourier, 
			 x::FourierCoordinates{T, T},
			 surf::FourierSurface{T};
		        ) where {T}
  dRds = inverseTransform(x, surf.rmn; deriv=:ds)
  dZds = inverseTransform(x, surf.zmn; deriv=:ds)
  dϕds = zero(typeof(x.θ))

  dRdθ = inverseTransform(x, surf.rmn; deriv=:dθ)
  dZdθ = inverseTransform(x, surf.zmn; deriv=:dθ)
  dϕdθ = zero(typeof(x.θ))

  dRdζ = inverseTransform(x, surf.rmn; deriv=:dζ)
  dZdζ = inverseTransform(x, surf.zmn; deriv=:dζ)
  dϕdζ = one(typeof(x.θ))
  return @SMatrix [dRds dRdθ dRdζ;
                   dϕds dϕdθ dϕdζ;
                   dZds dZdθ dZdζ]
end

function normal_vector(x::FourierCoordinates{T, T},
        surf::FourierSurface{T};
        ) where {T}

  dRdθ = inverseTransform(x, surf.rmn; deriv=:dθ)
  dZdθ = inverseTransform(x, surf.zmn; deriv=:dθ)
  dϕdθ = zero(typeof(x.θ))

  dRdζ = inverseTransform(x, surf.rmn; deriv=:dζ)
  dZdζ = inverseTransform(x, surf.zmn; deriv=:dζ)
  dϕdζ = one(typeof(x.θ))

  a = @SVector [dRdθ, dZdθ, dϕdθ]
  b = @SVector [dRdζ, dZdζ, dϕdζ]

  return cross(a,b)
end

#Template functions
#function surface_get() end
#function surface_get_exact() end
