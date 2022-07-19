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

function SurfaceGet() end
function SurfaceGetExact() end
