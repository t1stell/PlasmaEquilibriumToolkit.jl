function transform_deriv(::CylindricalFromFourier, 
                         x::FourierCoordinates,
                         surf::FourierSurface;
                         )
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

