import Base.abs
using LinearAlgebra

struct FluxFromPest <: Transformation; end
struct PestFromFlux <: Transformation; end
struct CylindricalFromFlux <: Transformation; end
struct CylindricalFromPest <: Transformation; end
struct CartesianFromFlux <: Transformation; end
struct CartesianFromPest <: Transformation; end
struct ContravariantFromCovariant <: Transformation; end
struct CovariantFromContravariant <: Transformation; end
struct Covariant end
struct Contravariant end

function abs(e::BasisVectors{T},
             component::Int=0;
            ) where T
  if component > 0 && component <= 3
    return norm(e[:,component])
  else
    return norm(e)
  end
end

function jacobian(::Covariant,
                  e::BasisVectors{T};
                 ) where T
  return dot(e[:,1],cross(e[:,2],e[:,3]))
end

function jacobian(::Contravariant,
                  e::BasisVectors{T};
                ) where T
  return 1.0 /dot(e[:,1],cross(e[:,2],e[:,3]))
end

function transform_basis(::ContravariantFromCovariant,
                         e::BasisVectors{T},
                         J::T;
                        ) where T
  grad_1 = cross(e[:,2],e[:,3])/J
  grad_2 = cross(e[:,3],e[:,1])/J
  grad_3 = cross(e[:,1],e[:,2])/J
  return hcat(grad_1,grad_2,grad_3)
end

function transform_basis(::CovariantFromContravariant,
                         e::BasisVectors{T},
                         J::T;
                        ) where T
  e_1 = cross(e[:,2],e[:,3])*J
  e_2 = cross(e[:,3],e[:,1])*J
  e_3 = cross(e[:,1],e[:,2])*J
  return hcat(e_1,e_2,e_3)
end

function transform_basis(::ContravariantFromCovariant,
                         e::BasisVectors;
                        )
  J = jacobian(Covariant(),e)
  return transform_basis(ContravariantFromCovariant(),e,J)
end

function transform_basis(::CovariantFromContravariant,
                         e::BasisVectors;
                        )
  J = jacobian(Contravariant(),e)
  return transform_basis(CovariantFromContravariant(),e,J)
end

function transform_basis(t::Transformation,
                         e::AbstractArray{BasisVectors{T}},
                         J::AbstractArray{T};
                        ) where T
  res = similar(e)
  Threads.@threads for i = 1:length(e)
    res[i] = transform_basis(t,e[i],J[i])
  end
  return res
end

function transform_basis(t::Transformation,
                         e::AbstractArray{BasisVectors};
                        )
  res = similar(e)
  Threads.@threads for i = 1:length(e)
    res[i] = transform_basis(t,e[i])
  end
  return res
end

function transform_basis(t::Transformation,
                         x::AbstractArray,
                         e::AbstractArray{BasisVectors},
                         eq::AbstractMagneticEquilibrium;
                        )
  @assert ndims(x) == ndims(e) && size(x) == size(e) "Incompatible coordinate/basis vector arrays!"
  res = similar(e)
  Threads.@threads for i = 1:length(x)
    res[i] = transform_basis(t,x[i],e[i],eq)
  end
  return res
end

function covariant_basis
end

function contravariant_basis
end

function covariant_basis(t::Transformation,
                         x::AbstractArray,
                         eq::AbstractMagneticEquilibrium);
  res = Array{BasisVectors{typeof(getfield(first(x),1))},ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    res[i] = covariant_basis(t,x[i],eq)
  end
  return res
end

function contravariant_basis(t::Transformation,
                             x::AbstractArray,
                             eq::AbstractMagneticEquilibrium;
                            )
  res = Array{BasisVectors{typeof(getfield(first(x),1))},ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    res[i] = contravariant_basis(t,x[i],eq)
  end
  return res
end

function (t::Transformation)(x::AbstractArray,
                             eq::AbstractMagneticEquilibrium;
                            )
  y = Array{typeof(t(first(x),eq)),ndims(x)}(undef,size(x))
  Threads.@threads for i = 1:length(x)
    y[i] = t(x[i],eq)
  end
  return y
end
