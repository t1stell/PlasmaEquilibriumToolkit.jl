module PlasmaEquilibriumToolkit

using LinearAlgebra
using StaticArrays
using StructArrays
using LabelledArrays
using CoordinateTransformations
using Interpolations
using PolygonOps
using Polyester
using Roots

# Export statements
# Abstract quantities
export AbstractGeometry, AbstractMagneticGeometry, AbstractMagneticCoordinates
export AbstractMagneticEquilibrium, NullEquilibrium
export AbstractMagneticSurface, MagneticSurface, SurfaceQuantity
export AbstractMagneticFieldline, MagneticFieldline
export MagneticCoordinateGrid, MagneticCoordinateCurve
export AbstractMagneticField, MagneticField

# Fourier Series
export SurfaceFourierData, SurfaceFourierArray, FourierCoordinates
export inverseTransform, cosineTransform, sineTransform

# Magnetic coordinate concretizations
export ClebschCoordinates, FluxCoordinates, PestCoordinates, BoozerCoordinates

# Surfaces
export FourierSurface, SplineSurface
export surface_get, surface_get_exact
export in_surface

# Magnetic coordinate transformations
export FluxFromPest, FluxFromBoozer, FluxFromClebsch
export PestFromFlux, PestFromBoozer, PestFromClebsch
export BoozerFromFlux, BoozerFromPest, BoozerFromClebsch
export CylindricalFromFlux, CylindricalFromPest, CylindricalFromBoozer
export CartesianFromFlux, CartesianFromPest, CartesianFromBoozer
export CylindricalFromFourierm, θ_internal, flux_surface_and_angle

# Basis vector quantities
export BasisTransformation, BasisTypes, Covariant, Contravariant
export CoordinateVector, BasisVectors
export basis_vectors, normal_vector

# Change of basis quantities
export CovariantFromContravariant, ContravariantFromCovariant
export transform_basis, jacobian

# Derived quantities
export grad_B, grad_B_projection, curvature_components
export normal_curvature, geodesic_curvature, metric
export grad_B, B_norm, B_field


include("Types.jl")
include("FourierTransformFunctions.jl")
include("MagneticCoordinates.jl")
include("MagneticCoordinateGrid.jl")
include("Transformations.jl")
include("DerivedQuantities.jl")
include("Surfaces.jl")
include("MagneticField.jl")

const PET = PlasmaEquilibriumToolkit
export PET
end
