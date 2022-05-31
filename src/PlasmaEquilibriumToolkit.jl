module PlasmaEquilibriumToolkit

using LinearAlgebra
using StaticArrays
using StructArrays
using LabelledArrays
using CoordinateTransformations
using Interpolations
using Polyester

# Export statements
# Abstract quantities
export AbstractMagneticGeometry, AbstractMagneticCoordinates
export AbstractMagneticEquilibrium, NullEquilibrium
export AbstractMagneticSurface, MagneticSurface
export AbstractMagneticFieldline, MagneticFieldline
export MagneticCoordinateGrid, MagneticCoordinateCurve

# Fourier Series
export SurfaceFourierData, SurfaceFourierArray, FourierCoordinates
export inverseTransform, cosineTransform, sineTransform

# Magnetic coordinate concretizations
export ClebschCoordinates, FluxCoordinates, PestCoordinates, BoozerCoordinates

# Surfaces
export FourierSurface

# Magnetic coordinate transformations
export FluxFromPest, FluxFromBoozer, FluxFromClebsch
export PestFromFlux, PestFromBoozer, PestFromClebsch
export BoozerFromFlux, BoozerFromPest, BoozerFromClebsch
export CylindricalFromFlux, CylindricalFromPest, CylindricalFromBoozer
export CartesianFromFlux, CartesianFromPest, CartesianFromBoozer
export CylindricalFromFourier

# Basis vector quantities
export BasisTransformation, BasisTypes, Covariant, Contravariant
export CoordinateVector, BasisVectors
export basis_vectors, transform_deriv

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


# Specialzed coordinate defintions, transformations, routines for different codes
include("Specializations.jl")

const PET = PlasmaEquilibriumToolkit
export PET
end
