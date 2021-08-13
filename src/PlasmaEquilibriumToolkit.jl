module PlasmaEquilibriumToolkit

using LinearAlgebra
using StaticArrays
using StructArrays
using LabelledArrays
using CoordinateTransformations
using Polyester

# Export statements
# Abstract quantities
export AbstractMagneticGeometry, AbstractMagneticCoordinates
export AbstractMagneticEquilibrium, NullEquilibrium
export AbstractMagneticSurface, MagneticSurface
export AbstractMagneticFieldline, MagneticFieldline
export MagneticCoordinateGrid

# Magnetic coordinate concretizations
export ClebschCoordinates, FluxCoordinates, PestCoordinates, BoozerCoordinates

# Magnetic coordinate transformations
export FluxFromPest, FluxFromBoozer, FluxFromClebsch
export PestFromFlux, PestFromBoozer, PestFromClebsch
export BoozerFromFlux, BoozerFromPest, BoozerFromClebsch
export CylindricalFromFlux, CylindricalFromPest, CylindricalFromBoozer
export CartesianFromFlux, CartesianFromPest, CartesianFromBoozer

# Basis vector quantities
export BasisTypes, Covariant, Contravariant
export CoordinateVector, BasisVectors
export basis_vectors

# Change of basis quantities
export CovariantFromContravariant, ContravariantFromCovariant
export transform_basis, jacobian

# Derived quantities
export gradB, curvatureProjection, curvatureComponents
export normalCurvature, geodesicCurvature, metric
export gradB, Bnorm, Bfield


include("Types.jl")
include("MagneticCoordinates.jl")
include("MagneticCoordinateGrid.jl")
include("Transformations.jl")
include("DerivedQuantities.jl")

# Specialzed coordinate defintions, transformations, routines for different codes
#include("Specializations.jl")
end
