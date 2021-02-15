module PlasmaEquilibriumToolkit

export AbstractMagneticGeometry, MagneticGeometry
export MagneticEquilibrium, MagneticCoordinates, NullEquilibrium
export FluxCoordinates, PestCoordinates, BoozerCoordinates
export FluxFromPest, PestFromFlux, CylindricalFromFlux, CylindricalFromPest
export CartesianFromFlux, CartesianFromPest
export Covariant, Contravariant, CoordinateVector, BasisVectors
export CovariantFromContravariant, ContravariantFromCovariant
export covariant_basis, contravariant_basis, transform_basis, jacobian
export gradB, curvatureProjection, curvatureComponents, metric
export AbstractMagneticSurface, MagneticSurface, AbstractMagneticFieldline, MagneticFieldline

include("MagneticCoordinates.jl")
include("DataTypes.jl")
include("Transformations.jl")
include("DerivedQuantities.jl")

# Specialzed coordinate defintions, transformations, routines for different codes
include("Specializations.jl")
end
