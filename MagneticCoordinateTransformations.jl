module MagneticCoordinateTransformations

export MagneticEquilibrium, MagneticCoordinates
export FluxCoordinates, PestCoordinates, BoozerCoordinates
export FluxFromPest, PestFromFlux, CylindricalFromFlux, CylindricalFromPest
export CartesianFromFlux, CartesianFromPest
export Covariant, Contravariant
export CovariantFromContravariant, ContravariantFromCovariant, BasisVectors
export covariant_basis, contravariant_basis, transform_basis, jacobian

include("src/MagneticCoordinates.jl")
include("src/Transformations.jl")

end
