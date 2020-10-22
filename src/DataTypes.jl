using StaticArrays

const CoordinateVector{T} = SVector{3,T} where T
const BasisVectors{T} = SArray{Tuple{3,3},T,2,9} where T
