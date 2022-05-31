"""
vector2range(v)

Convert an evenly spaced vector points to a range
"""
function vector2range(v::AbstractVector)
  if typeof(v) <: AbstractRange
    return v
  else
    v_diff = @view(v[2:end]) .- @view(v[1:end-1])
    v_diff_mean = sum(v_diff)/length(v_diff)
    all(v_diff .≈ v_diff_mean) || throw(DimensionMismatch("Step between vector elements must be consistent"))
    return range(v[1],step=v_diff_mean,length=length(v))
  end
end

function BField(coords::StructArray{Cylindrical{T, A}},
                field_data_x::AbstractArray{T, 3},
                field_data_y::AbstractArray{T, 3},
                field_data_z::AbstractArray{T, 3};
                bc::BoundaryCondition = Periodic,
                nfp::Int = 1,
               ) where {T, A}
    size(field_data_x) == size(field_data_y) == size(field_data_z) == size(coords) || throw(DimensionMismatch("Incompatible arrays sizes"))
    knots_dim_1 = coords[:, 1, 1]
    knots_dim_2 = coords[1, :, 1]
    knots_dim_3 = coords[1, 1, :]

    knots = (vector2range(knots_dim_1), vector2range(knots_dim_2), vector2range(knots_dim_3))

    itp_x = CubicSplineInterpolation(knots, field_data_x; bc = bc(OnGrid()))
    itp_y = CubicSplineInterpolation(knots, field_data_y; bc = bc(OnGrid()))
    itp_z = CubicSplineInterpolation(knots, field_data_z; bc = bc(OnGrid()))

    return BField{T, Cylindrical, bc}(nfp, coords, (itp_x, itp_y, itp_z))
end

function BField(coords::StructArray{Cylindrical{T, A}},
                data::AbstractArray{BasisVectors{T}};
                bc::BoundaryCondition = Periodic,
                nfp::Int = 1) where {T, A}
    size(coords) == size(data) || throw(DimensionMismatch("The coordinates array and field data array must have compatible sizes."))
    field_data_x = Array{T, 3}(undef, size(coords))
    field_data_y = Array{T, 3}(undef, size(coords))
    field_data_z = Array{T, 3}(undef, size(coords))
    for i in eachindex(field_data_x, field_data_y, field_data_z, data)
        field_data_x[i] = data[i][:, 1]
        field_data_y[i] = data[i][:, 2]
        field_data_z[i] = data[i][:, 3]
    end
    return BField(coords, field_data_x, field_data_y, field_data_z; bc = bc, nfp = nfp)
end

function (bfield::BField{F, C, B})(x::T,
                                   y::T,
                                   z::T;
                                  ) where {T, F, C <: Cartesian, B}
    Bx = bfield.field_data[1](x, y, z)
    By = bfield.field_data[2](x, y, z)
    Bz = bfield.field_data[3](x, y, z)
    return (Bx, By, Bz)
end

function (bfield::BField{F, C, B})(r::T,
                                   z::T,
                                   ϕ::T;
                                  ) where {F, T, C <: Cylindrical, B <: Periodic}
    ϕ = mod(ϕ, 2*π/bfield.nfp)
    Br = bfield.field_data[1](r,z,ϕ)
    Bz = bfield.field_data[2](r,z,ϕ)
    Bϕ = bfield.field_data[3](r,z,ϕ)

    return (Br, Bz, Bϕ)
end