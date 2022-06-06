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

function coordinate_grid(::Type{Cylindrical},
                         r::AbstractVector{T1},
                         θ::AbstractVector{T2},
                         z::AbstractVector{T3};
                        ) where {T1, T2, T3}
    full_size = (length(r), length(θ), length(z))
    r_grid = reshape(repeat(r, outer=length(θ)*length(z)), full_size)
    θ_grid = reshape(repeat(θ, inner=length(r), outer=length(z)), full_size)
    z_grid = reshape(repeat(z, inner=length(r)*length(θ)), full_size)
    return StructArray{Cylindrical}((r_grid, θ_grid, z_grid))
end

function MagneticField(coords::StructArray{Cylindrical},
                        field_data_r::AbstractArray{T},
                        field_data_ϕ::AbstractArray{T},
                        field_data_z::AbstractArray{T};
                        bc = Periodic,
                        nfp = 1,
                        ) where {T}

    println("rsize: ",size(field_data_r))
    println("coordsize: ",size(coords))
    size(field_data_r) == size(field_data_z) == size(field_data_ϕ) == size(coords) || throw(DimensionMismatch("Incompatible arrays sizes"))
    knots_dim_1 = getproperty(coords[:, 1, 1], 1)
    knots_dim_2 = getproperty(coords[1, :, 1], 2)
    knots_dim_3 = getproperty(coords[1, 1, :], 3)

    knots = (vector2range(knots_dim_1), vector2range(knots_dim_2), vector2range(knots_dim_3))
    itp_types = (BSpline(Cubic(Free(OnGrid()))),
                 BSpline(Cubic(Periodic(OnCell()))),
                 BSpline(Cubic(Free(OnGrid()))))
    itp = (f) -> scale(interpolate(f, itp_types), knots...)
    extp = (f) -> extrapolate(itp(f), (Throw(), Periodic(), Throw()))

    Br = extp(field_data_r)
    Bz = extp(field_data_z)
    Bϕ = extp(field_data_ϕ)

    return MagneticField{T, Cylindrical, Nothing}(nfp, coords, (Br, Bϕ, Bz), nothing)
end

function MagneticField(coords::StructArray{Cylindrical},
                        field_data_r::AbstractArray{T},
                        field_data_ϕ::AbstractArray{T},
                        field_data_z::AbstractArray{T},
                        potential_data_r::AbstractArray{T},
                        potential_data_ϕ::AbstractArray{T},
                        potential_data_z::AbstractArray{T};
                        bc = Periodic,
                        nfp = 1,
                        ) where {T}
    size(field_data_r) == size(field_data_z) == size(field_data_ϕ) == size(coords) || throw(DimensionMismatch("Incompatible arrays sizes"))
    knots_dim_1 = getproperty(coords[:, 1, 1], 1)
    knots_dim_2 = getproperty(coords[1, :, 1], 2)
    knots_dim_3 = getproperty(coords[1, 1, :], 3)

    knots = (vector2range(knots_dim_1), vector2range(knots_dim_2), vector2range(knots_dim_3))
    itp_types = (BSpline(Cubic(Free(OnGrid()))),
                 BSpline(Cubic(Periodic(OnCell()))),
                 BSpline(Cubic(Free(OnGrid()))))
    itp = (f) -> scale(interpolate(f, itp_types), knots...)
    extp = (f) -> extrapolate(itp(f), (Throw(), Periodic(), Throw()))

    Br = extp(field_data_r)
    Bz = extp(field_data_z)
    Bϕ = extp(field_data_ϕ)

    Ar = extp(potential_data_r)
    Az = extp(potential_data_z)
    Aϕ = extp(potential_data_ϕ)

    return MagneticField{T, Cylindrical, WithPotential}(nfp, coords, (Br, Bϕ, Bz), (Ar, Aϕ, Az))
end

function MagneticField(coords::StructArray{Cylindrical{T, A}},
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
    return MagneticField(coords, field_data_x, field_data_y, field_data_z; bc = bc, nfp = nfp)
end

function (magnetic_field::MagneticField{F, C})(x::T,
                                y::T,
                                z::T;
                               ) where {T, F, C}
    Bx = magnetic_field.field_data[1](x, y, z)
    By = magnetic_field.field_data[2](x, y, z)
    Bz = magnetic_field.field_data[3](x, y, z)
    return (Bx, By, Bz)
end

function (magnetic_field::MagneticField{F, C, Nothing})(r::T,
                                ϕ::T,
                                z::T;
                               ) where {F, T, C <: Cylindrical}
    ϕ = mod(ϕ, 2*π/magnetic_field.nfp)
    Br = magnetic_field.field_data[1](r,ϕ,z)
    Bϕ = magnetic_field.field_data[2](r,ϕ,z)
    Bz = magnetic_field.field_data[3](r,ϕ,z)

    return (Br, Bϕ, Bz)
end

function (magnetic_field::MagneticField{F, C, WithPotential})(r::T,
                                ϕ::T,
                                z::T;
                               ) where {F, T, C <: Cylindrical}
    ϕ = mod(ϕ, 2*π/magnetic_field.nfp)
    Br = magnetic_field.field_data[1](r,ϕ,z)
    Bϕ = magnetic_field.field_data[2](r,ϕ,z)
    Bz = magnetic_field.field_data[3](r,ϕ,z)

    Ar = magnetic_field.potential_data[1](r,ϕ,z)
    Aϕ = magnetic_field.potential_data[2](r,ϕ,z)
    Az = magnetic_field.potential_data[3](r,ϕ,z)

    return (Br, Bϕ, Bz), (Ar, Aϕ, Az)
end
