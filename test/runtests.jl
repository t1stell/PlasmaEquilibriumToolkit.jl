#using PlasmaEquilibriumToolkit
using Test
import Combinatorics

@testset "PlasmaEquilibriumToolkit.jl" begin
  # Ensure the magnetic coordinate types are working
  @testset "MagneticCoordinates" begin
    for c in (
      ClebschCoordinates,
      FluxCoordinates,
      PestCoordinates,
      BoozerCoordinates,
    )
      @test typeof(c(0.0, 0.0, 0.0)) == c{Float64,Float64}
      @test typeof(c(1, 0.0, 0)) == c{Float64,Float64}
      @test typeof(c(π, π, π)) == c{Irrational{:π},Irrational{:π}}
      @test typeof(c(0, 0, 1)) == c{Int64,Int64}
    end
  end


  # Test the MagneticCoordinateGrid functionality
  @testset "Vector" begin
    c = ClebschCoordinates
    coords = (:α, :β, :η)
    π_float = convert(Float64, π)
    for i = 1:3
      for j = 1:3
        if i == j
          eval(Expr(:(=), coords[j], 0:2π/8:2π))
        else
          eval(Expr(:(=), coords[j], 0.0))
        end
      end
      c_array = MagneticCoordinateGrid(c, eval.(coords)...)

      @test typeof(c_array) == StructVector{c{Float64,Float64}}
      @test length(c_array) == 9
      @test c_array[5] == c{Float64,Float64}(
        i == 1 ? π_float : 0.0,
        i == 2 ? π_float : 0.0,
        i == 3 ? π_float : 0.0,
      )
    end
  end
    #=
    @testset "Vector_2args" begin
      c = ClebschCoordinates
      coords = [:α, :β, :η]
      π_float = convert(Float64, π)
      coord_perms = Combinatorics.permutations(coords, 2)

      for p in coord_perms
        leftover = setdiff(coords, p)[]
        for cp in p
          eval(Expr(:(=), cp, 0:2π/8:2π))
        end
        eval(Expr(:(=), leftover, 0.0))
        c_array = MagneticCoordinateArray(c, eval.(coords)...)

        @test typeof(c_array) == Array{c{Float64,Float64},2}
        @test size(c_array) == (9, 9)
        @test c_array[5, 5] == c{Float64,Float64}(
          :α in p ? π_float : 0.0,
          :β in p ? π_float : 0.0,
          :η in p ? π_float : 0.0,
        )
      end

      for p in coord_perms
        leftover = setdiff(coords, p)[]
        for cp in p
          eval(Expr(:(=), cp, 0:2π/8:2π))
        end
        eval(Expr(:(=), leftover, 0.0))
        c_array = MagneticCoordinateArray(c, eval.(coords)..., grid = false)

        @test typeof(c_array) == Array{c{Float64,Float64},1}
        @test length(c_array) == 9
        @test c_array[5] == c{Float64,Float64}(
          :α in p ? π_float : 0.0,
          :β in p ? π_float : 0.0,
          :η in p ? π_float : 0.0,
        )

        # Check for incompatible dimensions
        for (i, cp) in enumerate(p)
          eval(Expr(:(=), cp, 0:2π/(8+iseven(i)):2π))
        end
        @test_throws DimensionMismatch MagneticCoordinateArray(
          c,
          eval.(coords)...,
          grid = false,
        )
      end

      @testset "Vector_3args" begin
        c = ClebschCoordinates
        α = 0:0.2:1
        β = 0:2π/4:2π
        η = 0:2π/8:2π
        π_float = convert(Float64, π)

        c_array = MagneticCoordinateArray(c, α, β, η, grid = true)
        @test typeof(c_array) == Array{c{Float64,Float64},3}
        @test size(c_array) == (9, 6, 5)
        @test c_array[5, 3, 3] == c{Float64,Float64}(0.4, π_float, π_float)

        c_array =
          MagneticCoordinateArray(c, α, β, η, grid = true, keep_order = true)
        @test size(c_array) == (6, 5, 9)
        @test c_array[3, 3, 5] == c{Float64,Float64}(0.4, π_float, π_float)

        c_array =
          MagneticCoordinateArray(c, α, β, η, grid = true, dim_order = [2, 3, 1])
        @test size(c_array) == (5, 9, 6)
        @test c_array[3, 5, 3] == c{Float64,Float64}(0.4, π_float, π_float)

        @test_throws DimensionMismatch MagneticCoordinateArray(
          c,
          α,
          β,
          η,
          grid = false,
        )
        β = η
        @test_throws DimensionMismatch MagneticCoordinateArray(
          c,
          α,
          β,
          η,
          grid = false,
        )
        α = η

        c_array = MagneticCoordinateArray(c, α, β, η, grid = false)
        @test typeof(c_array) == Array{c{Float64,Float64},1}
        @test length(c_array) == 9
        @test c_array[5] == c{Float64,Float64}(π_float, π_float, π_float)
      end
    end
    =#
end
