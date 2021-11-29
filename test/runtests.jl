using PlasmaEquilibriumToolkit
using Combinatorics
using StructArrays
using Test

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


  # Test the MagneticCoordinateCurve functionality
  @testset "MagneticCoordinateCurve - 1 Vector Argument" begin
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
      c_array = MagneticCoordinateCurve(c, eval.(coords)...)

      @test length(c_array) == 9
      @test c_array[5] == c{Float64,Float64}(
        i == 1 ? π_float : 0.0,
        i == 2 ? π_float : 0.0,
        i == 3 ? π_float : 0.0,
      )
    end
  end
  @testset "MagneticCoordinateCurve - 2 Vector Arguments" begin
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
      c_array = MagneticCoordinateCurve(c, eval.(coords)...)

      @test size(c_array) == (9,)
      @test c_array[5] == c{Float64,Float64}(
        :α in p ? π_float : 0.0,
        :β in p ? π_float : 0.0,
        :η in p ? π_float : 0.0,
      )
    end
  end

  @testset "MagneticCoordinateCurve - 3 Vector Arguments" begin
    c = ClebschCoordinates
    α = 0:0.2:1
    β = 0:2π/4:2π
    η = 0:2π/8:2π
    π_float = convert(Float64, π)

    @test_throws DimensionMismatch MagneticCoordinateCurve(c, α, β, η)
    β = η
    α = fill(0.5, length(η))
    c_array = MagneticCoordinateCurve(c, α, β, η)
    @test size(c_array) == (9,)
    @test c_array[5] == c{Float64,Float64}(0.5, π_float, π_float)
  end
end
