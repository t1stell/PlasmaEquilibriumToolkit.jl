@testset "PlasmaEquilibriumToolkit - Magnetic Coordinate Tests" begin
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
    coord_perms = unique(sort.(Combinatorics.permutations(coords, 2)))

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

  @testset "MagneticCoordinateGrid - 2 Vector Arguments" begin
    c = ClebschCoordinates
    coords = [:α, :β, :η]
    π_float = convert(Float64, π)
    coord_perms = unique(sort.(Combinatorics.permutations(coords, 2)))

    for p in coord_perms
      leftover = setdiff(coords, p)[]
      for (i, cp) in enumerate(p)
        eval(Expr(:(=), cp, 0:2π/(8-2*(i-1)):2π))
      end
      eval(Expr(:(=), leftover, 0.0))
      c_array = MagneticCoordinateGrid(c, eval.(coords)...)

      @test size(c_array) == (7, 9)
      @test c_array[4, 5] == c{Float64,Float64}(
        :α in p ? π_float : 0.0,
        :β in p ? π_float : 0.0,
        :η in p ? π_float : 0.0,
      )
    end

    @test_throws DimensionMismatch MagneticCoordinateGrid(
      c,
      eval.(coords)...,
      grid = false,
    )
  end

  @testset "MagneticCoordinateGrid - 2 Matrix Arguments" begin
    c = ClebschCoordinates
    coords = [:α, :β, :η]
    π_float = convert(Float64, π)
    coord_perms = unique(sort.(permutations(coords, 2)))

    for p in coord_perms
      leftover = setdiff(coords, p)[]
      for (i, cp) in enumerate(p)
        if i == 1
          eval(Expr(:(=), cp, reshape(repeat(0:2π/4:2π, inner = 9), (9, 5))))
        else
          eval(Expr(:(=), cp, repeat(0:2π/8:2π, outer = (1, 5))))
        end
      end
      eval(Expr(:(=), leftover, 0.0))
      c_array = MagneticCoordinateGrid(c, eval.(coords)...)

      @test size(c_array) == (9, 5)
      @test c_array[5, 3] == c{Float64,Float64}(
        :α in p ? π_float : 0.0,
        :β in p ? π_float : 0.0,
        :η in p ? π_float : 0.0,
      )

      for (i, cp) in enumerate(p)
        if i == 1
          eval(Expr(:(=), cp, reshape(repeat(0:2π/4:2π, inner = 7), (7, 5))))
        else
          eval(Expr(:(=), cp, repeat(0:2π/8:2π, outer = (1, 5))))
        end
      end
      eval(Expr(:(=), leftover, 0.0))
      @test_throws DimensionMismatch MagneticCoordinateGrid(c, eval.(coords)...)
    end
  end

  @testset "MagneticCoordinateGrid - 1 Vector, 2 Matrix Arguments" begin
    c = ClebschCoordinates
    coords = [:α, :β, :η]
    π_float = convert(Float64, π)
    coord_perms = unique(sort.(permutations(coords, 2)))

    for p in coord_perms
      leftover = setdiff(coords, p)[]
      for (i, cp) in enumerate(p)
        if i == 1
          eval(Expr(:(=), cp, reshape(repeat(0:2π/4:2π, inner = 9), (9, 5))))
        else
          eval(Expr(:(=), cp, repeat(0:2π/8:2π, outer = (1, 5))))
        end
      end
      eval(Expr(:(=), leftover, 0:2π/6:2π))
      c_array = MagneticCoordinateGrid(c, eval.(coords)...)

      @test prod(size(c_array)) == 315
      α_vec = typeof(eval(:α)) <: AbstractRange ? true : false
      β_vec = typeof(eval(:β)) <: AbstractRange ? true : false
      η_vec = typeof(eval(:η)) <: AbstractRange ? true : false
      c_tuple = (η_vec ? 4 : 5, β_vec ? 4 : α_vec ? 3 : 5, α_vec ? 4 : 3)
      @test c_array[c_tuple...] == c{Float64,Float64}(π_float, π_float, π_float)

      for (i, cp) in enumerate(p)
        if i == 1
          eval(Expr(:(=), cp, reshape(repeat(0:2π/4:2π, inner = 7), (7, 5))))
        else
          eval(Expr(:(=), cp, repeat(0:2π/8:2π, outer = (1, 5))))
        end
      end
      eval(Expr(:(=), leftover, 0:2π/6:2π))
      @test_throws DimensionMismatch MagneticCoordinateGrid(c, eval.(coords)...)
    end
  end

  @testset "MagneticCoordinateGrid - 3 Matrix Arguments" begin
    c = ClebschCoordinates
    π_float = convert(Float64, π)
    α = fill(0.5, (9, 5))
    β = reshape(repeat(0:2π/4:2π, inner = 9), (9, 5))
    η = repeat(0:2π/8:2π, outer = (1, 5))

    c_array = MagneticCoordinateGrid(c, α, β, η)

    @test size(c_array) == (9, 5)
    @test c_array[5, 3] == c{Float64,Float64}(0.5, π_float, π_float)

    β = reshape(repeat(0:2π/4:2π, inner = 7), (7, 5))
    @test_throws DimensionMismatch MagneticCoordinateGrid(c, α, β, η)
  end

  @testset "MagneticCoordinateGrid - 3 3D Array Arguments" begin
    c = ClebschCoordinates
    π_float = convert(Float64, π)
    α = reshape(repeat(0:2π/6:2π, inner = 45), (9, 5, 7))
    β = repeat(reshape(repeat(0:2π/4:2π, inner = 9), (9, 5)), outer = (1, 1, 7))
    η = repeat(0:2π/8:2π, outer = (1, 5, 7))

    c_array = MagneticCoordinateGrid(c, α, β, η)

    @test size(c_array) == (9, 5, 7)
    @test c_array[5, 3, 4] == c{Float64,Float64}(π_float, π_float, π_float)

    β = repeat(reshape(repeat(0:2π/4:2π, inner = 7), (7, 5)), outer = (1, 1, 7))
    @test_throws DimensionMismatch MagneticCoordinateGrid(c, α, β, η)
  end
end
