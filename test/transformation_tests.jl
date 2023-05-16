@testset "PlasmaEquilibriumToolkit - Transformation Tests" begin
  @testset "Vector abs" begin
    test_vec = BasisVectors{Float64}(reshape(1.0:9.0, (3, 3)))
    @test abs(test_vec) ≈ 16.881943016134134
    @test abs(test_vec, 0) ≈ 16.881943016134134
    @test abs(test_vec, 4) ≈ 16.881943016134134
    @test abs(test_vec, 1) ≈ 3.7416573867739413
    @test abs(test_vec, 2) ≈ 8.774964387392123
    @test abs(test_vec, 3) ≈ 13.92838827718412
  end

  @testset "Jacobian" begin
    test_vec = BasisVectors{Float64}(reshape(1.0:9.0, (3, 3)))
    @test jacobian(Covariant(), test_vec) == zero(Float64)
    @test isinf(jacobian(Contravariant(), test_vec))
    test_vec = @SArray([1.0 3.0 2.0; 9.0 5.0 4.0; 8.0 6.0 7.0])
    @test jacobian(Contravariant(), test_vec) ≈ -0.018518518518518517
  end

  @testset "Basis Transformations" begin
    test_basis = @SArray([1.0 3.0 2.0; 9.0 5.0 4.0; 8.0 6.0 7.0])
    jac = jacobian(Covariant(), test_basis)
    ans = @SArray[
      -0.2037037037037037 0.5740740740740741 -0.25925925925925924
      0.16666666666666666 0.16666666666666666 -0.3333333333333333
      -0.037037037037037035 -0.25925925925925924 0.4074074074074074
    ]
    @test prod(
      transform_basis(ContravariantFromCovariant(), test_basis, jac) .≈ ans,
    )
    @test transform_basis(ContravariantFromCovariant(), test_basis) ==
          transform_basis(ContravariantFromCovariant(), test_basis, jac)
    jac = jacobian(Contravariant(), test_basis)
    @test prod(
      transform_basis(CovariantFromContravariant(), test_basis, jac) .≈ ans,
    )
    @test transform_basis(CovariantFromContravariant(), test_basis) ==
          transform_basis(CovariantFromContravariant(), test_basis, jac)

    seed = 1948923
    rng = ARS1x(seed)
    vecs = [BasisVectors{Float64}(rand(rng, 9)) for i = 1:256]
    jacs = [jacobian(Covariant(), v) for v in vecs]
    @test transform_basis(ContravariantFromCovariant(), vecs) .==
          transform_basis(ContravariantFromCovariant(), vecs, jacs)
    jacs = [jacobian(Contravariant(), v) for v in vecs]
    @test transform_basis(CovariantFromContravariant(), vecs) .==
          transform_basis(CovariantFromContravariant(), vecs, jacs)
  end
end
