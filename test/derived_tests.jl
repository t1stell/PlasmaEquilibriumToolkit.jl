@testset "Derived Quantity Tests" begin
    seed = 1348548
    rng = ARS1x(seed)
    B_vec = BasisVectors{Float64}(0.8 .+ rand(rng, 9))
    ι = 1.0 + rand(rng)
    dpdψ = -1.1E5 
    ∇X = BasisVectors{Float64}(0.5 .+ rand(rng, 9))
    ∇B = CoordinateVector{Float64}(rand(rng, 3))
    @testset "B_norm" begin
        @test isapprox(B_norm(Contravariant(), B_vec), 0.9589876106776953)
        @test_throws ArgumentError B_norm(Covariant(), B_vec)
        @test isapprox(B_norm(Covariant(), B_vec, ι), 7.47127616783484)
        @test B_norm(B_vec) == B_norm(Contravariant(), B_vec)
    end

    @testset "B_field" begin
        @test prod(
            isapprox.(
                B_field(Contravariant(), B_vec),
                SVector{3,Float64}(
                    -0.7330182803485883,
                    0.6179562983133189,
                    0.021712933542331836,
                ),
            ),
        )
        @test_throws ArgumentError B_field(Covariant(), B_vec)
        @test prod(
            isapprox(
                B_field(Covariant(), B_vec, ι),
                SVector{3,Float64}(
                    -4.641973086777105,
                    -3.7295049271502645,
                    -4.512521073197973,
                ),
            ),
        )
        @test B_field(B_vec) == B_field(Contravariant(), B_vec)
    end

    @testset "curvatures" begin
        @test prod(isapprox.(
            curvature_components(∇X, ∇B),
                (1.1907558171184576, -0.8554865685407761, 6.6469252727084625)))
        @test prod(isapprox.(
            curvature_components(∇X, ∇B, dpdψ=dpdψ),
                (1.9755442095795814, -0.8554865685407761, 7.431713665169586)))
    end

end
