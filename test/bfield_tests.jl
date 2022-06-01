@testset "PlasmaEquilibriumToolkit Bfield tests " begin
  #generate a simple toroidal solenoidal field
  @testset "Load Bfield" begin
    nr = 21
    nz = 26
    nθ = 36
    rtol = 1.0E-5
    r = collect(range(9,11,nr))
    z = collect(range(-1, 1, nz))
    θ = collect(range(0,2*π/3,nθ))
    nfp = 3
    fullSize = (length(r), length(z), length(θ))
    r_grid = reshape(repeat(r,outer=length(z)*length(θ)),fullSize)
    z_grid = reshape(repeat(z,inner=length(r),outer=length(θ)),fullSize)
    θ_grid = reshape(repeat(θ,inner=length(z)*length(r)),fullSize)
    coords = StructArray{Cylindrical}((r_grid, z_grid, θ_grid))
    Br = zeros(size(r_grid))
    Bz = zeros(size(r_grid))
    Bθ = reshape(repeat(10.0./r,outer=length(z)*length(θ)),fullSize)
    testfield = PlasmaEquilibriumToolkit.BField(coords, Br, Bz, Bθ, nfp=3)
    @test size(testfield.coords.r) == fullSize
    @test length(testfield.field_data) == 3
    @test isapprox(testfield(10.0,0.0,0.0)[3],1.0,rtol=rtol)
    @test isapprox(testfield(10.0,0.0,Float64(π))[3],1.0,rtol=rtol)
  end
end


