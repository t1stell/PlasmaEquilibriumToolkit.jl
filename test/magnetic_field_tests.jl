@testset "PlasmaEquilibriumToolkit magnetic_field tests " begin
  #generate a simple toroidal solenoidal field
  nr = 21
  nz = 26
  nθ = 36
  rtol = 1.0E-5
  r = collect(range(9,11,nr))
  θ = collect(range(0,2*π/3,nθ))
  z = collect(range(-1, 1, nz))
  nfp = 3
  fullSize = (length(r), length(θ), length(z))
  r_grid = reshape(repeat(r,outer=length(z)*length(θ)),fullSize)
  θ_grid = reshape(repeat(θ,inner=length(r),outer=length(z)),fullSize)
  z_grid = reshape(repeat(z,inner=length(r)*length(θ)),fullSize)
  Bf_coords = StructArray{Cylindrical}((r_grid, θ_grid, z_grid))
  Br = zeros(size(r_grid))
  Bz = zeros(size(r_grid))
  Bθ = reshape(repeat(10.0./r,outer=length(z)*length(θ)),fullSize)
  testfield = PlasmaEquilibriumToolkit.MagneticField(Bf_coords, Br, Bθ, Bz, nfp=3)
  
  @testset "Load MagneticField" begin
    @test size(testfield.coords.r) == fullSize
    @test length(testfield.field_data) == 3
    @test minimum(testfield.coords.r) == 9.0
    @test maximum(testfield.coords.r) == 11.0
    @test minimum(testfield.coords.z) == -1.0
    @test maximum(testfield.coords.z) == 1.0
    @test minimum(testfield.coords.θ) == 0.0
    @test maximum(testfield.coords.θ) == 2*π/3
    @test isapprox(testfield(10.0,0.0,0.0)[1],0.0,rtol=rtol)
    @test isapprox(testfield(10.0,0.0,0.0)[2],1.0,rtol=rtol)
    @test isapprox(testfield(10.0,0.0,0.0)[3],0.0,rtol=rtol)
    @test isapprox(testfield(10.0,Float64(π), 0.0)[2],1.0,rtol=rtol)
    @test isapprox(testfield(10.05,0.0,0.0)[2],0.995024514288483,rtol=rtol)
  end
end


