@testset "PlasmaEquilibriumToolkit - Surface tests" begin
  #ensure Fourier series can load
  @testset "Load Fourier Arrays" begin
    m = [0, 1];
    n = [0, 0];
    rcos = [10.0, 1.0];
    zsin = [0.0, 1.0];
    dummy = [0.0, 0.0];
    rmn = StructArray{SurfaceFourierData{Float64}}((m, n, rcos, dummy, dummy, dummy));
    zmn = StructArray{SurfaceFourierData{Float64}}((m, n, dummy, zsin, dummy, dummy));
    @test length(rmn) == 2;
    @test length(zmn) == 2;
  end

  @testset "Generate Surface" begin
    testsurf = FourierSurface{Float64}(rmn, zmn, 1.0);
    @test testsurf.rmn == rmn;
    @test testsurf.zmn == zmn;
  end
end
