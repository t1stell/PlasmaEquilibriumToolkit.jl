@testset "PlasmaEquilibriumToolkit - Surface tests" begin
  m = [0, 1];
  n = [0, 0];
  rcos = [10.0, 1.0];
  zsin = [0.0, 1.0];
  dummy = [0.0, 0.0];
  rmn = StructArray{SurfaceFourierData{Float64}}((m, n, rcos, dummy, dummy, dummy));
  zmn = StructArray{SurfaceFourierData{Float64}}((m, n, dummy, zsin, dummy, dummy));
  fc = FourierCoordinates(1.0, 0.0, π)
  fc2 = FourierCoordinates(1.0, π/2, π)
  #ensure Types can load
  @testset "Load Fourier Arrays" begin
    @test length(rmn) == 2;
    @test length(zmn) == 2;

    @test fc.s == 1.0
    @test fc.θ == 0.0
    @test fc.ζ == Float64(π)
  end

  testsurf = FourierSurface{Float64}(rmn, zmn, 1.0);
  @testset "Generate Surface" begin
    @test testsurf.rmn == rmn;
    @test testsurf.zmn == zmn;
  end

  @testset "Calculate Derivative" begin
    derivs = transform_deriv(CylindricalFromFourier(),fc,testsurf)
    derivscheck = @SMatrix [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
    @test derivs==derivscheck
    derivs2 = transform_deriv(CylindricalFromFourier(),fc2,testsurf)
    @test derivs2[1,2] == -1.0

  end
end
