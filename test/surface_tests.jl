@testset "PlasmaEquilibriumToolkit - Surface tests" begin
  m = [0, 1];
  n = [0, 0];
  rcos = [10.0, 1.0];
  zsin = [0.0, 1.0];
  dummy = [0.0, 0.0];
  rmn = StructArray{SurfaceFourierData{Float64}}((m, n, rcos, dummy, dummy, dummy, dummy, dummy));
  zmn = StructArray{SurfaceFourierData{Float64}}((m, n, dummy, zsin, dummy, dummy, dummy, dummy));
  fc = FourierCoordinates(1.0, 0.0, π)
  fc2 = FourierCoordinates(1.0, π/2, π)
  rtol = 1.0E-4
  #ensure Types can load
  @testset "Load Fourier Arrays" begin
    @test length(rmn) == 2;
    @test length(zmn) == 2;

    @test fc.s == 1.0
    @test fc.θ == 0.0
    @test fc.ζ == Float64(π)
  end

  testsurf = FourierSurface(rmn, zmn, 1.0, 1);
  @testset "Generate Surface" begin
    @test testsurf.rmn == rmn;
    @test testsurf.zmn == zmn;
  end

  @testset "Calculate Derivative" begin
    derivs = transform_deriv(CylindricalFromFourier(),fc,testsurf)
    derivscheck = @SMatrix [0.0 0.0 0.0; 0.0 0.0 1.0; 0.0 1.0 0.0]
    @test isapprox(derivs, derivscheck, rtol=rtol)
    derivs2 = transform_deriv(CylindricalFromFourier(),fc2,testsurf)
    @test isapprox(derivs2[1,2], -1.0, rtol=rtol)
    #check cross product
    n1 = normal_vector(fc, testsurf)
    nt = @SVector [1.0, 0.0, 0.0]
    @test isapprox(n1, nt, rtol=rtol)
    n2 = normal_vector(fc2, testsurf)
    nt = @SVector [0.0, 1.0, 0.0]
    @test isapprox(n2, nt, rtol=rtol)
  end

  @testset "test inside surface" begin
    for r in range(10.0, 12.0, 10)
      cc = Cylindrical(r, 0.0, 0.0)
      testval = in_surface(cc, testsurf)
      xyz = SVector(r, 0.0, 0.0)
      testval2 = in_surface(xyz, testsurf)
      if r > 11.0
        @test testval == false
        @test testval2 == false
      else
        @test testval == true
        @test testval2 == true
      end
    end
    xw = [0.0, 1.0, 1.0, 0.0, 0.0]
    yw = [0.0, 0.0, 1.0, 1.0, 0.0]
    @test in_surface(0.5, 0.5, xw, yw)
    @test !in_surface(-0.5, 0.5, xw, yw)

  end
  
end
