# PlasmaEquilibriumToolkit.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://wistell.gitlab.io/PlasmaEquilibriumToolkit.jl/dev)
[![Build Status](https://gitlab.com/wistell/PlasmaEquilibriumToolkit.jl/badges/master/pipeline.svg)](https://gitlab.com/wistell/PlasmaEquilibriumToolkit.jl/pipelines)
[![Coverage](https://gitlab.com/wistell/PlasmaEquilibriumToolkit.jl/badges/master/coverage.svg)](https://gitlab.com/wistell/PlasmaEquilibriumToolkit.jl/commits/master)

A Julia package for working with 3D magnetic fields, coordinate systems and equilibria for plasma physics applications.
`PlasmaEquilibriumToolkit.jl` works closely with [CoordinateTransformations.jl](https://github.com/JuliaGeometry/CoordinateTransformations.jl) to efficiently execute change of basis transformations for 3D curvilinear coordinate systems defined by [magnetic coordinates](https://arxiv.org/abs/1904.01682) using data from ideal MHD equilibrium solvers such as [VMEC](https://github.com/ORNL-Fusion/PARVMEC).
The magnetic coordinate system is defined by the divergence-free form of the magnetic field: $`\mathbf{B}(\alpha, \beta, \eta) = \nabla \alpha \times \nabla \beta(\eta)`$, where $`\alpha`$ defines a *magnetic flux surface* and $`\beta`$ labels the *magnetic field line*.
The following magnetic coordinates are predefined by `PlasmaEquilibriumToolkit.jl`:
  - `ClebschCoordinates`: $`\mathbf{B}(\alpha, \beta, \eta) = \nabla \alpha \times \nabla \beta(\eta)`$, $`\alpha`$ and $`\beta`$ are arbitrary labels.
  - `FluxCoordinates`: $`\mathbf{B}(\psi, \theta, \zeta) = \nabla \psi \times \nabla (\theta - \iota \zeta`$, where $`\psi`$ is the signed *toroidal* magnetic flux label, $`\theta \in [0,2\pi)`$ is an angle like variable measuring the *poloidal* angle increasing positively over the top of the torus when viewed from above, $`\zeta`$ is the *toroidal* angle increasing positively in the counterclockwise direction, and $`\iota`$ is the rotational transform.
  - `PestCoordinates`: $`\mathbf{B}(\psi, \alpha, \zeta) = \nabla \psi \times \nabla \alpha`$, where $`\psi`$ is the signed *toroidal* magnetic flux label, $`\alpha \in [0,2\pi)`$ labels the magnetic field line, $`\zeta`$ is the *toroidal* angle increasing positively in the counterclockwise direction.
  - `BoozerCoordinates`: $`\mathbf{B}(\psi, \chi, \phi) = \iota \nabla \phi \times \nabla \psi - \nabla \chi \times \nabla \psi`$, where $`\psi`$ is the *toroidal* magnetic label, and the coordinates $`\chi`$ and $`\phi`$ are chosen such that the Jacobian of the coordinate system is constant, $`J = \frac{1}{\nabla \psi \cdot \nabla \chi \times \nabla \phi} = \frac{1}{B^2}`$.
