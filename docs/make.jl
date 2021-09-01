using PlasmaEquilibriumToolkit
using Documenter

makedocs(;
    modules=[PlasmaEquilibriumToolkit],
    authors="Benjamin Faber <bfaber@wisc.edu> and contributors",
    repo="https://gitlab.com/wistell/PlasmaEquilibriumToolkit.jl/blob/{commit}{path}#L{line}",
    sitename="PlasmaEquilibriumToolkit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://wistell.gitlab.io/PlasmaEquilibriumToolkit.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
