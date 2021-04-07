using PlasmaTurbulenceSaturationModel
using Documenter

makedocs(;
    modules=[PlasmaTurbulenceSaturationModel],
    authors="Benjamin Faber <bfaber@wisc.edu> and contributors",
    repo="https://gitlab.com/bfaber/PlasmaTurbulenceSaturationModel.jl/blob/{commit}{path}#L{line}",
    sitename="PlasmaTurbulenceSaturationModel.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://bfaber.gitlab.io/PlasmaTurbulenceSaturationModel.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
