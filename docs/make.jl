using UnitCellMC
using Documenter

makedocs(
    sitename = "UnitCellMC.jl",
    modules = [UnitCellMC],
    remotes = nothing,

    pages = [
        "Home" => "index.md",

        "Getting Started" => "getting_started.md",

        "Tutorials" => [
            "1D Ising Model" => "tutorials/ising_1d.md",
            "2D Ising Model" => "tutorials/ising_2d.md",
        ],

        "Concepts" => [
            "Geometry" => "concepts/geometry.md",
            "Models" => "concepts/models.md",
            "State" => "concepts/state.md",
            "Algorithms" => "concepts/algorithms.md",
            "Simulation" => "concepts/simulation.md",
            "Measurements" => "concepts/measurements.md",
        ],

        "API Reference" => [
            "Models" => "api/models.md",
            "Algorithms" => "api/algorithms.md",
            "Geometry" => "api/geometry.md",
            "Simulation" => "api/simulation.md",
            "Measurements" => "api/measurements.md",
        ],

        "TODO / In Progress" => "todo_in_progress.md",
    ],
)