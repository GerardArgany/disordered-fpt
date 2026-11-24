const GERARD_ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(GERARD_ROOT, "data")
const FIGURES_DIR = joinpath(GERARD_ROOT, "figures")

function figure_dir(parts...)
    path = joinpath(FIGURES_DIR, parts...)
    mkpath(path)
    return path
end
