ENV["GKSwstype"] = "100"
using Plots

figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

function frt_sim(weights::Matrix{Float64})
    position = 1
    t = 0

    n = size(weights, 1) + 1

    while true
        t += 1
        r = rand()
        if r < weights[position, 1]
            position += 1
        elseif r < weights[position, 1] + weights[position, 2]
            position -= 1
        else
            # Stay in place when the transition probabilities do not sum to one.
        end         

        if position == 0 || position == n
            return nothing
        elseif position == 1
            return t
        end
    end
end

function vm_weights(n::Int)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        transition_rate = i * (n - i) / n^2
        weights[i, :] .= transition_rate
    end
    return weights
end

function random_weights(n::Int)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        weights[i, 1] = rand()
        weights[i, 2] = 1.0 - weights[i, 1]
    end
    return weights
end

function random_stay_weights(n::Int)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        right_draw = -log(max(rand(), eps(Float64)))
        left_draw = -log(max(rand(), eps(Float64)))
        stay_draw = -log(max(rand(), eps(Float64)))
        normalization = right_draw + left_draw + stay_draw

        weights[i, 1] = right_draw / normalization
        weights[i, 2] = left_draw / normalization
    end
    return weights
end

function frt_samples(weights::Matrix{Float64}, n_runs::Int)
    samples = Int[]
    boundary_hits = 0

    for _ in 1:n_runs
        return_time = frt_sim(weights)
        if return_time === nothing
            boundary_hits += 1
        else
            push!(samples, return_time)
        end
    end

    return samples, boundary_hits
end

function log_binned_distribution(samples::Vector{Int}, n_bins::Int)
    minimum_time = minimum(samples)
    maximum_time = maximum(samples)
    edges = 10 .^ range(log10(minimum_time), log10(maximum_time), length=n_bins + 1)
    counts = zeros(Int, n_bins)

    for time in samples
        bin = clamp(searchsortedlast(edges, time), 1, n_bins)
        counts[bin] += 1
    end

    centers = sqrt.(edges[1:end-1] .* edges[2:end])
    widths = diff(edges)
    densities = counts ./ (length(samples) .* widths)
    occupied = counts .> 0

    return centers[occupied], densities[occupied]
end

n = 300
n_runs = 1_000_000
voter_weights = vm_weights(n)
return_samples, boundary_hits = frt_samples(voter_weights, n_runs)
return_times, return_densities = log_binned_distribution(return_samples, 40)

println("Voter-model first-return samples: ", length(return_samples))
println("Voter-model boundary hits: ", boundary_hits)
println("Fraction returning before touching a boundary: ", length(return_samples) / n_runs)

return_distribution = scatter(
    return_times,
    return_densities,
    xscale=:log10,
    yscale=:log10,
    markersize=3,
    markerstrokewidth=0,
    xlabel="First-return time, t",
    ylabel="Probability density",
    title="Voter-model first-return distribution (log-binned)",
    label="Log-binned empirical distribution",
    legend=:topright,
    grid=true,
)

savefig(return_distribution, joinpath(figures_dir, "voter_model_first_return_distribution_loglog.png"))

random_weight_matrix = random_weights(n)
random_samples, random_boundary_hits = frt_samples(random_weight_matrix, n_runs)
random_times, random_densities = log_binned_distribution(random_samples, 40)

println("Random-weight first-return samples: ", length(random_samples))
println("Random-weight boundary hits: ", random_boundary_hits)
println("Random-weight fraction returning before touching a boundary: ", length(random_samples) / n_runs)

random_return_distribution = scatter(
    random_times,
    random_densities,
    xscale=:log10,
    yscale=:log10,
    markersize=3,
    markerstrokewidth=0,
    xlabel="First-return time, t",
    ylabel="Probability density",
    title="Random-weight first-return distribution (log-binned)",
    label="Log-binned empirical distribution",
    legend=:topright,
    grid=true,
)

savefig(random_return_distribution,
    joinpath(figures_dir, "random_weight_first_return_distribution_loglog.png"))

random_stay_weight_matrix = random_stay_weights(n)
random_stay_samples, random_stay_boundary_hits = frt_samples(random_stay_weight_matrix, n_runs)
random_stay_times, random_stay_densities = log_binned_distribution(random_stay_samples, 40)

println("Random-weight-with-stay first-return samples: ", length(random_stay_samples))
println("Random-weight-with-stay boundary hits: ", random_stay_boundary_hits)
println("Random-weight-with-stay fraction returning before touching a boundary: ",
    length(random_stay_samples) / n_runs)

random_stay_return_distribution = scatter(
    random_stay_times,
    random_stay_densities,
    xscale=:log10,
    yscale=:log10,
    markersize=3,
    markerstrokewidth=0,
    xlabel="First-return time, t",
    ylabel="Probability density",
    title="Random-weight-with-stay first-return distribution (log-binned)",
    label="Log-binned empirical distribution",
    legend=:topright,
    grid=true,
)

savefig(random_stay_return_distribution,
    joinpath(figures_dir, "random_weight_with_stay_first_return_distribution_loglog.png"))