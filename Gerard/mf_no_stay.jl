println("Launching mf_no_stay.jl...")
flush(stdout)
ENV["GKSwstype"] = "100"
using Roots
using Plots
using LinearAlgebra
using Statistics

figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

function T_mean_field(b, N)
    if isapprox(b, 0.5, atol=1e-8)
        return N * (N - 1)
    end
    
    x = (1 - b) / b
    term1 = N - 1
    term2 = ((1 - b) / (2b - 1)) * (1 - x^(N - 1))
    
    return (1 / (2b - 1)) * (term1 - term2)
end

function find_effective_b(T_exact, N)
    # Explicitly handle the "no solution" domain
    if T_exact < N - 1
        return NaN 
    elseif isapprox(T_exact, N - 1, atol=1e-5)
        return 1.0
    end
    
    objective(b) = T_mean_field(b, N) - T_exact
    
    try
        return find_zero(objective, 0.5)
    catch
        return find_zero(objective, (1e-12, 0.999999)) 
    end
end

function no_stay_weights(n::Int)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        # Smooth drift from the left boundary toward the right boundary.
        # This keeps the walk moving without creating a near-absorbing left wall.
        p_right = 0.1 + 0.5 * (i / (n - 1))
        p_left = 1.0 - p_right
        weights[i, 1] = p_right
        weights[i, 2] = p_left
    end
    return weights
end

function compare_weight_orderings(weights::Matrix{Float64}, n_samples::Int)
    ascending_weights = weights[sortperm(weights[:, 1]), :]
    descending_weights = reverse(ascending_weights, dims=1)

    ascending_samples = fpt_samples(ascending_weights, n_samples)
    descending_samples = fpt_samples(descending_weights, n_samples)

    return ascending_weights, descending_weights, ascending_samples, descending_samples
end

function ornstein_uhlenbeck_weights(n::Int)
    weights = zeros(n - 1, 2)

    for i in 1:(n - 1)
        # Linear restoring force toward the center (n/2).
        # At i = 1 (left), it strongly pushes right.
        # At i = n-1 (right), it strongly pushes left.
        p_right = 1.0 - (i / n)
        p_left = 1.0 - p_right

        weights[i, 1] = p_right
        weights[i, 2] = p_left
    end

    return weights
end

function fpt_sim(weights::Matrix{Float64})
    position = 1
    t = 0

    while position < size(weights, 1)
        t += 1
        r = rand()
        if r < weights[position, 1]
            position += 1
        elseif position > 1 && r < weights[position, 1] + weights[position, 2]
            position -= 1
        else
            # Numerical roundoff only; no explicit stay state is intended.
        end
    end

    return t
end

function mfpt(weights::Matrix{Float64}, n::Int)
    total_time = 0
    for i in 1:n
        total_time += fpt_sim(weights)
    end
    return total_time / n
end

function fpt_samples(weights::Matrix{Float64}, n::Int)
    return [fpt_sim(weights) for _ in 1:n]
end

function fpt_moments(samples::Vector{Int})
    first_moment = sum(samples) / length(samples)
    second_moment = sum(samples .^ 2) / length(samples)
    return first_moment, second_moment
end

run_ou_comparison = false
println("Starting no-stay MFPT comparison...")
flush(stdout)

function curve_data(N::Int; n_points::Int = 200)
    T_min = N - 1
    T_max = N * (N - 1)
    T_values = range(T_min, stop=T_max, length=n_points)
    b_solutions = Vector{Float64}(undef, length(T_values))

    for (idx, T) in enumerate(T_values)
        if isapprox(T, T_min, atol=1e-8)
            b_solutions[idx] = 1.0
            continue
        end

        objective(b) = T_mean_field(b, N) - T
        # Valid solution interval is b in (0.5, 1), and the target T is in [N-1, N(N-1)]
        b_solutions[idx] = find_zero(objective, (0.5 + 1e-10, 1.0 - 1e-10))
    end

    return T_values, b_solutions
end

no_stay = no_stay_weights(100)
ou_weights = ornstein_uhlenbeck_weights(100)

no_stay_samples = fpt_samples(no_stay, 5000)

ascending_no_stay, descending_no_stay, ascending_samples, descending_samples =
    compare_weight_orderings(no_stay, 5000)
ascending_mfpt, ascending_second = fpt_moments(ascending_samples)
descending_mfpt, descending_second = fpt_moments(descending_samples)

empirical_mfpt, empirical_second = fpt_moments(no_stay_samples)

b_eff = find_effective_b(empirical_mfpt, 50)

uniform_no_stay_weights = zeros(size(no_stay))
for i in 1:size(uniform_no_stay_weights, 1)
    uniform_no_stay_weights[i, 1] = b_eff
    uniform_no_stay_weights[i, 2] = 1.0 - b_eff
end
uniform_no_stay_samples = fpt_samples(uniform_no_stay_weights, 5000)
uniform_mfpt, uniform_second = fpt_moments(uniform_no_stay_samples)

println("Empirical MFPT (no-stay): ", empirical_mfpt)
println("Empirical second moment (no-stay): ", empirical_second)
println("Effective b from MFPT: ", b_eff)
println("Uniform-weight MFPT (b, 1-b): ", uniform_mfpt)
println("Uniform-weight second moment (b, 1-b): ", uniform_second)
println("MFPT difference: ", uniform_mfpt - empirical_mfpt)
println("Second-moment difference: ", uniform_second - empirical_second)

println("Ascending-order MFPT (no-stay): ", ascending_mfpt)
println("Descending-order MFPT (no-stay): ", descending_mfpt)
println("Descending - ascending MFPT: ", descending_mfpt - ascending_mfpt)
println("Ascending-order second moment (no-stay): ", ascending_second)
println("Descending-order second moment (no-stay): ", descending_second)

if run_ou_comparison
    ou_samples = fpt_samples(ou_weights, 5000)
    ou_mfpt, ou_second = fpt_moments(ou_samples)
    b_eff_ou = find_effective_b(ou_mfpt, 50)

    uniform_ou_weights = zeros(size(ou_weights))
    for i in 1:size(uniform_ou_weights, 1)
        uniform_ou_weights[i, 1] = b_eff_ou
        uniform_ou_weights[i, 2] = 1.0 - b_eff_ou
    end
    uniform_ou_samples = fpt_samples(uniform_ou_weights, 5000)
    uniform_mfpt_ou, uniform_second_ou = fpt_moments(uniform_ou_samples)

    println("Empirical MFPT (Ornstein-Uhlenbeck): ", ou_mfpt)
    println("Empirical second moment (Ornstein-Uhlenbeck): ", ou_second)
    println("Effective b from MFPT (Ornstein-Uhlenbeck): ", b_eff_ou)
    println("Uniform-weight MFPT (b, 1-b) (Ornstein-Uhlenbeck): ", uniform_mfpt_ou)
    println("Uniform-weight second moment (b, 1-b) (Ornstein-Uhlenbeck): ", uniform_second_ou)
end

# No-stay histogram comparison
no_stay_bins = collect(range(minimum(vcat(no_stay_samples, uniform_no_stay_samples)), maximum(vcat(no_stay_samples, uniform_no_stay_samples)), length=41))
mean_no_stay = mean(no_stay_samples)
mean_uniform_no_stay = mean(uniform_no_stay_samples)

hist_no_stay = histogram(no_stay_samples,
    bins=no_stay_bins,
    alpha=0.7,
    normalize=true,
    label="Heterogeneous no-stay FPT",
    xlabel="FPT",
    ylabel="Density",
    title="FPT histogram: heterogeneous vs mean-field (no-stay)",
    legend=:topright,
    grid=true)

histogram!(hist_no_stay, uniform_no_stay_samples,
    bins=no_stay_bins,
    alpha=0.5,
    normalize=true,
    label="Mean-field uniform-weight FPT")

vline!(hist_no_stay, [mean_no_stay], line=(:dash, :blue), label="Mean no-stay")
vline!(hist_no_stay, [mean_uniform_no_stay], line=(:dash, :orange), label="Mean uniform")

savefig(hist_no_stay, joinpath(figures_dir, "fpt_histogram_comparison_no_stay.png"))

# Ornstein-Uhlenbeck histogram comparison
if run_ou_comparison
    ou_bins = collect(range(minimum(vcat(ou_samples, uniform_ou_samples)), maximum(vcat(ou_samples, uniform_ou_samples)), length=41))
    mean_ou = mean(ou_samples)
    mean_uniform_ou = mean(uniform_ou_samples)

hist_ou = histogram(ou_samples,
    bins=ou_bins,
    alpha=0.7,
    normalize=true,
    label="Heterogeneous OU FPT",
    xlabel="FPT",
    ylabel="Density",
    title="FPT histogram: heterogeneous vs mean-field (Ornstein-Uhlenbeck)",
    legend=:topright,
    grid=true)

histogram!(hist_ou, uniform_ou_samples,
    bins=ou_bins,
    alpha=0.5,
    normalize=true,
    label="Mean-field uniform-weight FPT")

vline!(hist_ou, [mean_ou], line=(:dash, :blue), label="Mean OU")
vline!(hist_ou, [mean_uniform_ou], line=(:dash, :orange), label="Mean uniform")

    savefig(hist_ou, joinpath(figures_dir, "fpt_histogram_comparison_ornstein_uhlenbeck.png"))
end

ordering_bins = collect(range(
    minimum(vcat(ascending_samples, descending_samples)),
    maximum(vcat(ascending_samples, descending_samples)),
    length=41))
ordering_plot = histogram(ascending_samples,
    bins=ordering_bins,
    alpha=0.7,
    normalize=true,
    label="Ascending weights",
    xlabel="FPT",
    ylabel="Density",
    title="FPT histogram: ascending vs descending no-stay weights",
    legend=:topright,
    grid=true)
histogram!(ordering_plot, descending_samples,
    bins=ordering_bins,
    alpha=0.5,
    normalize=true,
    label="Descending weights")
vline!(ordering_plot, [ascending_mfpt], line=(:dash, :blue), label="Ascending mean")
vline!(ordering_plot, [descending_mfpt], line=(:dash, :orange), label="Descending mean")
savefig(ordering_plot, joinpath(figures_dir, "fpt_histogram_ordering_no_stay.png"))

N_initial = 100
T_valid, b_valid = curve_data(N_initial)

figure = plot(T_valid, b_valid,
    xaxis=:log10,
    yaxis=:log10,
    title="Effective b versus heterogeneous time (N = $N_initial)",
    xlabel="Exact Heterogeneous Time (T)",
    ylabel="Effective Rate (b)",
    linewidth=2,
    color=:blue,
    label="Solution b(T)",
    legend=:topright,
    grid=true)

vline!(figure, [Float64(N_initial - 1)], line=(:dash, :red), label="Existence Boundary (T = N-1)")

savefig(figure, joinpath(figures_dir, "solution_plot.png"))