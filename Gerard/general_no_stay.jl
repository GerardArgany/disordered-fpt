ENV["GKSwstype"] = "100"
using Roots
using Plots
using LinearAlgebra
using Statistics

figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

function b_d_mean_field(N, T)
    # We take b as the middle of b_min to b_max
    b = 0.5*(1+N/T)
   return(b, b-N/T)
end


function random_weights(n::Int)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        # Randomly assign probabilities to move left or right, ensuring they sum to 1.
        p_right = rand()
        p_left = 1.0 - p_right
        weights[i, 1] = p_right
        weights[i, 2] = p_left
    end
    return weights
end

function mean_field_weights(n::Int, T::Float64)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        weights[i, :] .= b_d_mean_field(n, T)
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

NN = 100
target_mfpt = 4 * NN
mf_weights = mean_field_weights(NN, Float64(target_mfpt))
mf_fpt = fpt_moments(fpt_samples(mf_weights, 1000))
println("Target MFPT: $target_mfpt")
println("Mean field first moment: $(mf_fpt[1]), Mean field second moment: $(mf_fpt[2]^0.5)")
println("Absolute error: $(abs(mf_fpt[1] - target_mfpt)), Relative error: $(abs(mf_fpt[1] - target_mfpt) / target_mfpt)")

# Compare target MFPTs with simulated mean-field MFPTs for several N/T ratios.
system_sizes = [25, 50, 100]
n_over_t_values = range(0.1, stop=1.0, length=15)
samples_per_point = 1000

comparison = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
for n in system_sizes
    target_values = Float64[]
    simulated_values = Float64[]

    for n_over_t in n_over_t_values
        target = n / n_over_t
        weights = mean_field_weights(n, target)
        simulated_mfpt = mfpt(weights, samples_per_point)

        push!(target_values, target)
        push!(simulated_values, simulated_mfpt)
    end

    comparison[n] = (target_values, simulated_values)
end

plot_title = "Target MFPT versus simulated mean-field MFPT"
comparison_plot = plot(
    xlabel="Target MFPT (T)",
    ylabel="Simulated mean-field MFPT",
    title=plot_title,
    legend=:topleft,
    grid=true,
)

all_targets = Float64[]
all_simulated = Float64[]
for n in system_sizes
    target_values, simulated_values = comparison[n]
    append!(all_targets, target_values)
    append!(all_simulated, simulated_values)
    scatter!(comparison_plot, target_values, simulated_values,
        label="N = $n",
        markersize=4)
end

identity_limits = (10.0, 1.05 * max(maximum(all_targets), maximum(all_simulated)))
plot!(comparison_plot,
    [identity_limits[1], identity_limits[2]],
    [identity_limits[1], identity_limits[2]],
    line=(:dash, :black),
    label="x = y",
    xscale= :log10,
    yscale= :log10,
    )

xlims!(comparison_plot, identity_limits)
ylims!(comparison_plot, identity_limits)
savefig(comparison_plot, joinpath(figures_dir, "target_vs_mean_field_mfpt.png"))
