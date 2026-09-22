ENV["GKSwstype"] = "100"
using Random
using Plots

figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

function fpt_sim(weights::Matrix{Float64})
    position = 1
    t = 0

    while position < size(weights, 1)
        t += 1
        r = rand()
        if r < weights[position,1] #weights[:,0] is the probability of moving right
            position += 1
        elseif position > 1 && r < weights[position,1] + weights[position,2] #weights[:,1] is the probability of moving left
            position -= 1
        else
            # stay in place
        end         
            
    end

    return t

end

function mfpt(weights::Matrix{Float64}, n::Int)
    total_time = 0
    for i in 1:n
        total_time += fpt_sim(weights)
    end
    return total_time/n
end

function fpt_samples(weights::Matrix{Float64}, n::Int)
    return [fpt_sim(weights) for _ in 1:n]
end

function fpt_moments(samples::Vector{Int})
    first_moment = sum(samples) / length(samples)
    second_moment = sum(samples .^ 2) / length(samples)
    return first_moment, second_moment
end

function theory_mfpt(weights::Matrix{Float64})
    n = size(weights, 1) + 1
    mfpt = 0.0

    for m in 1:(n - 1)
        for i in 1:m
            product = 1.0
            for j in (i + 1):m
                product *= weights[j, 2] / weights[j, 1]
            end
            mfpt += product / weights[i, 1]
        end
    end

    return mfpt
end

function vm_weights(n::Int)
    weights = zeros(n-1, 2)
    for i in 1:n-1
        weights[i, :] = [i*(n-i)/(n^2), (n-i)*(i)/(n^2)]
    end    
    return weights
end

function symmetric_mean_field_weights(weights::Matrix{Float64})
    n = size(weights, 1)
    sum_weights = sum((n .- [i for i in 1:n]) ./ weights[:, 1])         
    mf_weights = zeros(n-1, 2)

    for i in 1:n-1
        mf_weights[i, :] = [0.5*n*(n-1)/sum_weights, 0.5*n*(n-1)/sum_weights]
    end
    
    return mf_weights
end

function noisy_voter_weights(n::Int, noise::Float64)
    weights = zeros(n - 1, 2)
    for i in 1:(n - 1)
        position = -1.0 + 2.0 * i / n
        weights[i, 1] = 0.5 * noise * (1.0 - position) +
                        0.25 * (1.0 - noise) * (1.0 - position^2)
        weights[i, 2] = 0.5 * noise * (1.0 + position) +
                        0.25 * (1.0 - noise) * (1.0 - position^2)
    end
    return weights
end





function noisy_voter_shuffled_comparison(
    n::Int,
    noises::Vector{Float64},
    n_shuffles::Int,
    n_runs::Int,
)
    theory_values = Float64[]
    simulation_values = Float64[]
    noise_values = Float64[]

    for noise in noises
        weights = noisy_voter_weights(n, noise)
        for _ in 1:n_shuffles
            shuffled_weights = weights[randperm(size(weights, 1)), :]
            push!(theory_values, theory_mfpt(shuffled_weights))
            push!(simulation_values, mfpt(shuffled_weights, n_runs))
            push!(noise_values, noise)
        end
    end

    return theory_values, simulation_values, noise_values
end

function shuffled_mfpt_comparison(weights::Matrix{Float64}, n_shuffles::Int, n_runs::Int)
    theory_values = zeros(n_shuffles)
    simulation_values = zeros(n_shuffles)

    for shuffle_index in 1:n_shuffles
        shuffled_weights = weights[randperm(size(weights, 1)), :]
        theory_values[shuffle_index] = theory_mfpt(shuffled_weights)
        simulation_values[shuffle_index] = mfpt(shuffled_weights, n_runs)
    end

    return theory_values, simulation_values
end

weights = vm_weights(300)
mean_field = symmetric_mean_field_weights(weights)
n_runs = 10000

normal_samples = fpt_samples(weights, n_runs)
mean_field_samples = fpt_samples(mean_field, n_runs)
normal_first, normal_second = fpt_moments(normal_samples)
mean_field_first, mean_field_second = fpt_moments(mean_field_samples)

println("Normal mfpt: ", normal_first)
println("Normal second FPT moment: ", sqrt(normal_second))
println("Mean field mfpt: ", mean_field_first)
println("Mean field second FPT moment: ", sqrt(mean_field_second))
println("Theory mfpt: ", theory_mfpt(weights))

Random.seed!(1234)
n_shuffles = 100
shuffle_runs = 1000
theory_values, simulation_values = shuffled_mfpt_comparison(weights, n_shuffles, shuffle_runs)
println("Mean absolute shuffled MFPT error: ",
    sum(abs.(simulation_values .- theory_values)) / n_shuffles)

max_mfpt = 1.05 * maximum(vcat(theory_values, simulation_values))
p_mfpt = scatter(theory_values, simulation_values;
    label="Shuffled configurations",
    xlabel="Theory MFPT",
    ylabel="Simulation MFPT",
    title="Theory vs simulation MFPT",
    legend=:topleft)
plot!(p_mfpt, [0.0, max_mfpt], [0.0, max_mfpt];
    label="y = x",
    linestyle=:dash,
    color=:black)
savefig(joinpath(figures_dir, "mfpt_theory_vs_simulation.png"))

noises = [0.0001, 0.001, 0.01, 0.05]
noisy_shuffles = 50
noisy_runs = 500
noisy_theory, noisy_simulation, noisy_labels = noisy_voter_shuffled_comparison(
    100, noises, noisy_shuffles, noisy_runs)
println("Noisy-voter mean absolute shuffled MFPT error: ",
    sum(abs.(noisy_simulation .- noisy_theory)) / length(noisy_theory))

max_noisy_mfpt = 1.05 * maximum(vcat(noisy_theory, noisy_simulation))
p_noisy_mfpt = scatter(
    noisy_theory,
    noisy_simulation;
    group=noisy_labels,
    xlabel="Theory MFPT",
    ylabel="Simulation MFPT",
    title="Noisy-voter theory vs simulation MFPT",
    label="noise",
    legend=:topleft,
)
plot!(p_noisy_mfpt, [0.0, max_noisy_mfpt], [0.0, max_noisy_mfpt];
    label="y = x",
    linestyle=:dash,
    color=:black)
savefig(joinpath(figures_dir, "noisy_voter_mfpt_theory_vs_simulation.png"))

p_normal = histogram(normal_samples;
    bins=50,
    normalize=true,
    label="Normal",
    xlabel="First-passage time",
    ylabel="Probability density",
    title="Normal FPT")
p_mean_field = histogram(mean_field_samples;
    bins=50,
    normalize=true,
    label="Mean field",
    xlabel="First-passage time",
    ylabel="Probability density",
    title="Mean-field FPT")
plot(p_normal, p_mean_field;
    layout=(1, 2),
    size=(1200, 500))
savefig(joinpath(figures_dir, "fpt_histograms.png"))