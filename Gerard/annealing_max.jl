ENV["GKSwstype"] = "100"
using Plots
using Random
using Statistics

figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

function calculate_MFPT(b::Vector{Float64})
    n_minus_1 = length(b)
    tau = zeros(Float64, n_minus_1)

    tau[1] = 1.0 / b[1]
    for m in 2:n_minus_1
        ratio = (1.0 - b[m]) / b[m]
        tau[m] = (1.0 / b[m]) + ratio * tau[m - 1]
    end

    return sum(tau)
end

function pendulum_ordering(b::Vector{Float64})
    # The paper's p_i is the backward rate d_i = 1 - b_i in this script.
    sorted_backward_rates = sort(1.0 .- b)
    pendulum_backward_rates = similar(sorted_backward_rates)
    n_rates = length(sorted_backward_rates)

    for sorted_index in 1:n_rates
        position = if sorted_index <= (n_rates + 1) / 2
            2 * sorted_index - 1
        else
            2 * (n_rates + 1 - sorted_index)
        end
        pendulum_backward_rates[position] = sorted_backward_rates[sorted_index]
    end

    return 1.0 .- pendulum_backward_rates
end

function optimize_ordering_max(
    b_initial::Vector{Float64};
    max_iter::Int=100000,
    record_every::Int=100,
)
    b_current = copy(b_initial)
    current_mfpt = calculate_MFPT(b_current)
    b_best = copy(b_current)
    best_mfpt = current_mfpt

    initial_temperature = 1.0
    temperature = initial_temperature
    cooling_rate = 0.9999
    stagnation_counter = 0
    stagnation_limit = 2000
    reheat_multiplier = 0.3
    convergence = Float64[best_mfpt]

    for iteration in 1:max_iter
        index_one, index_two = rand(1:length(b_current), 2)
        while index_one == index_two
            index_one, index_two = rand(1:length(b_current), 2)
        end

        b_proposed = copy(b_current)
        b_proposed[index_one], b_proposed[index_two] =
            b_proposed[index_two], b_proposed[index_one]
        proposed_mfpt = calculate_MFPT(b_proposed)
        relative_change = (proposed_mfpt - current_mfpt) / current_mfpt

        if relative_change > 0 || rand() < exp(relative_change / temperature)
            b_current = b_proposed
            current_mfpt = proposed_mfpt

            if current_mfpt > best_mfpt
                b_best = copy(b_current)
                best_mfpt = current_mfpt
                stagnation_counter = 0
            else
                stagnation_counter += 1
            end
        else
            stagnation_counter += 1
        end

        if stagnation_counter >= stagnation_limit
            temperature = initial_temperature * reheat_multiplier
            stagnation_counter = 0
        else
            temperature *= cooling_rate
        end

        if iteration % record_every == 0
            push!(convergence, best_mfpt)
        end
    end

    if max_iter % record_every != 0
        push!(convergence, best_mfpt)
    end

    return b_best, best_mfpt, convergence
end

function run_maximization_ensemble(
    n_states::Int;
    n_configurations::Int=100,
    max_iter::Int=100000,
    record_every::Int=100,
)
    output_prefix = "max_"
    base_weights = rand(n_states - 1)
    convergence_runs = Vector{Vector{Float64}}()
    optimized_configurations = Vector{Vector{Float64}}()
    optimized_mfpts = Float64[]

    println("Running $n_configurations MFPT-maximizing configurations")
    println("Using no-stay rates: d_i = 1 - b_i")

    for configuration_index in 1:n_configurations
        println("Configuration $configuration_index / $n_configurations")
        initial_configuration = base_weights[randperm(length(base_weights))]
        optimized_configuration, optimized_mfpt, convergence = optimize_ordering_max(
            initial_configuration;
            max_iter=max_iter,
            record_every=record_every,
        )
        push!(optimized_configurations, optimized_configuration)
        push!(optimized_mfpts, optimized_mfpt)
        push!(convergence_runs, convergence)
    end

    n_recorded_points = length(convergence_runs[1])
    convergence_matrix = Matrix{Float64}(undef, n_configurations, n_recorded_points)
    for configuration_index in 1:n_configurations
        convergence_matrix[configuration_index, :] .= convergence_runs[configuration_index]
    end

    plotted_convergence = convergence_matrix[:, 2:end]
    mean_convergence = vec(mean(plotted_convergence, dims=1))
    iterations = collect(record_every:record_every:(size(plotted_convergence, 2) * record_every))

    convergence_plot = plot(
        xlabel="Annealing iteration",
        ylabel="Best-so-far MFPT",
        yscale=:log10,
        title="No-stay MFPT maximization convergence",
        legend=:topleft,
        grid=true,
    )
    for configuration_index in 1:n_configurations
        plot!(convergence_plot, iterations, plotted_convergence[configuration_index, :],
            color=:darkorange, linealpha=0.14, linewidth=1, label=false)
    end
    plot!(convergence_plot, iterations, mean_convergence,
        color=:black, linewidth=3, label="Ensemble mean")
    savefig(convergence_plot,
        joinpath(figures_dir, "$(output_prefix)annealing_mfpt_convergence.pdf"))

    best_indices = sortperm(optimized_mfpts, rev=true)[1:min(10, n_configurations)]
    best_plots = Any[]
    for (rank, configuration_index) in enumerate(best_indices)
        best_plot = plot(
            1:(n_states - 1),
            optimized_configurations[configuration_index],
            color=:darkorange,
            linewidth=1.5,
            legend=false,
            xlabel="State i",
            ylabel="b_i",
            ylims=(0.0, 1.0),
            title="Rank $rank, MFPT=$(round(optimized_mfpts[configuration_index], digits=2))",
            grid=true,
        )
        push!(best_plots, best_plot)
    end

    best_configurations_plot = plot(
        best_plots...,
        layout=(5, 2),
        size=(1200, 1600),
        plot_title="Ten maximal no-stay MFPT configurations",
    )
    savefig(best_configurations_plot,
        joinpath(figures_dir, "$(output_prefix)ten_maximal_configurations.pdf"))

    pendulum_configuration = pendulum_ordering(base_weights)
    pendulum_mfpt = calculate_MFPT(pendulum_configuration)
    comparison_labels = ["Pendulum"; ["Annealed $rank" for rank in 1:length(best_indices)]]
    comparison_values = [pendulum_mfpt; optimized_mfpts[best_indices]]
    pendulum_comparison_plot = bar(
        comparison_labels,
        comparison_values,
        yscale=:log10,
        color=[:forestgreen; fill(:darkorange, length(best_indices))],
        xlabel="Configuration",
        ylabel="MFPT",
        title="Pendulum versus ten maximal annealed configurations",
        legend=false,
        xrotation=45,
        grid=true,
    )
    savefig(pendulum_comparison_plot,
        joinpath(figures_dir, "$(output_prefix)pendulum_vs_ten_maximal.pdf"))

    println("Maximum MFPT: ", maximum(optimized_mfpts))
    println("Pendulum MFPT: ", pendulum_mfpt)
    println("Annealed configurations below pendulum: ",
        sum(optimized_mfpts[best_indices] .< pendulum_mfpt), "/", length(best_indices))
    println("Saved maximization convergence and top-10 configuration plots.")
end

run_maximization_ensemble(50)
