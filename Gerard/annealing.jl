ENV["GKSwstype"] = "100"
using Plots
using Random
using Statistics
using LinearAlgebra

figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)

function calculate_MFPT(b::Vector{Float64})
    N_minus_1 = length(b)
    tau = zeros(Float64, N_minus_1)
    
    tau[1] = 1.0 / b[1]
    for m in 2:N_minus_1
        x_m = (1.0 - b[m]) / b[m]
        tau[m] = (1.0 / b[m]) + x_m * tau[m-1]
    end
    
    return sum(tau)
end

function calculate_MFPT_symmetric(b::Vector{Float64})
    N_minus_1 = length(b)
    mfpt = 0.0

    for i in 1:N_minus_1
        mfpt += (N_minus_1 + 1 - i) / b[i]
    end

    return mfpt
end

function center_high_edge_low_initialization(b_random::Vector{Float64})
    n_positions = length(b_random)
    center = (n_positions + 1) / 2
    positions_from_center = sortperm(abs.((1:n_positions) .- center))
    sorted_b = sort(b_random, rev=true)
    initialized_b = similar(b_random)

    initialized_b[positions_from_center] = sorted_b
    return initialized_b
end

function boundary_inward_extremes_ordering(b::Vector{Float64})
    sorted_b = sort(b)
    ordered_b = similar(b)
    left_position = 1
    right_position = length(sorted_b)
    low_value = 1
    high_value = length(sorted_b)

    while left_position <= right_position
        ordered_b[left_position] = sorted_b[low_value]
        left_position += 1
        low_value += 1

        if left_position <= right_position
            ordered_b[right_position] = sorted_b[low_value]
            right_position -= 1
            low_value += 1
        end

        if left_position <= right_position
            ordered_b[left_position] = sorted_b[high_value]
            left_position += 1
            high_value -= 1
        end

        if left_position <= right_position
            ordered_b[right_position] = sorted_b[high_value]
            right_position -= 1
            high_value -= 1
        end
    end

    return ordered_b
end

function optimize_ordering(
    b_initial::Vector{Float64};
    max_iter=100000,
    record_every=100,
    mfpt_function::Function=calculate_MFPT,
)
    b_curr = copy(b_initial)
    T1_curr = mfpt_function(b_curr)
    
    b_best = copy(b_curr)
    T1_best = T1_curr
    
    T_init = 1.0
    T = T_init 
    cooling_rate = 0.9994
    
    stagnation_counter = 0
    stagnation_limit = 2000 # Corrected from 20 to allow the algorithm time to search
    reheat_multiplier = 0.3 # Reheat to 30% of initial temperature
    
    convergence = Float64[T1_best]
    
    for iter in 1:max_iter
        # Pick two distinct indices to swap
        idx1, idx2 = rand(1:length(b_curr), 2)
        while idx1 == idx2
            idx1, idx2 = rand(1:length(b_curr), 2)
        end
        
        b_new = copy(b_curr)
        b_new[idx1], b_new[idx2] = b_new[idx2], b_new[idx1]
        
        T1_new = mfpt_function(b_new)
        delta_rel = (T1_new - T1_curr) / T1_curr
        
        # Acceptance logic
        if delta_rel < 0 || rand() < exp(-delta_rel / T)
            b_curr = copy(b_new)
            T1_curr = T1_new
            
            # Track the global best
            if T1_curr < T1_best
                b_best = copy(b_curr)
                T1_best = T1_curr
                stagnation_counter = 0 # Reset counter when we find a new global minimum
            else
                stagnation_counter += 1
            end
        else
            stagnation_counter += 1
        end
        
        # Adaptive Cooling / Reheating Logic
        if stagnation_counter >= stagnation_limit
            T = T_init * reheat_multiplier # Inject heat to escape the local minimum
            stagnation_counter = 0
        else
            T *= cooling_rate
        end
        
        if iter % record_every == 0
            push!(convergence, T1_best)
        end
    end

    if length(convergence) == 1 || (max_iter % record_every != 0)
        push!(convergence, T1_best)
    end

    return b_best, T1_best, convergence
end

function run_ensemble(
    N::Int;
    n_configurations::Int=200,
    max_iter::Int=50000,
    record_every::Int=100,
    symmetric::Bool=false,
)
    if symmetric
        # For d = b, the stay probability is 1 - 2b, so b must be at most 0.5.
        mfpt_function = calculate_MFPT_symmetric
        output_prefix = "symmetric_"
    else
        mfpt_function = calculate_MFPT
        output_prefix = ""
    end
    
    println("Running $n_configurations annealing configurations (symmetric = $symmetric)...")
    convergence_runs = Vector{Vector{Float64}}()
    optimized_configurations = Vector{Vector{Float64}}()
    optimized_mfpts = Float64[]

    # Generate the base weights exactly ONCE outside the loop
    base_random_weights = rand(N - 1)

    for configuration_index in 1:n_configurations
        println("Configuration $configuration_index / $n_configurations")
        
        # Apply the initialization to the exact same weights every time
        b_initial = if symmetric
            center_high_edge_low_initialization(base_random_weights .* 0.45 .+ 0.05)
        else
            center_high_edge_low_initialization(base_random_weights)
        end

        b_opt, T1_opt, convergence = optimize_ordering(
            b_initial,
            max_iter=max_iter,
            record_every=record_every,
            mfpt_function=mfpt_function,
        )
        push!(optimized_configurations, b_opt)
        push!(optimized_mfpts, T1_opt)
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
        title="MFPT convergence across annealing configurations (symmetric = $symmetric)",
        legend=:topright,
        grid=true,
    )
    for configuration_index in 1:n_configurations
        plot!(convergence_plot, iterations, plotted_convergence[configuration_index, :],
            color=:steelblue, linealpha=0.18, linewidth=1, label=false)
    end
    plot!(convergence_plot, iterations, mean_convergence,
        color=:black, linewidth=3, label="Ensemble mean")
    savefig(convergence_plot,
        joinpath(figures_dir, "$(output_prefix)annealing_mfpt_convergence.pdf"))

    best_indices = sortperm(optimized_mfpts)[1:min(10, n_configurations)]
    best_plots = Any[]
    for (rank, configuration_index) in enumerate(best_indices)
        best_plot = plot(
            1:(N - 1),
            optimized_configurations[configuration_index],
            color=:blue,
            linewidth=1.5,
            legend=false,
            xlabel="State i",
            ylabel="b_i",
            ylims=(0.0, symmetric ? 0.5 : 1.0),
            title="Rank $rank, MFPT=$(round(optimized_mfpts[configuration_index], digits=2))",
            grid=true,
        )
        push!(best_plots, best_plot)
    end
    best_configurations_plot = plot(
        best_plots...,
        layout=(5, 2),
        size=(1200, 1600),
        plot_title="Ten best optimized configurations (symmetric = $symmetric)",
    )
    savefig(best_configurations_plot,
        joinpath(figures_dir, "$(output_prefix)ten_best_configurations.pdf"))

    best_index = best_indices[1]
    b_alternating = boundary_inward_extremes_ordering(optimized_configurations[best_index])
    T1_alternating = mfpt_function(b_alternating)
    println("Best optimized MFPT: ", round(optimized_mfpts[best_index], digits=4))
    println("Boundary-inward alternating MFPT: ", round(T1_alternating, digits=4))
    println("Boundary-inward alternating - optimized MFPT: ",
        round(T1_alternating - optimized_mfpts[best_index], digits=4))
    println("Saved convergence and top-10 configuration plots.")
end

function sample_trajectory(b::Vector{Float64}; rng::AbstractRNG=Random.default_rng())
    position = 1
    positions = Int[position]
    times = Int[0]
    terminal_position = length(b)

    while position < terminal_position
        right_probability = b[position]
        random_value = rand(rng)
        if random_value < right_probability
            position += 1
        elseif position > 1 && random_value < right_probability + (1.0 - right_probability)
            position -= 1
        end

        push!(times, times[end] + 1)
        push!(positions, position)
    end

    return times, positions
end

function plot_trajectory(times, positions, ordering_name, mfpt_value, fpt, chain_size)
    n_states = chain_size - 1
    visit_counts = zeros(Int, n_states)
    for position in positions
        visit_counts[position] += 1
    end

    trajectory_plot = plot(
        times,
        positions,
        color=:steelblue,
        linewidth=1.5,
        alpha=0.85,
        marker=:circle,
        markersize=2.5,
        markerstrokewidth=0,
        xlabel="Time step",
        ylabel="Particle position",
        title="$ordering_name ordering (FPT = $fpt, MFPT = $(round(mfpt_value, digits=2)))",
        legend=:topright,
        grid=true,
        ylims=(0, chain_size),
    )
    scatter!(trajectory_plot, [times[1]], [positions[1]],
        color=:green, marker=:circle, markersize=7, label="Start")
    scatter!(trajectory_plot, [times[end]], [positions[end]],
        color=:red, marker=:star5, markersize=9, label="FPT / end")
    annotate!(trajectory_plot, times[1], positions[1], text(" start", :green, 8))
    annotate!(trajectory_plot, times[end], positions[end], text(" end", :red, 8))

    visit_histogram = bar(
        1:n_states,
        visit_counts,
        orientation=:h,
        color=:darkorange,
        alpha=0.8,
        legend=false,
        xlabel="Visits",
        ylabel="State",
        title="State visits",
        grid=true,
        ylims=(0, chain_size),
    )

    return plot(
        trajectory_plot,
        visit_histogram,
        layout=(1, 2),
        widths=(0.72, 0.28),
        link=:y,
        left_margin=5Plots.mm,
        right_margin=5Plots.mm,
    )
end

function expected_visit_counts(b::Vector{Float64})
    n_states = length(b)
    transient_states = n_states - 1
    transition_matrix = zeros(Float64, transient_states, transient_states)

    for state in 1:transient_states
        right_probability = b[state]
        if state < transient_states
            transition_matrix[state, state + 1] += right_probability
        end
        if state == 1
            transition_matrix[state, state] += 1.0 - right_probability
        else
            transition_matrix[state, state - 1] += 1.0 - right_probability
        end
    end

    initial_visits = zeros(Float64, transient_states)
    initial_visits[1] = 1.0
    expected_transient_visits = (I - transition_matrix)' \ initial_visits
    return vcat(expected_transient_visits, 1.0)
end

N = 15
base_weights = rand(N - 1)
b_initial = center_high_edge_low_initialization(base_weights)
b_optimal, optimal_mfpt, _ = optimize_ordering(
    b_initial,
    max_iter=100000,
    record_every=100,
    mfpt_function=calculate_MFPT,
)

b_ascending = sort(b_optimal)
b_descending = sort(b_optimal, rev=true)

println("N = $N")
println("Optimal ordering: ", round.(b_optimal, digits=4))
println("Optimal MFPT: ", round(optimal_mfpt, digits=4))
println("Ascending MFPT: ", round(calculate_MFPT(b_ascending), digits=4))
println("Descending MFPT: ", round(calculate_MFPT(b_descending), digits=4))

trajectory_data = Tuple{String, Vector{Int}, Vector{Int}, Float64}[]
for (ordering_name, ordering, mfpt_value) in [
    ("Optimal", b_optimal, calculate_MFPT(b_optimal)),
    ("Ascending", b_ascending, calculate_MFPT(b_ascending)),
    ("Descending", b_descending, calculate_MFPT(b_descending)),
]
    times, positions = sample_trajectory(ordering)
    push!(trajectory_data, (ordering_name, times, positions, mfpt_value))
    println("$(ordering_name) trajectory FPT: ", times[end])
end

trajectory_plots = [
    plot_trajectory(times, positions, ordering_name, mfpt_value, times[end], N)
    for (ordering_name, times, positions, mfpt_value) in trajectory_data
]
trajectory_comparison = plot(
    trajectory_plots...,
    layout=(3, 1),
    size=(1100, 1200),
    plot_title="N = $N particle trajectories",
)
savefig(trajectory_comparison, joinpath(figures_dir, "N15_trajectory_comparison.pdf"))
println("Saved trajectory comparison plot.")

ensemble_size = 200
ensemble_orderings = [b_optimal, b_ascending, b_descending]
ensemble_names = ["Optimal", "Ascending", "Descending"]
ensemble_visit_matrix = Matrix{Float64}(undef, N - 1, length(ensemble_orderings))

for (ordering_index, ordering) in enumerate(ensemble_orderings)
    expected_visits = expected_visit_counts(ordering)
    ensemble_visit_matrix[:, ordering_index] .= expected_visits
    println("$(ensemble_names[ordering_index]) expected ensemble FPT: ",
        round(sum(expected_visits) - 1.0, digits=2))
end

ensemble_histogram = bar(
    1:(N - 1),
    ensemble_visit_matrix,
    bar_position=:dodge,
    label=ensemble_names,
    xlabel="Particle state",
    ylabel="Expected visits per trajectory",
    title="State visits over an ensemble of $ensemble_size trajectories (N = $N)",
    legend=:topright,
    grid=true,
    ylims=(0, nothing),
    size=(1200, 700),
)
savefig(ensemble_histogram,
    joinpath(figures_dir, "N15_ensemble_state_visit_histograms.pdf"))
savefig(ensemble_histogram,
    joinpath(figures_dir, "N15_ensemble_state_visit_histograms.png"))
println("Saved ensemble state-visit histogram plot.")