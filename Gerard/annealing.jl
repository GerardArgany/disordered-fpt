ENV["GKSwstype"] = "100"
using Plots
using Random
using Statistics
using LinearAlgebra

# --- 1. Set Global Precision ---
setprecision(64) # Sets BigFloat precision to 256 bits globally

figures_dir = joinpath(@__DIR__, "figures", "annealing")
mkpath(figures_dir)

# --- 2. Use Parametric Types (T <: AbstractFloat) ---
function calculate_MFPT(b::Vector{T}) where {T <: AbstractFloat}
    L = length(b) # This is N - 1
    N_val = T(L + 1)
    
    total_sum = zero(T)
    # O(N^2) optimized evaluation instead of O(N^3) triple loop
    for i in 1:L
        current_prod = one(T)
        for j in i:L
            current_prod *= (one(T) - b[j]) / b[j]
            total_sum += current_prod
        end
    end
    
    return N_val + T(2.0) * total_sum
end

function calculate_MFPT_symmetric(b::Vector{T}) where {T <: AbstractFloat}
    N_minus_1 = length(b)
    mfpt = zero(T)

    for i in 1:N_minus_1
        mfpt += T(N_minus_1 + 1 - i) / b[i]
    end

    return mfpt
end

function center_high_edge_low_initialization(b_random::Vector{T}) where {T <: AbstractFloat}
    n_positions = length(b_random)
    center = (n_positions + 1) / 2
    positions_from_center = sortperm(abs.((1:n_positions) .- center))
    sorted_b = sort(b_random, rev=true)
    initialized_b = similar(b_random)

    initialized_b[positions_from_center] = sorted_b
    return initialized_b
end

function boundary_inward_extremes_ordering(b::Vector{T}) where {T <: AbstractFloat}
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
    b_initial::Vector{T};
    max_iter=100000,
    record_every=100,
    mfpt_function::Function=calculate_MFPT,
) where {T <: AbstractFloat}
    b_curr = copy(b_initial)
    T0_curr = mfpt_function(b_curr)
    
    b_best = copy(b_curr)
    T0_best = T0_curr
    
    temp = T(1.0)
    cooling_rate = T(0.9994)
    
    convergence = T[T0_best]
    
    for iter in 1:max_iter
        # Pick two distinct indices to swap
        idx1, idx2 = rand(1:length(b_curr), 2)
        while idx1 == idx2
            idx1, idx2 = rand(1:length(b_curr), 2)
        end
        
        b_new = copy(b_curr)
        b_new[idx1], b_new[idx2] = b_new[idx2], b_new[idx1]
        
        T0_new = mfpt_function(b_new)
        delta_rel = (T0_new - T0_curr) / T0_curr
        
        # Acceptance logic
        if delta_rel < zero(T) || rand(T) < exp(-delta_rel / temp)
            b_curr = copy(b_new)
            T0_curr = T0_new
            
            # Track the global best
            if T0_curr < T0_best
                b_best = copy(b_curr)
                T0_best = T0_curr
            end
        end
        
        temp *= cooling_rate
        
        if iter % record_every == 0
            push!(convergence, T0_best)
        end
    end

    if length(convergence) == 1 || (max_iter % record_every != 0)
        push!(convergence, T0_best)
    end

    return b_best, T0_best, convergence
end

function run_ensemble(
    N::Int;
    n_configurations::Int=100,
    max_iter::Int=300000,
    record_every::Int=10000,
    symmetric::Bool=false,
    b_min::Float64=0.01, # Keep API standard, convert internally
    b_max::Float64=0.99,
    case_name::String="normal",
    n_outliers::Int=0,
    outlier_min::Float64=0.01,
    outlier_max::Float64=0.99,
)
    0.0 < b_min < b_max < 1.0 ||
        throw(ArgumentError("Require 0 < b_min < b_max < 1 for finite MFPTs"))
    0 <= n_outliers <= N - 1 ||
        throw(ArgumentError("Require 0 <= n_outliers <= N - 1"))
    0.0 < outlier_min < outlier_max < 1.0 ||
        throw(ArgumentError("Require 0 < outlier_min < outlier_max < 1"))

    if symmetric
        mfpt_function = calculate_MFPT_symmetric
        output_prefix = "symmetric_"
    else
        mfpt_function = calculate_MFPT
        output_prefix = "$(case_name)_"
    end
    
    println("Running $n_configurations annealing configurations (symmetric = $symmetric)...")
    
    convergence_runs = Vector{Vector{Float64}}()
    optimized_configurations = Vector{Vector{Float64}}()
    optimized_mfpts = Float64[]

    # --- 3. Initialize Arrays directly in 256-bit BigFloat ---
    base_random_weights = BigFloat(b_min) .+ (BigFloat(b_max) - BigFloat(b_min)) .* rand(BigFloat, N - 1)
    
    if n_outliers > 0
        outlier_positions = randperm(N - 1)[1:n_outliers]
        outlier_values = BigFloat(outlier_min) .+ (BigFloat(outlier_max) - BigFloat(outlier_min)) .* rand(BigFloat, n_outliers)
        base_random_weights[outlier_positions] = outlier_values
    end

    for configuration_index in 1:n_configurations
        println("Configuration $configuration_index / $n_configurations")
        
        b_initial = if symmetric
            center_high_edge_low_initialization(base_random_weights .* BigFloat("0.45") .+ BigFloat("0.05"))
        else
            center_high_edge_low_initialization(base_random_weights)
        end

        b_opt, T0_opt, convergence = optimize_ordering(
            b_initial,
            max_iter=max_iter,
            record_every=record_every,
            mfpt_function=mfpt_function,
        )
        
        # Convert back to Float64 for plotting/saving to save memory
        push!(optimized_configurations, Float64.(b_opt))
        push!(optimized_mfpts, Float64(T0_opt))
        push!(convergence_runs, Float64.(convergence))
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
        title="MFPT convergence: $case_name (symmetric = $symmetric)",
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
        opt_config = optimized_configurations[configuration_index]
        
        best_plot = plot(
            1:(N - 1),
            opt_config,
            color=:blue,
            linewidth=2,
            label="Annealed",
            xlabel="State i",
            ylabel="b_i",
            ylims=(0.0, symmetric ? 0.5 : 1.0),
            title="Rank $rank, MFPT=$(round(optimized_mfpts[configuration_index], digits=2))",
            grid=true,
            legend=:topright,
            legendfontsize=6,
        )
        
        if b_max <= 0.5
            theoretical_opt = center_high_edge_low_initialization(opt_config)
            theo_mfpt = Float64(mfpt_function(BigFloat.(theoretical_opt)))
            
            plot!(best_plot, 
                  1:(N - 1), 
                  theoretical_opt, 
                  color=:red, 
                  linestyle=:dash, 
                  linewidth=1.5, 
                  label="Theo: $(round(theo_mfpt, digits=2))")
        end
        
        push!(best_plots, best_plot)
    end
    
    best_configurations_plot = plot(
        best_plots...,
        layout=(5, 2),
        size=(1200, 1600),
        plot_title="Ten best optimized configurations: $case_name",
    )
    savefig(best_configurations_plot,
        joinpath(figures_dir, "$(output_prefix)ten_best_configurations.pdf"))

    best_index = best_indices[1]
    b_alternating = Float64.(boundary_inward_extremes_ordering(BigFloat.(optimized_configurations[best_index])))
    T0_alternating = Float64(mfpt_function(BigFloat.(b_alternating)))
    
    println("Best optimized MFPT: ", round(optimized_mfpts[best_index], digits=4))
    println("Boundary-inward alternating MFPT: ", round(T0_alternating, digits=4))
    println("Boundary-inward alternating - optimized MFPT: ",
        round(T0_alternating - optimized_mfpts[best_index], digits=4))
    println("Saved convergence and top-10 configuration plots.")
end


# run_ensemble(100;
#     n_configurations=100,
#     b_min=0.01,
#     b_max=0.99,
#     case_name="normal")
run_ensemble(50;
    n_configurations=100,
    b_min=0.01,
    b_max=0.49,
    case_name="b_below_half")
# run_ensemble(200;
#     n_configurations=100,
#     b_min=0.51,
#     b_max=0.99,
#     case_name="b_above_half",
#     )
# run_ensemble(50;
#     n_configurations=100,
#     b_min=0.01,
#     b_max=0.49,
#     n_outliers=1,
#     outlier_min=0.51,
#     outlier_max=0.99,
#     case_name="b_below_half_plus_one_above")
# run_ensemble(50;
#     n_configurations=100,
#     b_min=0.51,
#     b_max=0.99,
#     n_outliers=1,
#     outlier_min=0.01,
#     outlier_max=0.49,
#     case_name="b_above_half_plus_one_below")

# run_ensemble(50;
#     n_configurations=100,
#     b_min=0.01,
#     b_max=0.49,
#     n_outliers=2,
#     outlier_min=0.51,
#     outlier_max=0.99,
#     case_name="b_below_half_plus_two_above")
# run_ensemble(50;
#     n_configurations=100,
#     b_min=0.51,
#     b_max=0.99,
#     n_outliers=2,
#     outlier_min=0.01,
#     outlier_max=0.49,
#     case_name="b_above_half_plus_two_below")

