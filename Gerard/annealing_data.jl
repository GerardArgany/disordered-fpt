using Random
using DelimitedFiles

# Ensure the data directory exists
data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

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

function center_high_edge_low_initialization(b_random::Vector{Float64})
    n_positions = length(b_random)
    center = (n_positions + 1) / 2
    positions_from_center = sortperm(abs.((1:n_positions) .- center))
    sorted_b = sort(b_random, rev=true)
    initialized_b = similar(b_random)

    initialized_b[positions_from_center] = sorted_b
    return initialized_b
end

function optimize_ordering(
    b_initial::Vector{Float64};
    max_iter=300000,
    patience=50000, # Stops if no new global best is found for this many iterations
    mfpt_function::Function=calculate_MFPT,
)
    b_curr = copy(b_initial)
    T1_curr = mfpt_function(b_curr)
    
    b_best = copy(b_curr)
    T1_best = T1_curr
    
    T = 1.0
    cooling_rate = 0.9994
    
    iters_since_best = 0
    
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
                iters_since_best = 0 # Reset counter when a new best is found
            else
                iters_since_best += 1
            end
        else
            iters_since_best += 1
        end
        
        T *= cooling_rate
        
        # Early convergence stop
        if iters_since_best >= patience
            println("    -> Converged early at iteration $iter.")
            break
        end
    end

    return b_best, T1_best
end

"""
Generates, optimizes, and saves M sequences.
- `N`: System size
- `M`: Number of realizations to generate
- `range_type`: Symbol indicating the range. Options are `:above` ([0.51, 0.99]), `:below` ([0.01, 0.49]), or `:all` ([0.01, 0.99]).
"""
function generate_and_save_sequence(N::Int, M::Int, range_type::Symbol=:all; max_iter::Int=5000000, patience::Int=50000)
    if range_type == :above
        b_min, b_max = 0.51, 0.99
        suffix = "above_half"
    elseif range_type == :below
        b_min, b_max = 0.01, 0.49
        suffix = "below_half"
    elseif range_type == :all
        b_min, b_max = 0.01, 0.99
        suffix = "whole_range"
    else
        throw(ArgumentError("range_type must be :above, :below, or :all"))
    end

    println("Generating $M optimal sequences for N=$N, bounds=[$b_min,$b_max]...")

    # Create a matrix to store all realizations (M rows x N-1 columns)
    all_realizations = zeros(M, N - 1)

    for i in 1:M
        println("Optimizing realization $i / $M")
        # Generate and initialize weights
        base_random_weights = b_min .+ (b_max - b_min) .* rand(N - 1)
        b_initial = center_high_edge_low_initialization(base_random_weights)

        # Optimize
        b_opt, T1_opt = optimize_ordering(b_initial; max_iter=max_iter, patience=patience)
        
        # Store in matrix
        all_realizations[i, :] = b_opt
    end

    # Save to file
    filename = joinpath(data_dir, "optimal_sequences_N$(N)_M$(M)_$(suffix).csv")
    writedlm(filename, all_realizations, ',')
    
    println("Data saved to: $filename")
    
    return all_realizations
end

# Example usage:
for N in [16, 32, 64, 128, 256, 512, 1024, 2048]
    generate_and_save_sequence(N, 250, :above) # Generates 250 realizations for each N
end
