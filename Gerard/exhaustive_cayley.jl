ENV["GKSwstype"] = "100" # Forces headless mode to prevent the GKS Qt driver crash
using Combinatorics
using Plots
using Random
using LinearAlgebra

figures_dir = joinpath(@__DIR__, "figures", "exhaustive_cayley")
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

function calculate_second_moment(b::Vector{Float64}, d::Vector{Float64})
    n_minus_1 = length(b)
    ratios = d ./ b

    inner_sums = zeros(Float64, n_minus_1)
    inner_sums[1] = 1.0 / b[1]
    for k in 2:n_minus_1
        inner_sums[k] = 1.0 / b[k] + ratios[k] * inner_sums[k - 1]
    end

    suffix_sums = zeros(Float64, n_minus_1 + 1)
    for index in n_minus_1:-1:1
        suffix_sums[index] = suffix_sums[index + 1] + inner_sums[index]
    end

    second_sum = 0.0
    for m in 1:n_minus_1
        for i in 1:m
            product = 1.0
            for j in (i + 1):m
                product *= ratios[j]
            end
            second_sum += (1.0 / b[i]) * suffix_sums[i] * product
        end
    end

    return 2.0 * second_sum - calculate_MFPT(b)
end

function direct_markov_moments(b::Vector{Float64})
    n_transient = length(b)
    transition_matrix = zeros(Float64, n_transient, n_transient)

    for state in 1:length(b)
        row = state
        if state == 1
            transition_matrix[row, row] += 1.0 - b[state]
        else
            transition_matrix[row, state - 1] += 1.0 - b[state]
        end
        if state < length(b)
            transition_matrix[row, state + 1] += b[state]
        end
        # A right move from the last transient state reaches absorption.
    end

    system_matrix = I - transition_matrix
    first_moments = system_matrix \ ones(n_transient)
    second_moments = system_matrix \ (ones(n_transient) + 2.0 * transition_matrix * first_moments)
    return first_moments[1], second_moments[1]
end

function prove_rugged_landscape(N::Int=7)
    # 1. Generate a fixed set of random probabilities for reproducibility
    Random.seed!(42)
    base_b = rand(N - 1)
    println("Base probabilities: ", round.(base_b, digits=3))
    
    # 2. Generate all (N-1)! permutations using Combinatorics
    all_perms = collect(permutations(1:(N-1)))
    n_states = length(all_perms)
    println("Mapped complete phase space: $n_states permutations.")

    # 3. Compute MFPT for every state
    mfpt_map = Dict{Vector{Int}, Float64}()
    variance_map = Dict{Vector{Int}, Float64}()
    for p in all_perms
        b = base_b[p]
        mfpt = calculate_MFPT(b)
        second_moment = calculate_second_moment(b, 1.0 .- b)
        mfpt_map[p] = mfpt
        variance_map[p] = second_moment - mfpt^2
    end

    # 4. Identify Local Minima
    local_minima = Vector{Vector{Int}}()
    
    for p in all_perms
        current_mfpt = mfpt_map[p]
        is_local_min = true
        
        # Check all adjacent pair swaps (N-2 possible neighbors)
        for i in 1:(N-2)
            neighbor = copy(p)
            neighbor[i], neighbor[i+1] = neighbor[i+1], neighbor[i]
            
            # If any neighbor is strictly better, we are not in a local minimum
            if mfpt_map[neighbor] < current_mfpt
                is_local_min = false
                break
            end
        end
        
        if is_local_min
            push!(local_minima, p)
        end
    end

    # 5. Extract Global Minimum
    sorted_states = sort(collect(mfpt_map), by = x -> x.second)
    global_min_perm = sorted_states[1].first
    global_min_mfpt = sorted_states[1].second
    variance_states = sort(collect(variance_map), by=x -> x.second)
    global_min_variance_perm = variance_states[1].first
    global_min_variance = variance_states[1].second

    mfpt_min_variance = variance_map[global_min_perm]
    variance_min_mfpt = mfpt_map[global_min_variance_perm]

    println("\n--- PROOF OF LOCAL MINIMA ---")
    println("Global Minimum MFPT: ", round(global_min_mfpt, digits=3))
    println("Variance at MFPT-minimizing permutation: ", round(mfpt_min_variance, digits=3))
    println("Global Minimum Variance: ", round(global_min_variance, digits=3))
    println("MFPT at variance-minimizing permutation: ", round(variance_min_mfpt, digits=3))
    println("Same permutation minimizes both: ", global_min_perm == global_min_variance_perm)
    println("MFPT-minimizing ordering: ", round.(base_b[global_min_perm], digits=3))
    println("Variance-minimizing ordering: ", round.(base_b[global_min_variance_perm], digits=3))

    for (label, permutation) in [
        ("MFPT minimum", global_min_perm),
        ("Variance minimum", global_min_variance_perm),
    ]
        b = base_b[permutation]
        closed_form_mfpt = mfpt_map[permutation]
        closed_form_second = calculate_second_moment(b, 1.0 .- b)
        direct_mfpt, direct_second = direct_markov_moments(b)
        println("\nNumerical check: $label")
        println("Closed-form MFPT: ", closed_form_mfpt)
        println("Direct Markov MFPT: ", direct_mfpt)
        println("MFPT absolute error: ", abs(closed_form_mfpt - direct_mfpt))
        println("Closed-form variance: ", closed_form_second - closed_form_mfpt^2)
        println("Direct Markov variance: ", direct_second - direct_mfpt^2)
        println("Variance absolute error: ",
            abs((closed_form_second - closed_form_mfpt^2) - (direct_second - direct_mfpt^2)))
    end
    println("Total Local Minima discovered: ", length(local_minima))
    
    # Show a specific counterexample
    trap_perm = local_minima[end] # Pick one that isn't the global min
    if trap_perm != global_min_perm
        println("\nFound Sub-optimal Trap State:")
        println("Sequence: ", round.(base_b[trap_perm], digits=3))
        println("Local Min MFPT: ", round(mfpt_map[trap_perm], digits=3))
        println("Proof: Every adjacent swap increases the MFPT, yet it is higher than the Global Min.")
    end

    # 6. Visualization: The Rugged Landscape
    ranks = 1:n_states
    mfpt_values = [s.second for s in sorted_states]
    
    p = plot(ranks, mfpt_values, yscale=:log10, seriestype=:scatter, 
             color=:lightgray, markersize=3, markerstrokewidth=0, legend=:topleft,
             xlabel="Permutation Rank (Best to Worst)", ylabel="MFPT (log scale)",
             title="Constructive Proof: Rugged Phase Space (N=$N)")

    # Highlight Local Minima
    trap_ranks = [findfirst(x -> x.first == lm, sorted_states) for lm in local_minima]
    trap_mfpts = [mfpt_map[lm] for lm in local_minima]
    
    scatter!(p, trap_ranks, trap_mfpts, color=:red, markersize=5, 
             markerstrokewidth=1, label="Local Minima (Traps)")
             
    # Highlight Global Minimum
    scatter!(p, [1], [global_min_mfpt], color=:gold, marker=:star, markersize=10, 
             markerstrokewidth=1, label="Global Minimum")

    savefig(p, joinpath(figures_dir, "landscape_proof.pdf"))

    mfpt_values = [mfpt_map[p] for p in all_perms]
    variance_values = [variance_map[p] for p in all_perms]
    moment_plot = scatter(
        mfpt_values,
        variance_values,
        xscale=:log10,
        yscale=:log10,
        color=:steelblue,
        markersize=4,
        markerstrokewidth=0,
        xlabel="MFPT, <T1>",
        ylabel="Variance, Var(T1)",
        title="MFPT versus fluctuations across all permutations (N=$N)",
        label="Permutations",
        legend=:topright,
        grid=true,
    )
    scatter!(moment_plot,
        [global_min_mfpt],
        [mfpt_min_variance],
        color=:gold,
        marker=:star,
        markersize=10,
        label="MFPT minimum",
    )
    scatter!(moment_plot,
        [variance_min_mfpt],
        [global_min_variance],
        color=:red,
        marker=:diamond,
        markersize=8,
        label="Variance minimum",
    )
    savefig(moment_plot, joinpath(figures_dir, "mfpt_vs_variance_all_permutations.pdf"))
    println("\nSaved plots to ", figures_dir)
end

prove_rugged_landscape(10)