include(joinpath(@__DIR__, "..", "paths.jl"))
using Random
using Statistics
using Plots
using LaTeXStrings

# ==========================================
# 1. Physics & Optimization Functions
# ==========================================

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
    max_iter=5000000, # Increased for a much deeper search
    patience=500000,  # Increased to prevent premature convergence
    T_fixed=0.0,
    mfpt_function::Function=calculate_MFPT,
)
    b_curr = copy(b_initial)
    T1_curr = mfpt_function(b_curr)
    
    b_best = copy(b_curr)
    T1_best = T1_curr
    
    iters_since_best = 0
    
    for iter in 1:max_iter
        idx1, idx2 = rand(1:length(b_curr), 2)
        while idx1 == idx2
            idx1, idx2 = rand(1:length(b_curr), 2)
        end
        
        b_new = copy(b_curr)
        b_new[idx1], b_new[idx2] = b_new[idx2], b_new[idx1]
        
        T1_new = mfpt_function(b_new)
        delta_rel = (T1_new - T1_curr) / T1_curr
        
        if delta_rel < 0 || rand() < exp(-delta_rel / T_fixed)
            b_curr = copy(b_new)
            T1_curr = T1_new
            
            if T1_curr < T1_best
                b_best = copy(b_curr)
                T1_best = T1_curr
                iters_since_best = 0 
            else
                iters_since_best += 1
            end
        else
            iters_since_best += 1
        end
        
        if iters_since_best >= patience
            break
        end
    end

    return b_best
end

function generate_ensemble(N::Int, M::Int, range_type::Symbol; max_iter::Int=5000000, patience::Int=500000, T_fixed::Float64=0.0)
    if range_type == :above
        b_min, b_max = 0.51, 0.99
    elseif range_type == :below
        b_min, b_max = 0.01, 0.49
    elseif range_type == :all
        b_min, b_max = 0.01, 0.99
    else
        throw(ArgumentError("range_type must be :above, :below, or :all"))
    end

    println("Generating $M sequences for N=$N, case: $range_type ...")
    all_realizations = zeros(M, N - 1)

    Threads.@threads for i in 1:M
        base_random_weights = b_min .+ (b_max - b_min) .* rand(N - 1)
        b_initial = center_high_edge_low_initialization(base_random_weights)
        b_opt = optimize_ordering(b_initial; max_iter=max_iter, patience=patience, T_fixed=T_fixed)
        all_realizations[i, :] = b_opt
    end

    return all_realizations
end

function get_global_best_realization(ens_matrix)
    best_idx = 1
    best_mfpt = Inf
    for i in 1:size(ens_matrix, 1)
        t = calculate_MFPT(ens_matrix[i, :])
        if t < best_mfpt
            best_mfpt = t
            best_idx = i
        end
    end
    return ens_matrix[best_idx, :]
end

# ==========================================
# 2. Data Generation & Processing
# ==========================================

N_sites = 64  
M_realizations = 100 # Increased for a more exhaustive ensemble
sites = 1:(N_sites - 1)

ens_all   = generate_ensemble(N_sites, M_realizations, :all)
ens_above = generate_ensemble(N_sites, M_realizations, :above)
ens_below = generate_ensemble(N_sites, M_realizations, :below)

best_all   = get_global_best_realization(ens_all)
best_above = get_global_best_realization(ens_above)
best_below = get_global_best_realization(ens_below)

# ==========================================
# 3. Plotting
# ==========================================

default(
    linewidth = 1.5,
    markershape = :none,
    markersize = 4,
    fontfamily = "Computer Modern",
    tickfontsize = 10,
    guidefontsize = 12,
    titlefontsize = 12,
    grid = true,
    gridalpha = 0.3,
    legend = false
)

# Plot (a) - Complete Case
p1 = plot(sites, best_all,
    color = :blue,
    xlabel = L"Lattice Site \ $i$",
    ylabel = L"Probability \ $b_i$",
    title = "(a) Complete Case",
    ylims = (0.0, 1.0)
)

# Plot (b) - Above Case
p2 = plot(sites, best_above,
    color = :red,
    xlabel = L"Lattice Site \ $i$",
    ylabel = L"Probability \ $b_i$",
    title = L"(b) Above Case ($b_i \geq 0.5$)",
    ylims = (0.5, 1.0)
)

# Plot (c) - Below Case
p3 = plot(sites, best_below,
    color = :green,
    xlabel = L"Lattice Site \ $i$",
    ylabel = L"Probability \ $b_i$",
    title = L"(c) Below Case ($b_i \leq 0.5$)",
    ylims = (0.0, 0.5)
)

# Combine into a 1x3 horizontal layout, widening the figure explicitly
fig = plot(p1, p2, p3, layout = (1, 3), size = (1200, 350), margin=5Plots.mm)


# Ensure the output directory exists before saving (avoids directory not found errors)
mkpath(figure_dir("publication", "abstract")); savefig(fig, joinpath(figure_dir("publication", "abstract"), "MFPT_configurations_horizontal.pdf"))