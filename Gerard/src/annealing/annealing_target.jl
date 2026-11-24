include(joinpath(@__DIR__, "..", "paths.jl"))
using Random
using Plots
using Statistics
using Base.Threads

# ==============================================================================
# 1. Fast O(N) Exact Symmetric MFPT and 2nd Moment Calculator
# ==============================================================================
function calc_moments_sym!(b::Vector{Float64}, c_buf::Vector{Float64}, S_buf::Vector{Float64})
    L = length(b)
    T1 = 0.0
    T2_term1 = 0.0
    
    c_buf[1] = 1.0 / b[1]
    @inbounds for k in 2:L
        c_buf[k] = c_buf[k-1] + 1.0 / b[k]
    end
    
    S_buf[L] = c_buf[L]
    @inbounds for i in L-1:-1:1
        S_buf[i] = S_buf[i+1] + c_buf[i]
    end
    
    @inbounds for i in 1:L
        term = (L + 1 - i) / b[i]
        T1 += term
        T2_term1 += term * S_buf[i]
    end
    
    T2 = 2.0 * T2_term1 - T1
    return T1, T2
end

function calc_moments_sym(b::Vector{Float64})
    L = length(b)
    c_buf = zeros(Float64, L)
    S_buf = zeros(Float64, L)
    return calc_moments_sym!(b, c_buf, S_buf)
end

# ==============================================================================
# 2. Simulated Annealing (Fast Scout to map degenerate T2 bounds)
# ==============================================================================
function anneal_to_target_T1(b_init::Vector{Float64}, T1_target::Float64; max_iter=10000, initial_temp=0.1, cooling=0.999)
    L = length(b_init)
    b_curr = copy(b_init)
    b_best = copy(b_init)
    
    T1_curr, T2_curr = calc_moments_sym(b_curr)
    E_curr = ((T1_curr - T1_target) / T1_target)^2
    E_best = E_curr
    T1_best, T2_best = T1_curr, T2_curr
    
    temp = initial_temp
    for iter in 1:max_iter
        idx = rand(1:L-1)
        b_curr[idx], b_curr[idx+1] = b_curr[idx+1], b_curr[idx]
        
        T1_new, T2_new = calc_moments_sym(b_curr)
        E_new = ((T1_new - T1_target) / T1_target)^2
        
        delta_E = E_new - E_curr
        if delta_E < 0 || rand() < exp(-delta_E / temp)
            E_curr = E_new
            if E_curr < E_best
                E_best = E_curr
                b_best .= b_curr
                T1_best, T2_best = T1_new, T2_new
            end
        else
            b_curr[idx], b_curr[idx+1] = b_curr[idx+1], b_curr[idx]
        end
        temp = max(temp * cooling, 1e-7)
    end
    return b_best, E_best, T1_best, T2_best
end

# ==============================================================================
# 3. Dual-Constraint Gillespie Engine
# ==============================================================================
function calculate_rates_sym_moments!(rates::Vector{Float64}, b::Vector{Float64}, alpha::Float64, beta1::Float64, beta2::Float64, T1_target::Float64, T2_target::Float64, E_curr::Float64, c_buf::Vector{Float64}, S_buf::Vector{Float64})
    L = length(b)
    @inbounds for y in 1:(L-1)
        b[y], b[y+1] = b[y+1], b[y]
        T1_new, T2_new = calc_moments_sym!(b, c_buf, S_buf)
        
        E_new = beta1 * ((T1_new - T1_target) / T1_target)^2 + beta2 * ((T2_new - T2_target) / T2_target)^2
        delta_E = E_new - E_curr
        
        rates[y] = alpha * min(1.0, exp(-delta_E))
        b[y], b[y+1] = b[y+1], b[y]
    end
end

function gillespie_trajectory_sym_moments!(b_sampled::Matrix{Float64}, b_init::Vector{Float64}, alpha::Float64, beta1::Float64, beta2::Float64, T_obs::Vector{Float64}, rates::Vector{Float64}, c_buf::Vector{Float64}, S_buf::Vector{Float64}, T1_target::Float64, T2_target::Float64)
    L = length(b_init)
    num_obs = length(T_obs)
    b = copy(b_init)
    t = 0.0
    obs_idx = 1
    
    T1_curr, T2_curr = calc_moments_sym!(b, c_buf, S_buf)
    E_curr = beta1 * ((T1_curr - T1_target) / T1_target)^2 + beta2 * ((T2_curr - T2_target) / T2_target)^2
    
    while obs_idx <= num_obs
        calculate_rates_sym_moments!(rates, b, alpha, beta1, beta2, T1_target, T2_target, E_curr, c_buf, S_buf)
        R = sum(rates)
        
        if R == 0.0 
            while obs_idx <= num_obs
                b_sampled[obs_idx, :] .= b
                obs_idx += 1
            end
            break
        end
        
        u1 = rand()
        u2 = rand()
        dt = -log(u1) / R
        t_next = t + dt
        
        while obs_idx <= num_obs && T_obs[obs_idx] <= t_next
            b_sampled[obs_idx, :] .= b
            obs_idx += 1
        end
        
        if obs_idx > num_obs; break; end
        
        target = u2 * R
        cum_rate = 0.0
        y_star = 1
        for y in 1:(L-1)
            cum_rate += rates[y]
            if cum_rate >= target
                y_star = y; break
            end
        end
        
        b[y_star], b[y_star+1] = b[y_star+1], b[y_star]
        T1_curr, T2_curr = calc_moments_sym!(b, c_buf, S_buf)
        E_curr = beta1 * ((T1_curr - T1_target) / T1_target)^2 + beta2 * ((T2_curr - T2_target) / T2_target)^2
        t = t_next
    end
end

function simulate_ensemble_sym_moments(L::Int, alpha::Float64, beta1::Float64, beta2::Float64, T_max_time::Float64, num_points::Int, M::Int, b_fixed::Vector{Float64}, T1_target::Float64, T2_target::Float64)
    T_obs = collect(range(0.0, stop=T_max_time, length=num_points))
    all_sampled = [zeros(Float64, num_points, L) for _ in 1:M]
    
    Threads.@threads for i in 1:M
        rates = zeros(Float64, L-1)
        c_buf = zeros(Float64, L)
        S_buf = zeros(Float64, L)
        b_start = shuffle(b_fixed) 
        gillespie_trajectory_sym_moments!(all_sampled[i], b_start, alpha, beta1, beta2, T_obs, rates, c_buf, S_buf, T1_target, T2_target)
    end
    
    B_sum = zeros(Float64, num_points, L)
    B_sq_sum = zeros(Float64, num_points, L) 
    for i in 1:M
        B_sum .+= all_sampled[i]
        B_sq_sum .+= all_sampled[i] .^ 2 
    end
    
    B_avg = B_sum ./ M
    B_var = max.((B_sq_sum ./ M) .- (B_avg .^ 2), 0.0)
    B_std = sqrt.(B_var)
    return T_obs, B_avg, B_std
end

# ==============================================================================
# Execution Setup
# ==============================================================================
Random.seed!(42)

L = 25
b_fixed = rand(L) .* 0.5

b_min_config = sort(b_fixed, rev=true)
b_max_config = sort(b_fixed, rev=false)

T1_min, _ = calc_moments_sym(b_min_config)
T1_max, _ = calc_moments_sym(b_max_config)

targets_T1 = range(T1_min, stop=T1_max, length=10)

dir_dual_gillespie = figure_dir("annealing", "target", "gillespie")
mkpath(dir_dual_gillespie)

# Gillespie parameters (Reduced slightly for 40 total ensemble simulations)
alpha = 1.0
T_max_time = 100.0  
num_points = 200
M = 200             

beta1 = 15000.0 
beta2 = 15000.0 

# ==============================================================================
# Simulation Loop (Wrapped in let to avoid soft scope issues)
# ==============================================================================
let
    for (idx, T1_tgt) in enumerate(targets_T1)
        println("Processing Target $idx/10: T1 ≈ $(round(T1_tgt, digits=1)) ...")
        
        # 1. SCOUTING: Find natural T2 bounds using fast SA
        results_T1 = []
        for run in 1:30
            b_start = shuffle(b_fixed)
            b_res, E_res, T1_res, T2_res = anneal_to_target_T1(b_start, T1_tgt)
            push!(results_T1, (T2_res, b_res))
        end
        
        top_T2s = [res[1] for res in results_T1]
        local targets_T2 = range(minimum(top_T2s), stop=maximum(top_T2s), length=4)
        
        # 2. GILLESPIE: Run the full ensemble for the 4 specific (T1, T2) targets
        subplots_dual = []
        for (j, T2_tgt) in enumerate(targets_T2)
            
            T_obs, B_avg, B_std = simulate_ensemble_sym_moments(L, alpha, beta1, beta2, T_max_time, num_points, M, b_fixed, T1_tgt, T2_tgt)
            
            # Extract steady state
            window_size = div(num_points, 10)
            window_avg = B_avg[end-window_size+1:end, :]
            window_std = B_std[end-window_size+1:end, :]
            
            b_steady_raw = vec(mean(window_avg, dims=1))
            mean_of_vars = vec(mean(window_std .^ 2, dims=1))
            var_of_means = vec(var(window_avg, dims=1, corrected=false))
            b_steady_std = sqrt.(mean_of_vars .+ var_of_means)
            
            # Snap to the real discrete values
            b_matched = zeros(Float64, L)
            sorted_indices = sortperm(b_steady_raw)      
            b_matched[sorted_indices] .= sort(b_fixed)
            
            # Calculate the ACTUAL moments of the snapped steady-state configuration
            act_T1, act_T2 = calc_moments_sym(b_matched)
            
            title_str = "T1: Tgt=$(round(T1_tgt, digits=1)) | Act=$(round(act_T1, digits=1))\n" *
                        "T2: Tgt=$(round(T2_tgt, digits=0)) | Act=$(round(act_T2, digits=0))"
                        
            sub_p = plot(1:L, b_matched, st=:stepmid, ls=:dash, color=:black, lw=2, label="Best-Matched Config",
                         title=title_str, titlefontsize=9,
                         xlabel= j > 2 ? "Site Index (i)" : "", 
                         ylabel= j % 2 == 1 ? "Probability b_i" : "",
                         legend=false, gridalpha=0.3)
                         
            scatter!(sub_p, 1:L, b_matched, marker=:square, color=:black, alpha=0.5, ms=4) 
            plot!(sub_p, 1:L, b_steady_raw, ribbon=b_steady_std, fillalpha=0.2, marker=:circle, color=:purple, lw=2, ms=4, label="Ensemble <b_i>")
            
            push!(subplots_dual, sub_p)
        end
        
        # 3. PLOT: Combine the 4 Gillespie steady states into a 2x2 grid
        p_grid = plot(subplots_dual..., layout=(2, 2), size=(1000, 800),
                      plot_title="Gillespie Steady States (Fixed T1 ≈ $(round(T1_tgt, digits=1)))",
                      plot_titlefontsize=14, margin=5Plots.mm)
        savefig(p_grid, joinpath(dir_dual_gillespie, "gillespie_dual_$idx.pdf"))
    end
    println("All Gillespie steady-state dual-target grids successfully saved to: ", dir_dual_gillespie)
end