using Random
using Plots
using Statistics

# 1. The O(N) Rate Engine (Symmetry Fixed)
function calculate_rates!(rates::Vector{Float64}, b::Vector{Float64}, x_buffer::Vector{Float64}, alpha::Float64, beta::Float64)
    L = length(b)
    
    # Precompute x_j = (1-b_j)/b_j into a preallocated buffer to avoid allocations
    @inbounds for i in 1:L
        x_buffer[i] = (1.0 - b[i]) / b[i]
    end
    
    @inbounds for y in 1:(L-1)
        # Calculate Left-anchored contiguous sum (S_L)
        S_L = 0.0
        p_val = 1.0
        for j in 1:(y - 1)                      # FIX: Loop exactly y-1 times
            p_val *= x_buffer[y - j]            # FIX: Start at y-1
            S_L += p_val
        end
        
        # Calculate Right-anchored contiguous sum (S_R)
        S_R = 0.0
        p_val = 1.0
        for j in 1:(L - y - 1)                  
            p_val *= x_buffer[y + 1 + j]        # Correctly starts at y+2
            S_R += p_val
        end
        
        # Exact decoupled energy change
        delta_Ty = 2.0 * (x_buffer[y+1] - x_buffer[y]) * (S_L - S_R)
        
        # Metropolis-Hastings transition rate
        rates[y] = alpha * min(1.0, exp(-beta * delta_Ty))
    end
end

# 2. The Gillespie Core Loop
function gillespie_trajectory!(b_sampled::Matrix{Float64}, b_init::Vector{Float64}, alpha::Float64, beta::Float64, T_obs, rates::Vector{Float64}, x_buffer::Vector{Float64})
    L = length(b_init)
    num_obs = length(T_obs)
    
    b = copy(b_init)
    t = 0.0
    obs_idx = 1
    
    while obs_idx <= num_obs
        calculate_rates!(rates, b, x_buffer, alpha, beta)
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
        
        if obs_idx > num_obs
            break
        end
        
        target = u2 * R
        cum_rate = 0.0
        y_star = 1
        for y in 1:(L-1)
            cum_rate += rates[y]
            if cum_rate >= target
                y_star = y
                break
            end
        end
        
        b[y_star], b[y_star+1] = b[y_star+1], b[y_star]
        t = t_next
    end
end

# 3. Ensemble Averaging (Now accepts pre-computed b_fixed)
function simulate_ensemble(N::Int, alpha::Float64, beta::Float64, T_max::Float64, num_points::Int, M::Int, b_fixed::Vector{Float64})
    T_obs = collect(range(0.0, stop=T_max, length=num_points))
    B_sum = zeros(Float64, num_points, N)
    B_sq_sum = zeros(Float64, num_points, N) 
    
    rates = zeros(Float64, N-1)
    x_buffer = zeros(Float64, N)
    b_sampled = zeros(Float64, num_points, N)
    
    for i in 1:M
        b_start = shuffle(b_fixed) 
        gillespie_trajectory!(b_sampled, b_start, alpha, beta, T_obs, rates, x_buffer)
        
        B_sum .+= b_sampled
        B_sq_sum .+= b_sampled .^ 2 
    end
    
    B_avg = B_sum ./ M
    B_var = max.((B_sq_sum ./ M) .- (B_avg .^ 2), 0.0)
    B_std = sqrt.(B_var)
    
    return T_obs, B_avg, B_std
end

# --- Execution and Visualization Loop ---
Random.seed!(42)

N = 25          
alpha = 1.0     
T_max = 250.0   
num_points = 500
M = 1000

# Generate the universal base sequence once in [0, 1]
b_base = rand(N)

figures_dir = joinpath(@__DIR__, "figures", "annealing_site")
mkpath(figures_dir)

regimes = [
    ("normal", 0.01, 0.99),
    ("low", 0.01, 0.49),
    ("high", 0.51, 0.99)
]
beta_values = [0.1, 1.0, 10.0, 10000.0]

for (case_name, b_min, b_max) in regimes
    # Map the universal base sequence to the specific regime bounds
    b_fixed = b_min .+ (b_max - b_min) .* b_base
    
    for beta in beta_values
        println("Simulating: $case_name regime with beta = $beta ...")
        
        T_obs, B_avg, B_std = simulate_ensemble(N, alpha, beta, T_max, num_points, M, b_fixed)
        
        # --- Time Evolution Plot ---
        p = plot(title="Time Evolution <b_j> ($case_name, β=$beta)", 
                 xlabel="Time t", ylabel="Expectation Value <b_j>", 
                 legend=:outertopright, size=(800, 500), gridalpha=0.3)

        # for i in 1:N
        #     plot!(p, T_obs, B_avg[:, i], ribbon=B_std[:, i], fillalpha=0.2, label="Site j=$i", lw=2)
        # end
        
        plot!(p, T_obs, B_avg[:, 1], ribbon=B_std[:, 1], fillalpha=0.2, label="Left Boundary (j=1)", lw=2, color=:blue)
        plot!(p, T_obs, B_avg[:, div(N, 2)], ribbon=B_std[:, div(N, 2)], fillalpha=0.2, label="Center Site (j=$(div(N, 2)))", lw=2, color=:orange)
        plot!(p, T_obs, B_avg[:, 3], ribbon=B_std[:, 3], fillalpha=0.2, label="Third Site (j=3)", lw=2, color=:red)
        plot!(p, T_obs, B_avg[:, 2], ribbon=B_std[:, 2], fillalpha=0.2, label="Second Site (j=2)", lw=2, color=:purple)
        plot!(p, T_obs, B_avg[:, N], ribbon=B_std[:, N], fillalpha=0.2, label="Right Boundary (j=$N)", lw=2, color=:green)
        
        hline!(p, [mean(b_fixed)], label="Array Mean", ls=:dash, color=:black, alpha=0.5)
        savefig(p, joinpath(figures_dir, "$(case_name)_time_evo_beta_$(beta).pdf"))

        # --- Steady-State Spatial Configuration ---
        window_size = div(num_points, 10)
        window_avg = B_avg[end-window_size+1:end, :]
        window_std = B_std[end-window_size+1:end, :]

        b_steady_raw = vec(mean(window_avg, dims=1))
        mean_of_vars = vec(mean(window_std .^ 2, dims=1))
        var_of_means = vec(var(window_avg, dims=1, corrected=false))
        b_steady_std = sqrt.(mean_of_vars .+ var_of_means)

        b_matched = zeros(Float64, N)
        sorted_indices = sortperm(b_steady_raw)      
        b_fixed_sorted = sort(b_fixed)               
        b_matched[sorted_indices] .= b_fixed_sorted  

        sites = 1:N
        p2 = plot(title="Steady-State Spatial Config ($case_name, β=$beta)",
                  xlabel="Site Index (j)", ylabel="Probability b_j",
                  legend=:topright, size=(800, 500), gridalpha=0.3)

        plot!(p2, sites, b_matched, 
              st=:stepmid, ls=:dash, color=:black, lw=2, 
              label="Best-Matched Real Rates")

        scatter!(p2, sites, b_matched, 
                 marker=:square, color=:black, alpha=0.5, ms=6, label="") 

        plot!(p2, sites, b_steady_raw, 
              ribbon=b_steady_std, fillalpha=0.2,
              marker=:circle, color=:purple, lw=2, ms=5,
              label="Raw Average <b_j> (Pooled Window)")

        savefig(p2, joinpath(figures_dir, "$(case_name)_steady_state_beta_$(beta).pdf"))
    end
end
println("All loop regimes and temperatures completed.")