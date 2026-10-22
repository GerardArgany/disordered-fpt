using Random
using Plots
using Statistics

# 1. The O(N) Rate Engine
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
        for j in 1:y
            p_val *= x_buffer[y - j + 1]
            S_L += p_val
        end
        
        # Calculate Right-anchored contiguous sum (S_R)
        S_R = 0.0
        p_val = 1.0
        for j in 1:(L - y - 1)
            p_val *= x_buffer[y + 1 + j]
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
        
        if R == 0.0 # System completely frozen
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
        
        # Record state for observation checkpoints
        while obs_idx <= num_obs && T_obs[obs_idx] <= t_next
            b_sampled[obs_idx, :] .= b
            obs_idx += 1
        end
        
        if obs_idx > num_obs
            break
        end
        
        # Execute swap selection
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
        
        # Swap adjacent probabilities
        b[y_star], b[y_star+1] = b[y_star+1], b[y_star]
        t = t_next
    end
end

# 3. Ensemble Averaging
function simulate_ensemble(N::Int, alpha::Float64, beta::Float64, T_max::Float64, num_points::Int, M::Int, b_min::Float64, b_max::Float64)
    T_obs = collect(range(0.0, stop=T_max, length=num_points))
    B_sum = zeros(Float64, num_points, N)
    
    # Generate a single fixed random initial vector of probabilities restricted to [b_min, b_max]
    b_fixed = b_min .+ (b_max - b_min) .* rand(N) 
    
    # Pre-allocate thread-local variables for efficiency
    rates = zeros(Float64, N-1)
    x_buffer = zeros(Float64, N)
    b_sampled = zeros(Float64, num_points, N)
    
    for i in 1:M
        # Shuffle the starting positions of the probabilities for each trajectory
        b_start = shuffle(b_fixed) 
        gillespie_trajectory!(b_sampled, b_start, alpha, beta, T_obs, rates, x_buffer)
        B_sum .+= b_sampled
        
        if i % 100 == 0
            println("Completed $i / $M trajectories...")
        end
    end
    
    B_avg = B_sum ./ M
    return T_obs, B_avg, b_fixed
end

# --- Execution and Visualization ---
# Random.seed!(42)

N = 25          # Size of the chain
alpha = 1.0     # Base transition rate
beta = 10.0      # Inverse temperature
T_max = 250.0    # Total simulation time
num_points = 500
M = 1000

# Define the restricted range for b_j
b_min = 0.5
b_max = 1.0

T_obs, B_avg, b_fixed = simulate_ensemble(N, alpha, beta, T_max, num_points, M, b_min, b_max)

p = plot(title="Time Evolution of Expected Site Probabilities (β=$beta)", 
         xlabel="Time t", ylabel="Expectation Value <b_j>", 
         legend=:outertopright, size=(800, 500), gridalpha=0.3)

plot!(p, T_obs, B_avg[:, 1], label="Left Boundary (j=1)", lw=2, color=:blue)
plot!(p, T_obs, B_avg[:, div(N, 2)], label="Center Site (j=$(div(N, 2)))", lw=2, color=:orange)
plot!(p, T_obs, B_avg[:, 3], label="Third Site (j=3)", lw=2, color=:red)
plot!(p, T_obs, B_avg[:, 2], label="Second Site (j=2)", lw=2, color=:purple)
plot!(p, T_obs, B_avg[:, N], label="Right Boundary (j=$N)", lw=2, color=:green)

hline!(p, [mean(b_fixed)], label="Array Mean", ls=:dash, color=:black, alpha=0.5)

figures_dir = joinpath(@__DIR__, "figures", "annealing_site")
mkpath(figures_dir)

savefig(p, joinpath(figures_dir, "high_expected_site_probabilities_beta_$beta.pdf"))

# --- Steady-State Spatial Configuration ---

# 1. Average over the last 10% of the simulation window to smooth trajectory noise
window_size = div(num_points, 10)
b_steady_raw = vec(mean(B_avg[end-window_size+1:end, :], dims=1))

# 2. Match the exact real rates to the steady-state shape using rank ordering
b_matched = zeros(Float64, N)
sorted_indices = sortperm(b_steady_raw)      # Get indices that sort the noisy average
b_fixed_sorted = sort(b_fixed)               # Sort the available real rates
b_matched[sorted_indices] .= b_fixed_sorted  # Place real rates into their matched rank positions

# Plotting the Spatial Configuration
sites = 1:N

p2 = plot(title="Steady-State Spatial Configuration (β=$beta)",
          xlabel="Site Index (j)", ylabel="Probability b_j",
          legend=:topright, size=(800, 500), gridalpha=0.3)

# Plot the best-matched exact rates
plot!(p2, sites, b_matched, 
      st=:stepmid, ls=:dash, color=:black, lw=2, 
      label="Best-Matched Real Rates (b_fixed)")

# Scatter the exact rates on top for clear discrete points
scatter!(p2, sites, b_matched, 
         marker=:square, color=:black, alpha=0.5, ms=6,
         label="") # Empty label to avoid duplicating the legend entry

# Plot the raw time-averaged steady state
plot!(p2, sites, b_steady_raw, 
      marker=:circle, color=:purple, lw=2, ms=5,
      label="Raw Average <b_j> (Final Window)")

savefig(p2, joinpath(figures_dir, "high_steady_state_configuration_beta_$beta.pdf"))