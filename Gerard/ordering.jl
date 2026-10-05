ENV["GKSwstype"] = "100"
using Plots

figures_dir = joinpath(@__DIR__, "figures")

# Your original forward step function
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

# New function to calculate the S_i cascade backward
function calculate_S(b::Vector{Float64})
    N_minus_1 = length(b)
    S = zeros(Float64, N_minus_1)
    
    # Base case at the right boundary
    S[end] = 1.0
    
    # Backward recurrence: S_i = 1 + x_{i+1} * S_{i+1}
    for i in (N_minus_1 - 1):-1:1
        x_next = (1.0 - b[i+1]) / b[i+1]
        S[i] = 1.0 + x_next * S[i+1]
    end
    
    return S
end

function simulate_fpt(b::Vector{Float64})
    position = 1
    time = 0
    target = length(b) + 1

    while position < target
        time += 1
        if rand() < b[position]
            position += 1
        elseif position > 1
            position -= 1
        end
    end

    return time
end

function simulated_mfpt(b::Vector{Float64}, n_runs::Int)
    total_time = 0
    for _ in 1:n_runs
        total_time += simulate_fpt(b)
    end
    return total_time / n_runs
end

function compare_orderings_and_plot(N::Int; n_runs::Int = 10_000)
    # Generate random probabilities spanning [0, 1]
    b_random = rand(N-1)#.*0.75 .+ 0.2  # Ensure probabilities are in [0.3, 0.7] for stability
    
    b_ascending = sort(b_random)
    b_descending = sort(b_random, rev=true)
    
    # Calculate MFPTs
    T1_asc = calculate_MFPT(b_ascending)
    T1_rand = calculate_MFPT(b_random)
    T1_desc = calculate_MFPT(b_descending)
    
    # Calculate S_i arrays
    S_asc = calculate_S(b_ascending)
    S_rand = calculate_S(b_random)
    S_desc = calculate_S(b_descending)
    
    println("MFPT Comparison for N = $N")
    println("---------------------------------")
    println("Random order T1     : ", T1_rand)
    println("Ascending order T1  : ", T1_asc)
    println("Descending order T1 : ", T1_desc)

    simulated_rand = simulated_mfpt(b_random, n_runs)
    simulated_asc = simulated_mfpt(b_ascending, n_runs)
    simulated_desc = simulated_mfpt(b_descending, n_runs)

    println("\nSimulated MFPTs ($n_runs runs)")
    println("Random order        : ", simulated_rand)
    println("Ascending order     : ", simulated_asc)
    println("Descending order    : ", simulated_desc)
    
    # Plot S_i vs i with a logarithmic y-axis
    p = plot(1:(N-1), S_asc, label="Ascending b_i", linewidth=2, yaxis=:log10, color=:blue)
    plot!(p, 1:(N-1), S_rand, label="Random b_i", linewidth=2, color=:green)
    plot!(p, 1:(N-1), S_desc, label="Descending b_i", linewidth=2, color=:red)
    
    xlabel!(p, "State i")
    ylabel!(p, "S_i (Log10 Scale)")
    title!(p, "Cascade Multiplier S_i vs State i")
    
    # Ensure the directory exists, then save the plot
    mkpath(figures_dir)
    output_path = joinpath(figures_dir, "S_i_cascade_comparison.pdf")
    savefig(p, output_path)
    println("Plot successfully saved to ", output_path)
end

compare_orderings_and_plot(50)