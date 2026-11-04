using GLMakie
using Combinatorics

# --- Mathematical Functions ---

function calculate_MFPT(b::Vector{Float64})
    L = length(b)
    N = L + 1
    
    total_sum = 0.0
    for m in 1:L
        inner_sum = 0.0
        for i in 1:(N - m)
            prod_val = 1.0
            for j in i:(i + m - 1)
                prod_val *= (1.0 - b[j]) / b[j]
            end
            inner_sum += prod_val
        end
        total_sum += inner_sum
    end
    
    return N + 2.0 * total_sum
end

function calculate_variance(b::Vector{Float64}, mfpt::Float64)
    L = length(b)
    N = L + 1

    B = vcat([1.0], b)
    ratio = (1.0 .- B) ./ B

    Q = zeros(Float64, N)
    Q[1] = 1.0 / B[1]
    for k in 2:N
        Q[k] = 1.0 / B[k] + ratio[k] * Q[k-1]
    end

    P = zeros(Float64, N)
    P[N] = Q[N]
    for i in (N-1):-1:1
        P[i] = Q[i] + P[i+1]
    end

    R = zeros(Float64, N)
    R[N] = 1.0
    for i in (N-1):-1:1
        R[i] = 1.0 + ratio[i+1] * R[i+1]
    end

    S = 0.0
    for i in 1:N
        S += P[i] * R[i] / B[i]
    end

    return 2.0 * S - mfpt - mfpt^2
end

# --- Pendulum Arrangement Generator ---
# --- Off-center Pendulum Arrangement Generator ---

# Largest value at index s, remaining values (descending) placed outward,
# alternating sides in proportion to the number of free slots on each side.
function create_offcenter_pendulum(L::Int, s::Int, custom_vals::Vector{Float64})
    all_vals = vcat(custom_vals, fill(0.5, L - length(custom_vals)))
    sort!(all_vals, rev = true)

    pend = zeros(Float64, L)
    pend[s] = all_vals[1]

    nl = s - 1      # free slots to the left
    nr = L - s      # free slots to the right
    l = 0           # placed on the left
    r = 0           # placed on the right

    for i in 2:L
        go_left = if l >= nl
            false
        elseif r >= nr
            true
        else
            (l + 1) / nl <= (r + 1) / nr
        end

        if go_left
            l += 1
            pend[s - l] = all_vals[i]
        else
            r += 1
            pend[s + r] = all_vals[i]
        end
    end

    return pend
end

# Scan every center s and keep the arrangement with the lowest MFPT
function best_offcenter_pendulum(L::Int, custom_vals::Vector{Float64})
    best_pend = fill(0.5, L)
    best_mfpt = Inf
    best_s = 1

    for s in 1:L
        pend = create_offcenter_pendulum(L, s, custom_vals)
        mfpt = calculate_MFPT(pend)
        if mfpt < best_mfpt
            best_mfpt = mfpt
            best_pend = pend
            best_s = s
        end
    end

    return best_pend, best_mfpt, best_s
end

# --- Optimized Permutation Search ---

function find_optimal_subspace(L::Int, custom_vals::Vector{Float64}, is_min::Bool)
    n = length(custom_vals)
    
    opt_mfpt = is_min ? Inf : -Inf
    opt_var = is_min ? Inf : -Inf
    best_b_mfpt = fill(0.5, L)
    best_b_var = fill(0.5, L)

    pos_combos = combinations(1:L, n)
    val_perms = collect(permutations(custom_vals))

    for pos in pos_combos
        for val_p in val_perms
            b = fill(0.5, L)
            for i in 1:n
                b[pos[i]] = val_p[i]
            end

            if b <= reverse(b)
                mfpt = calculate_MFPT(b)
                if (is_min && mfpt < opt_mfpt) || (!is_min && mfpt > opt_mfpt)
                    opt_mfpt = mfpt
                    best_b_mfpt = copy(b)
                end

                variance = calculate_variance(b, mfpt)
                if (is_min && variance < opt_var) || (!is_min && variance > opt_var)
                    opt_var = variance
                    best_b_var = copy(b)
                end
            end
        end
    end
    
    return best_b_mfpt, opt_mfpt, best_b_var, opt_var
end

# --- Interactive Makie App ---

function interactive_window(; L::Int = 15, n_vars::Int = 3)
    fig = Figure(size = (1000, 850))

    top_grid = GridLayout(fig[1, 1])
    
    slider_args = [(label = "Base b element $i", range = 0.01:0.01:0.99, startvalue = 0.2 + (i*0.1)) for i in 1:n_vars]
    lsgrid = SliderGrid(top_grid[1, 1], slider_args...)

    toggle_grid = GridLayout(top_grid[1, 2], tellwidth=false, halign=:right)
    Label(toggle_grid[1, 1], "Optimize for:", font=:bold)
    min_max_toggle = Toggle(toggle_grid[1, 2], active = true)
    
    is_min_obs = min_max_toggle.active
    Label(toggle_grid[1, 3], @lift($is_min_obs ? "MINIMUM" : "MAXIMUM"), width=80)

    custom_vals = lift([s.value for s in lsgrid.sliders]...) do vals...
        return Float64[vals...]
    end

    best_mfpt_config = Observable(fill(0.5, L))
    best_var_config = Observable(fill(0.5, L))
    pendulum_config = Observable(fill(0.5, L))
    
    mfpt_val = Observable(0.0)
    var_val = Observable(0.0)
    pend_mfpt_val = Observable(0.0)
    pend_s_val = Observable(1)

    onany(custom_vals, is_min_obs) do vals, is_min
        b_mfpt, m_mfpt, b_var, m_var = find_optimal_subspace(L, vals, is_min)
        best_mfpt_config[] = b_mfpt
        best_var_config[] = b_var
        mfpt_val[] = m_mfpt
        var_val[] = m_var
        
        pend, pend_m, pend_s = best_offcenter_pendulum(L, vals)
        pendulum_config[] = pend
        pend_mfpt_val[] = pend_m
        pend_s_val[] = pend_s
    end

    notify(is_min_obs)

    # Overlaid MFPT and Pendulum plot
    title_mfpt = @lift("Optimal MFPT = $(round($(mfpt_val), digits=2))  |  Off-center pendulum MFPT = $(round($(pend_mfpt_val), digits=2)) (s = $($pend_s_val))")
    ax_mfpt = Axis(fig[2, 1], title = title_mfpt, ylabel = "b value", xlabel = "Index")

    barplot!(ax_mfpt, 1:L, best_mfpt_config, color=:dodgerblue, label="Optimal")
    barplot!(ax_mfpt, 1:L, pendulum_config, color=:mediumpurple, width=0.3, label="Off-center pendulum")
    
    axislegend(ax_mfpt, position=:rt)
    ylims!(ax_mfpt, 0.0, 1.0)

    # Variance plot
    title_var = @lift("Optimal Array for $( $is_min_obs ? "MINIMUM" : "MAXIMUM" ) VARIANCE (Var = $(round($(var_val), digits=2)))")
    ax_var = Axis(fig[3, 1], title = title_var, ylabel = "b value", xlabel = "Index")
    barplot!(ax_var, 1:L, best_var_config, color=:tomato)
    ylims!(ax_var, 0.0, 1.0)

    display(fig)
    return fig
end

# Run the app
interactive_window(L=7, n_vars=7)