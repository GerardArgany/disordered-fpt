ENV["GKSwstype"] = "100"
using Plots
using Random
using LinearAlgebra
using Statistics
using FFTW

figures_dir = joinpath(@__DIR__, "figures", "trajectory_ensemble")
mkpath(figures_dir)

function calculate_MFPT(b::Vector{Float64})
    n_minus_1 = length(b)
    tau = zeros(Float64, n_minus_1)

    tau[1] = 1.0 / b[1]
    for m in 2:n_minus_1
        ratio = (1.0 - b[m]) / b[m]
        tau[m] = (1.0 / b[m]) + ratio * tau[m - 1]
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
    max_iter::Int=1_000_000,
    record_every::Int=100,
)
    b_current = copy(b_initial)
    current_mfpt = calculate_MFPT(b_current)
    best_mfpt = current_mfpt
    best_ordering = copy(b_current)
    temperature = 1.0
    cooling_rate = 0.9994
    convergence = Float64[best_mfpt]

    for iteration in 1:max_iter
        index_one, index_two = rand(1:length(b_current), 2)
        while index_one == index_two
            index_one, index_two = rand(1:length(b_current), 2)
        end

        proposed_ordering = copy(b_current)
        proposed_ordering[index_one], proposed_ordering[index_two] =
            proposed_ordering[index_two], proposed_ordering[index_one]
        proposed_mfpt = calculate_MFPT(proposed_ordering)
        relative_change = (proposed_mfpt - current_mfpt) / current_mfpt

        if relative_change < 0 || rand() < exp(-relative_change / temperature)
            b_current = proposed_ordering
            current_mfpt = proposed_mfpt
            if current_mfpt < best_mfpt
                best_mfpt = current_mfpt
                best_ordering = copy(b_current)
            end
        end

        temperature *= cooling_rate
        if iteration % record_every == 0
            push!(convergence, best_mfpt)
        end
    end

    return best_ordering, best_mfpt, convergence
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
        elseif position > 1
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

function savitzky_golay_smooth(
    values::Vector{Float64};
    window_size::Int=7,
    polynomial_order::Int=2,
)
    isodd(window_size) || throw(ArgumentError("Savitzky-Golay window must be odd"))
    polynomial_order < window_size ||
        throw(ArgumentError("Polynomial order must be smaller than the window size"))

    half_window = window_size ÷ 2
    offsets = collect(-half_window:half_window)
    design = Float64[offset^power for offset in offsets, power in 0:polynomial_order]
    coefficient_matrix = (design' * design) \ design'
    smoothing_coefficients = coefficient_matrix[1, :]
    smoothed = similar(values)

    for position in eachindex(values)
        neighborhood = [
            values[clamp(position + offset, firstindex(values), lastindex(values))]
            for offset in offsets
        ]
        smoothed[position] = dot(smoothing_coefficients, neighborhood)
    end

    return smoothed
end

function wave_packet_envelope(b::Vector{Float64})
    analysis_signal = b[2:end]
    centered_signal = analysis_signal .- mean(analysis_signal)
    abs_signal = abs.(centered_signal)

    # Use a short Savitzky-Golay filter to preserve small packets and valley positions.
    smoothed_abs = savitzky_golay_smooth(
        abs_signal;
        window_size=7,
        polynomial_order=2,
    )
    
    return collect(2:length(b)), abs_signal, smoothed_abs
end

function wave_packet_boundaries(
    b::Vector{Float64};
    min_distance::Int=5,
    max_valley_fraction::Float64=0.5,
)
    # Unpack the updated return signature
    positions, _, smoothed_abs = wave_packet_envelope(b)
    valley_threshold = max_valley_fraction * maximum(smoothed_abs)

    valley_candidates = Int[]
    if smoothed_abs[1] <= smoothed_abs[2] && smoothed_abs[1] <= valley_threshold
        push!(valley_candidates, 1)
    end
    for index in 2:(length(smoothed_abs) - 1)
        if smoothed_abs[index] <= smoothed_abs[index - 1] && smoothed_abs[index] <= smoothed_abs[index + 1]
            if smoothed_abs[index] <= valley_threshold
                push!(valley_candidates, index)
            end
        end
    end
    last_index = length(smoothed_abs)
    if smoothed_abs[last_index] <= smoothed_abs[last_index - 1] &&
       smoothed_abs[last_index] <= valley_threshold
        push!(valley_candidates, last_index)
    end

    # Select the deepest valleys first, enforcing the minimum edge separation.
    ordered_candidates = sort(valley_candidates, by=index -> smoothed_abs[index])
    selected_valleys = Int[]
    for candidate in ordered_candidates
        if all(abs(candidate - selected) >= min_distance for selected in selected_valleys)
            push!(selected_valleys, candidate)
        end
    end
    return sort(positions[selected_valleys])
end

function run_trajectory_case(
    N::Int;
    b_min::Float64,
    b_max::Float64,
    case_name::String,
)
    base_weights = b_min .+ (b_max - b_min) .* rand(N - 1)
    b_initial = center_high_edge_low_initialization(base_weights)
    b_optimal, optimal_mfpt, _ = optimize_ordering(b_initial)
    b_ascending = sort(b_optimal)
    b_descending = sort(b_optimal, rev=true)

    println("N = $N, case = $case_name")
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
        plot_title="N = $N particle trajectories: $case_name",
    )
    savefig(trajectory_comparison,
        joinpath(figures_dir, "N$(N)_$(case_name)_trajectory_comparison.pdf"))

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
        label=ensemble_names,
        xlabel="Particle state",
        ylabel="Expected visits per trajectory",
        title="State visits over $ensemble_size trajectories (N = $N, $case_name)",
        legend=:topright,
        grid=true,
        ylims=(0.0, 1.05 * maximum(ensemble_visit_matrix)),
        size=(1200, 700),
    )
    savefig(ensemble_histogram,
        joinpath(figures_dir, "N$(N)_$(case_name)_ensemble_state_visits.pdf"))
end

function plot_optimal_trajectory_ensemble(
    N::Int;
    n_trajectories::Int=10,
    b_min::Float64=0.51,
    b_max::Float64=0.99,
    max_iter::Int=1_000_000,
)
    base_weights = b_min .+ (b_max - b_min) .* rand(N - 1)
    b_initial = center_high_edge_low_initialization(base_weights)
    b_optimal, optimal_mfpt, _ = optimize_ordering(b_initial; max_iter=max_iter)

    trajectory_plots = Any[]
    for trajectory_index in 1:n_trajectories
        times, positions = sample_trajectory(b_optimal)
        println("N = $N optimal trajectory $trajectory_index FPT: ", times[end])
        trajectory_plot = plot(
            times,
            positions,
            color=:steelblue,
            linewidth=1.2,
            xlabel="Time step",
            ylabel="Position",
            title="Trajectory $trajectory_index, FPT=$(times[end])",
            legend=false,
            grid=true,
            ylims=(0, N),
        )
        push!(trajectory_plots, trajectory_plot)
    end

    optimal_trajectories_plot = plot(
        trajectory_plots...,
        layout=(5, 2),
        size=(1200, 1500),
        plot_title="Ten trajectories from optimal ordering, N = $N, b_i > 0.5, MFPT=$(round(optimal_mfpt, digits=2))",
    )
    savefig(optimal_trajectories_plot,
        joinpath(figures_dir, "N$(N)_b_above_half_optimal_ten_trajectories.pdf"))
    println("Saved N = $N optimal trajectory ensemble.")
end

function plot_n200_optimal_packets_and_trajectories(
    N::Int=200;
    n_trajectories::Int=10,
    b_min::Float64=0.51,
    b_max::Float64=0.99,
    max_iter::Int=1_000_000,
)
    base_weights = b_min .+ (b_max - b_min) .* rand(N - 1)
    b_initial = center_high_edge_low_initialization(base_weights)
    b_optimal, optimal_mfpt, _ = optimize_ordering(b_initial; max_iter=max_iter)
    packet_boundaries = wave_packet_boundaries(b_optimal)

    println("N = $N above-half optimal MFPT: ", round(optimal_mfpt, digits=4))
    println("Detected packet boundaries: ", packet_boundaries)
    packet_sizes = length(packet_boundaries) >= 2 ? diff(packet_boundaries) : Int[]
    println("Detected packet sizes: ", packet_sizes)
    envelope_positions, abs_signal, smoothed_abs = wave_packet_envelope(b_optimal)
    envelope_range = maximum(smoothed_abs) - minimum(smoothed_abs)
    weight_range = maximum(b_optimal) - minimum(b_optimal)
    scaled_envelope = if envelope_range > 0
        minimum(b_optimal) .+
        (smoothed_abs .- minimum(smoothed_abs)) .* weight_range ./ envelope_range
    else
        fill(mean(b_optimal), length(smoothed_abs))
    end

    weights_plot = plot(
        1:(N - 1),
        b_optimal,
        color=:blue,
        linewidth=1.5,
        xlabel="State i",
        ylabel="b_i",
        ylims=(0.0, 1.0),
        title="Optimized weights with wave-packet boundaries, N = $N",
        legend=:topright,
        grid=true,
    )
    plot!(weights_plot,
        envelope_positions,
        scaled_envelope,
        color=:purple,
        linewidth=2.0,
        linestyle=:dash,
        label="Scaled smoothed envelope")
    packet_intervals = zip(
        packet_boundaries[1:max(0, length(packet_boundaries) - 1)],
        packet_boundaries[2:end])
    for (packet_index, (left_boundary, right_boundary)) in enumerate(packet_intervals)
        vspan!(weights_plot,
            [left_boundary, right_boundary],
            fillalpha=0.10,
            color=:darkorange,
            label=packet_index == 1 ? "Detected packet" : false)
        vline!(weights_plot,
            [left_boundary, right_boundary],
            color=:darkorange,
            linestyle=:dash,
            linewidth=1.2,
            label=false)
    end
    scatter!(weights_plot,
        packet_boundaries,
        b_optimal[packet_boundaries],
        color=:darkorange,
        markersize=4,
        label="Packet edge",
        legend=:bottomright)
    savefig(weights_plot,
        joinpath(figures_dir, "N$(N)_b_above_half_optimal_weights_with_packets.pdf"))

    centered_weights = b_optimal[2:end] .- mean(b_optimal[2:end])
    spectrum = fft(centered_weights)
    n_weights = length(centered_weights)
    positive_indices = 2:div(n_weights, 2)
    frequencies = (positive_indices .- 1) ./ n_weights
    amplitudes = abs.(spectrum[positive_indices])
    dominant_frequency_index = argmax(amplitudes)
    dominant_frequency = frequencies[dominant_frequency_index]

    fft_plot = plot(
        frequencies,
        amplitudes,
        color=:purple,
        linewidth=1.5,
        xlabel="Spatial frequency (cycles per state)",
        ylabel="FFT amplitude",
        title="FFT of optimized weights, N = $N",
        legend=:topright,
        grid=true,
    )
    vline!(fft_plot,
        [dominant_frequency],
        color=:black,
        linestyle=:dash,
        label="Dominant frequency = $(round(dominant_frequency, digits=4))")
    savefig(fft_plot,
        joinpath(figures_dir, "N$(N)_b_above_half_optimal_weights_fft.pdf"))
    println("Dominant spatial frequency: ", dominant_frequency)

    envelope_plot = plot(
        envelope_positions,
        abs_signal,
        color=:lightgray,
        linewidth=1.2,
        label="|b_i - mean(b)|",
        xlabel="State i",
        ylabel="Amplitude",
        title="Absolute centered signal and smoothed edges, N = $N",
        legend=:topright,
        grid=true,
    )
    plot!(envelope_plot,
        envelope_positions,
        smoothed_abs,
        color=:darkorange,
        linewidth=2.0,
        label="Smoothed signal"
    )
    for boundary in packet_boundaries
        vline!(envelope_plot,
            [boundary],
            color=:black,
            linestyle=:dash,
            linewidth=1.5,
            label=false)
    end
    savefig(envelope_plot, joinpath(figures_dir, "N$(N)_b_above_half_packet_abs_signal.pdf"))

    trajectory_plots = Any[]
    for trajectory_index in 1:n_trajectories
        times, positions = sample_trajectory(b_optimal)
        println("N = $N optimal trajectory $trajectory_index FPT: ", times[end])
        trajectory_plot = plot(
            times,
            positions,
            color=:steelblue,
            linewidth=1.0,
            xlabel="Time step",
            ylabel="Position",
            title="Trajectory $trajectory_index, FPT=$(times[end])",
            legend=false,
            grid=true,
            ylims=(0, N),
        )
        for boundary in packet_boundaries
            hline!(trajectory_plot,
                [boundary],
                color=:darkorange,
                linestyle=:dash,
                linewidth=0.8,
                label=false)
        end
        push!(trajectory_plots, trajectory_plot)
    end

    trajectory_ensemble_plot = plot(
        trajectory_plots...,
        layout=(5, 2),
        size=(1200, 1500),
        plot_title="Ten trajectories from one optimized ordering, N = $N, b_i > 0.5",
    )
    savefig(trajectory_ensemble_plot,
        joinpath(figures_dir, "N$(N)_b_above_half_optimal_ten_trajectories_with_packets.pdf"))
end

run_trajectory_case(15; b_min=0.01, b_max=0.99, case_name="normal")
run_trajectory_case(15; b_min=0.51, b_max=0.99, case_name="b_above_half")
# run_trajectory_case(15; b_min=0.01, b_max=0.49, case_name="b_below_half")

plot_n200_optimal_packets_and_trajectories(300;
    n_trajectories=10,
    b_min=0.51,
    b_max=0.99)

