include(joinpath(@__DIR__, "..", "paths.jl"))
ENV["GKSwstype"] = "100"
using Plots
using Plots.PlotMeasures
using Random
using Printf
using Statistics
using Base.Threads

figures_dir = figure_dir("landscape", "phase_transition")
println("Active Threads: ", nthreads())

const LOG2 = log(2.0)

# ==============================================================================
# 1. Exact O(N^2) Log-Space Cocktail-Shaker Adjacent Descent
# ==============================================================================
@inline function logaddexp(a::Float64, b::Float64)
    m = max(a, b)
    d = abs(a - b)
    return d > 37.0 ? m : m + log1p(exp(-d))
end

function sweep_lr!(
    p::Vector{Int},
    b_base::Vector{Float64},
    log_b::Vector{Float64},
    log_x::Vector{Float64},
    log_tau::Vector{Float64},
    log_R::Vector{Float64},
)
    K = length(p)
    swaps = 0
    @inbounds for k in 1:(K - 1)
        ik  = p[k]
        ik1 = p[k + 1]
        bk  = b_base[ik]
        bk1 = b_base[ik1]

        L_k = (k == 1) ? 0.0 : logaddexp(0.0, log_tau[k - 1])
        W_k = LOG2 + log_R[k + 1]

        if abs(L_k - W_k) > 1e-12
            if (bk > bk1 && L_k < W_k) || (bk < bk1 && L_k > W_k)
                p[k] = ik1
                p[k + 1] = ik
                ik = ik1
                swaps += 1
            end
        end

        log_tau[k] = (k == 1) ? -log_b[ik] : logaddexp(-log_b[ik], log_x[ik] + log_tau[k - 1])
    end
    return swaps
end

function sweep_rl!(
    p::Vector{Int},
    b_base::Vector{Float64},
    log_x::Vector{Float64},
    log_tau::Vector{Float64},
    log_R::Vector{Float64},
)
    K = length(p)
    swaps = 0
    log_R[K] = 0.0
    @inbounds for k in (K - 1):-1:1
        ik  = p[k]
        ik1 = p[k + 1]
        bk  = b_base[ik]
        bk1 = b_base[ik1]

        L_k = (k == 1) ? 0.0 : logaddexp(0.0, log_tau[k - 1])
        W_k = LOG2 + log_R[k + 1]

        if abs(L_k - W_k) > 1e-12
            if (bk > bk1 && L_k < W_k) || (bk < bk1 && L_k > W_k)
                p[k] = ik1
                p[k + 1] = ik
                ik1 = ik
                swaps += 1
            end
        end

        log_R[k] = logaddexp(0.0, log_x[ik1] + log_R[k + 1])
    end
    return swaps
end

function exact_adjacent_descent(
    b_base::Vector{Float64},
    log_b::Vector{Float64},
    log_x::Vector{Float64},
    p_start::Vector{Int},
    start_lr::Bool,
)
    K = length(b_base)
    p = copy(p_start)
    log_tau = Vector{Float64}(undef, K)
    log_R   = Vector{Float64}(undef, K)

    max_passes = 4 * K
    converged = false

    if start_lr
        log_R[K] = 0.0
        @inbounds for m in (K - 1):-1:1
            idx_next = p[m + 1]
            log_R[m] = logaddexp(0.0, log_x[idx_next] + log_R[m + 1])
        end
        for _ in 1:max_passes
            if sweep_lr!(p, b_base, log_b, log_x, log_tau, log_R) == 0
                converged = true; break
            end
            if sweep_rl!(p, b_base, log_x, log_tau, log_R) == 0
                converged = true; break
            end
        end
    else
        log_tau[1] = -log_b[p[1]]
        @inbounds for m in 2:(K - 1)
            idx = p[m]
            log_tau[m] = logaddexp(-log_b[idx], log_x[idx] + log_tau[m - 1])
        end
        for _ in 1:max_passes
            if sweep_rl!(p, b_base, log_x, log_tau, log_R) == 0
                converged = true; break
            end
            if sweep_lr!(p, b_base, log_b, log_x, log_tau, log_R) == 0
                converged = true; break
            end
        end
    end

    return p, converged
end

# ==============================================================================
# 2. Maximal-Support Sampling & Observables
# ==============================================================================
function sample_maximal_support(rng::AbstractRNG, K::Int, mu::Float64)
    lo = max(0.0, 2.0 * mu - 1.0)
    hi = min(1.0, 2.0 * mu)
    b = lo .+ rand(rng, K) .* (hi - lo)
    return clamp.(b, 1e-12, 1.0 - 1e-12)
end

function count_peaks(vals::Vector{Float64})
    K = length(vals)
    K == 1 && return 1
    peaks = 0
    if vals[1] > vals[2]; peaks += 1; end
    @inbounds for i in 2:(K - 1)
        if vals[i] > vals[i - 1] && vals[i] > vals[i + 1]
            peaks += 1
        end
    end
    if vals[K] > vals[K - 1]; peaks += 1; end
    return peaks
end

function evaluate_disorder_sample(b_base::Vector{Float64}; samples::Int=300, seed::Int=1000)
    K = length(b_base)
    log_b = log.(b_base)
    log_x = log.(1.0 .- b_base) .- log_b

    traps = Vector{Vector{Int}}(undef, samples)
    conv_flags = Vector{Bool}(undef, samples)

    @threads for s in 1:samples
        rng_s = Xoshiro(seed + s)
        p_start = shuffle(rng_s, 1:K)
        traps[s], conv_flags[s] = exact_adjacent_descent(b_base, log_b, log_x, p_start, rand(rng_s, Bool))
    end

    n_unconverged = count(!, conv_flags)

    mu_b = mean(b_base)
    sig_b = std(b_base, corrected=false)
    S_base = (b_base .- mu_b) ./ sig_b

    mean_S = zeros(Float64, K)
    sum_b  = zeros(Float64, K)
    sum_b2 = zeros(Float64, K)
    total_peaks = 0
    unimodal_count = 0

    for p in traps
        bp = b_base[p]
        @inbounds for i in 1:K
            mean_S[i] += S_base[p[i]]
            val = bp[i]
            sum_b[i]  += val
            sum_b2[i] += val * val
        end
        pk = count_peaks(bp)
        total_peaks += pk
        if pk == 1
            unimodal_count += 1
        end
    end

    mean_S ./= samples
    mean_b = sum_b ./ samples
    std_b  = sqrt.(max.(sum_b2 ./ samples .- mean_b .^ 2, 0.0))

    q_bar = (samples * mean(mean_S .^ 2) - 1.0) / (samples - 1)
    excess_peak_density = ((total_peaks / samples) - 1.0) / K
    unimodal_frac = unimodal_count / samples

    return q_bar, excess_peak_density, unimodal_frac, mean_b, std_b, n_unconverged
end

# ==============================================================================
# 3. Finite-Size Scaling Sweep over mu in [0.05, 0.95]
# ==============================================================================
function run_mu_fss(;
    N_list::Vector{Int}=[25, 50, 100, 250, 500, 1000],
    mu_vals::Vector{Float64}=collect(range(0.05, 0.95, length=37)),
    samples::Int=300,
    disorder_realizations::Int=6,
)
    colors  = palette(:turbo, max(length(N_list), 2))
    markers = [:circle, :square, :diamond, :utriangle, :dtriangle, :pentagon, :star5, :hexagon]

    p_q    = plot(title="Inter-Valley Overlap q̄(μ)", xlabel="Mean Weight μ", ylabel="q̄",
                  ylims=(-0.05, 1.05), gridalpha=0.3, legend=:topright)
    p_pk   = plot(title="Excess Peak Density (⟨peaks⟩ - 1)/(N - 1)", xlabel="Mean Weight μ",
                  ylabel="Excess Peak Density", ylims=(-0.02, 0.50), gridalpha=0.3, legend=:topleft)
    p_uni  = plot(title="Unimodal Trap Fraction P(peaks = 1)", xlabel="Mean Weight μ",
                  ylabel="Fraction Unimodal", ylims=(-0.05, 1.05), gridalpha=0.3, legend=:topright)

    for plt in (p_q, p_pk, p_uni)
        vline!(plt, [0.25], color=:black, ls=:dash, lw=1.2, alpha=0.6, label="")
        vline!(plt, [0.50], color=:black, ls=:dot,  lw=1.5, alpha=0.7, label="")
        vline!(plt, [0.75], color=:gray,  ls=:dash, lw=1.0, alpha=0.5, label="")
    end

    master_rng = Xoshiro(42)

    for (ni, N) in enumerate(N_list)
        K = N - 1
        n_mu = length(mu_vals)
        q_mean = zeros(n_mu); q_std = zeros(n_mu)
        pk_mean = zeros(n_mu); pk_std = zeros(n_mu)
        uni_mean = zeros(n_mu)
        total_unconv = 0

        t_elapsed = @elapsed begin
            for (mi, mu) in enumerate(mu_vals)
                qs = Float64[]; pks = Float64[]; unis = Float64[]
                for r in 1:disorder_realizations
                    b_base = sample_maximal_support(master_rng, K, mu)
                    seed = 100_000 * N + 1_000 * mi + 100 * r
                    qb, epd, uf, _, _, unconv = evaluate_disorder_sample(b_base; samples=samples, seed=seed)
                    push!(qs, qb); push!(pks, epd); push!(unis, uf)
                    total_unconv += unconv
                end
                q_mean[mi]   = mean(qs);   q_std[mi]  = std(qs)
                pk_mean[mi]  = mean(pks);  pk_std[mi] = std(pks)
                uni_mean[mi] = mean(unis)
            end
        end

        @printf("Finished N = %4d in %.1f s (unconverged descents: %d)\n", N, t_elapsed, total_unconv)

        # Checkpoint each N immediately so completed runs are never lost
        ckpt_path = joinpath(figures_dir, "fss_N$(N).tsv")
        open(ckpt_path, "w") do io
            println(io, "mu\tq_mean\tq_std\tpk_mean\tpk_std\tuni_mean")
            for mi in 1:n_mu
                @printf(io, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                        mu_vals[mi], q_mean[mi], q_std[mi], pk_mean[mi], pk_std[mi], uni_mean[mi])
            end
        end

        c = colors[ni]
        m = markers[mod1(ni, length(markers))]
        plot!(p_q,   mu_vals, q_mean,   ribbon=q_std,  fillalpha=0.14, lw=2.2, color=c, marker=m, ms=3, label="N = $N")
        plot!(p_pk,  mu_vals, pk_mean,  ribbon=pk_std, fillalpha=0.14, lw=2.2, color=c, marker=m, ms=3, label="N = $N")
        plot!(p_uni, mu_vals, uni_mean, lw=2.2, color=c, marker=m, ms=3, label="N = $N")
    end

    p_grid = plot(
        p_q, p_pk, p_uni,
        layout=(1, 3),
        size=(1550, 480),
        margin=5Plots.mm,
        plot_title="Maximal-Support Uniform Sweep: b ∈ [max(0, 2μ-1), min(1, 2μ)] (Lines: μ = 0.25, 0.50, 0.75)",
        plot_titlefontsize=12,
    )

    outpath = joinpath(figures_dir, "maximal_support_mu_fss.pdf")
    savefig(p_grid, outpath)
    display(p_grid)
    println("Saved FSS figure to ", outpath)
end

# ==============================================================================
# 4. Quenched Spatial Profiles <b_i> ± sigma_i (Core + Wings Coexistence Check)
# ==============================================================================
function plot_spatial_profiles(;
    N::Int=100,
    mu_slices::Vector{Float64}=[0.20, 0.25, 0.35, 0.45, 0.50, 0.75],
    samples::Int=500,
)
    K = N - 1
    rng = Xoshiro(1234)
    subplots = []

    for (si, mu) in enumerate(mu_slices)
        b_base = sample_maximal_support(rng, K, mu)
        q_bar, epd, _, mean_b, std_b, _ = evaluate_disorder_sample(b_base; samples=samples, seed=500_000 + 1000 * si)
        avg_peaks = epd * K + 1.0

        sp = plot(
            1:K, mean_b,
            ribbon=std_b,
            fillalpha=0.30,
            lw=2.2,
            color=:crimson,
            xlabel=si > 3 ? "Site Index i" : "",
            ylabel=(si == 1 || si == 4) ? "⟨b_i⟩ ± σ_i" : "",
            title=@sprintf("μ = %.2f | q̄ = %.3f | peaks = %.1f", mu, q_bar, avg_peaks),
            titlefontsize=10,
            ylims=(-0.02, 1.02),
            legend=false,
            gridalpha=0.3,
        )
        hline!(sp, [0.5], color=:black, ls=:dash, lw=1.2, alpha=0.6)
        push!(subplots, sp)
    end

    p_prof = plot(
        subplots...,
        layout=(2, 3),
        size=(1400, 760),
        margin=4Plots.mm,
        plot_title="Quenched Replica-Averaged Profiles ⟨b_i⟩ ± σ_i (N = $N, M = $samples)",
        plot_titlefontsize=13,
    )

    outpath = joinpath(figures_dir, "spatial_profiles_mu_N$(N).pdf")
    savefig(p_prof, outpath)
    display(p_prof)
    println("Saved spatial profiles to ", outpath)
end

run_mu_fss(N_list=[50, 100, 250, 500, 1000], samples=300, disorder_realizations=6)
plot_spatial_profiles(N=500, samples=1000)