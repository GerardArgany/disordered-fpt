# =============================================================================
#  Check of the "General Mean Field" first and second FPT moments
#  Chain on sites 1..N-1, reflecting at 1, absorbing at N, rates b (right),
#  d (left), stay = 1-b-d  (so b + d <= 1 is required for a valid chain).
#  Run:  julia mfpt_meanfield_check.jl      Needs: Plots
# =============================================================================
ENV["GKSwstype"] = "nul"            # no display window (headless / no Qt)
using Random, Statistics, LinearAlgebra, Plots, Printf
Random.seed!(1234)
default(legend = :topleft, framestyle = :box,
        left_margin = 12Plots.mm, bottom_margin = 10Plots.mm,
        top_margin = 4Plots.mm, right_margin = 4Plots.mm)

const OUTDIR = joinpath(".", "Gerard", "Figures", "mf_moments")
mkpath(OUTDIR)
save(p, name) = (savefig(p, joinpath(OUTDIR, name)); println("saved ", joinpath(OUTDIR, name)))
pos(v) = v > 0 ? v : NaN            # hide non-positive values on log axes

# ----------------------------------------------------------------------------
# 1. EXACT moments for arbitrary b_i, d_i (tridiagonal linear systems)
#    b_i T_{i+1} + d_i T_{i-1} - (b_i+d_i) T_i = rhs_i ,  T_N = 0,
#    site 1 reflecting (no d_1 term, i.e. tau_0 = 0)
# ----------------------------------------------------------------------------
function solve_chain(b::Vector{Float64}, d::Vector{Float64}, rhs::Vector{Float64})
    n  = length(b)
    dm = [-(b[i] + d[i]) for i in 1:n];  dm[1] = -b[1]
    dl = [d[i] for i in 2:n]
    du = [b[i] for i in 1:n-1]
    return Tridiagonal(dl, dm, du) \ rhs
end

function exact_moments(b::Vector{Float64}, d::Vector{Float64})
    n = length(b)
    M = solve_chain(b, d, -ones(n))              # <T_i>
    S = solve_chain(b, d, -(2 .* M .- 1))        # <T_i^2>
    return M[1], S[1]
end
exact_moments_mf(b, d, N) = exact_moments(fill(float(b), N-1), fill(float(d), N-1))

# ----------------------------------------------------------------------------
# 2. Analytic mean-field formulas  (x = d/b)
# ----------------------------------------------------------------------------
function T1_full(b, d, N)
    x = d / b
    isapprox(x, 1; atol = 1e-12) && return N * (N - 1) / (2b)
    return (N - (1 - x^N) / (1 - x)) / (b - d)
end
T1_large(b, d, N) = (N - b / (b - d)) / (b - d)

function T2_full(b, d, N)
    x  = d / b
    T1 = T1_full(b, d, N)
    xN = x^N
    A  = N * (N - 1) / 2
    B  = 2 * (x - N * xN + (N - 1) * xN * x) / (1 - x)^2
    C  = xN / (1 - x) * (N - 1 - (x - xN) / (1 - x))
    return 2 / (b - d)^2 * (A - B + C) - T1
end
# large N exactly as in the notes (4 N^2 d b factor)
T2_large_asWritten(b, d, N) = N^2 / (b - d)^2 - 4 * N^2 * d * b / (d - b)^4 - T1_large(b, d, N)
# 4N^2 -> 4
T2_large_fixed(b, d, N)     = N^2 / (b - d)^2 - 4 * d * b / (b - d)^4 - T1_large(b, d, N)
# 4N^2 -> 4 and keep N(N-1) instead of N^2  (needed for the variance)
T2_large_keepN(b, d, N)     = N * (N - 1) / (b - d)^2 - 4 * d * b / (b - d)^4 - T1_large(b, d, N)

# ----------------------------------------------------------------------------
# 3. Monte-Carlo simulation of the first passage time
# ----------------------------------------------------------------------------
function simulate_fpt(b::Vector{Float64}, d::Vector{Float64}, N::Int, nsamples::Int)
    all(b .+ d .<= 1 + 1e-12) || error("invalid chain: b + d > 1")
    times = Vector{Float64}(undef, nsamples)
    for s in 1:nsamples
        pos = 1; t = 0
        while pos < N
            t += 1
            r = rand()
            if r < b[pos]
                pos += 1
            elseif r < b[pos] + d[pos]
                pos = max(pos - 1, 1)
            end
        end
        times[s] = t
    end
    return times
end
simulate_mf(b, d, N, ns) = simulate_fpt(fill(float(b), N-1), fill(float(d), N-1), N, ns)

# =============================================================================
# FIGURE 1: N, b fixed, scan d
# =============================================================================
let N = 100, b = 0.5, ns = 20_000
    ds  = collect(range(0.0, 0.45, length = 10))     # b + d <= 1 always
    dsf = collect(range(0.0, 0.46, length = 200))

    mc1 = Float64[]; mc1e = Float64[]; mc2 = Float64[]; mc2e = Float64[]
    ex1 = Float64[]; ex2 = Float64[]
    for d in ds
        t = simulate_mf(b, d, N, ns)
        push!(mc1, mean(t));      push!(mc1e, std(t) / sqrt(ns))
        push!(mc2, mean(t .^ 2)); push!(mc2e, std(t .^ 2) / sqrt(ns))
        m, s = exact_moments_mf(b, d, N)
        push!(ex1, m); push!(ex2, s)
    end

    p1 = plot(dsf, [T1_full(b, d, N) for d in dsf], lw = 2, label = "T1 full formula",
              xlabel = "d", ylabel = "<T1>", yscale = :log10, title = "First moment (N=$N, b=$b)")
    plot!(p1, dsf, [T1_large(b, d, N) for d in dsf], lw = 2, ls = :dash, label = "T1 large-N")
    scatter!(p1, ds, ex1, marker = :x, ms = 7, label = "exact linear solve")
    scatter!(p1, ds, mc1, yerror = mc1e, marker = :circle, ms = 4, label = "Monte Carlo")

    p2 = plot(dsf, [pos(T2_full(b, d, N)) for d in dsf], lw = 2, label = "T2 full formula",
              xlabel = "d", ylabel = "<T1^2>", yscale = :log10, title = "Second moment")
    plot!(p2, dsf, [pos(T2_large_asWritten(b, d, N)) for d in dsf], lw = 2, ls = :dash, label = "large-N (as written)")
    plot!(p2, dsf, [pos(T2_large_fixed(b, d, N)) for d in dsf], lw = 2, ls = :dashdot, label = "large-N (4N^2 -> 4)")
    plot!(p2, dsf, [pos(T2_large_keepN(b, d, N)) for d in dsf], lw = 2, ls = :dot, label = "large-N (4N^2->4, N(N-1))")
    scatter!(p2, ds, ex2, marker = :x, ms = 7, label = "exact linear solve")
    scatter!(p2, ds, mc2, yerror = mc2e, marker = :circle, ms = 4, label = "Monte Carlo")
    ylims!(p2, (0.5 * minimum(ex2), 2 * maximum(ex2)))

    save(plot(p1, p2, layout = (1, 2), size = (1400, 520)), "fig1_scan_d.png")

    println("\nFIG 1 numbers (N=$N, b=$b)")
    @printf("%6s %12s %12s %12s %12s %14s %14s %14s\n",
            "d", "T1 MC", "T1 exact", "T1 full", "T1 large", "T2 MC", "T2 exact", "T2 full")
    for (k, d) in enumerate(ds)
        @printf("%6.3f %12.3f %12.3f %12.3f %12.3f %14.1f %14.1f %14.1f\n",
                d, mc1[k], ex1[k], T1_full(b, d, N), T1_large(b, d, N),
                mc2[k], ex2[k], T2_full(b, d, N))
    end
end

# =============================================================================
# FIGURE 2: fixed (b,d), scan N
# =============================================================================
let b = 0.6, d = 0.3
    Ns  = collect(10:10:400)
    exv = [exact_moments_mf(b, d, N) for N in Ns]
    e1  = first.(exv);  e2 = last.(exv);  evar = e2 .- e1 .^ 2
    var_of(f1, f2) = [f2(b, d, N) - f1(b, d, N)^2 for N in Ns]

    p1 = plot(Ns, e2, lw = 3, label = "exact", xlabel = "N", ylabel = "<T1^2>", yscale = :log10,
              title = "Second moment (b=$b, d=$d)")
    plot!(p1, Ns, [pos(T2_full(b, d, N)) for N in Ns], lw = 2, ls = :dash, label = "full")
    plot!(p1, Ns, [pos(T2_large_asWritten(b, d, N)) for N in Ns], lw = 2, ls = :dot, label = "large-N as written (negative -> hidden)")
    plot!(p1, Ns, [pos(T2_large_fixed(b, d, N)) for N in Ns], lw = 2, ls = :dashdot, label = "large-N (4N^2->4)")
    ylims!(p1, (0.5 * minimum(e2), 2 * maximum(e2)))

    p2 = plot(Ns, evar, lw = 3, label = "exact", xlabel = "N", ylabel = "Var(T1)",
              title = "Variance")
    plot!(p2, Ns, var_of(T1_full, T2_full), lw = 2, ls = :dash, label = "full")
    plot!(p2, Ns, var_of(T1_large, T2_large_fixed), lw = 2, ls = :dashdot, label = "large-N, N^2 (drops N/2)")
    plot!(p2, Ns, var_of(T1_large, T2_large_keepN), lw = 2, ls = :dot, label = "large-N, N(N-1)")
    plot!(p2, Ns, [N * ((b + d) - (b - d)^2) / (b - d)^3 for N in Ns], lw = 1, c = :black,
          label = "asymptotic N s2/v^3")

    save(plot(p1, p2, layout = (1, 2), size = (1400, 520)), "fig2_scan_N.png")
    println("\nFIG 2: max rel. error |T2_full - exact|/exact = ",
            maximum(abs.(T2_full.(b, d, Ns) .- e2) ./ e2))
end

# =============================================================================
# FIGURE 3: inversion d(b,T1) and what a T1-matched mean field predicts
# =============================================================================
function d_from_T1(b, target, N)
    lo, hi = 0.0, b * (1 - 1e-9)
    (target < T1_full(b, lo, N) || target > T1_full(b, hi, N)) && return NaN
    for _ in 1:200
        mid = (lo + hi) / 2
        T1_full(b, mid, N) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

let N = 200, b = 0.7
    T1s = collect(range(N / b * 1.02, N / b * 6, length = 120))
    dex = [d_from_T1(b, T, N) for T in T1s]
    p1 = plot(T1s, dex, lw = 3, label = "exact inversion", xlabel = "T1", ylabel = "d",
              title = "d(T1), b=$b, N=$N")
    plot!(p1, T1s, b .- N ./ T1s, lw = 2, ls = :dash, label = "d = b - N/T1")
    plot!(p1, T1s, (1 + 1 / N) * b .- N ./ T1s, lw = 2, ls = :dot, label = "d = (1+1/N) b - N/T1")

    # heterogeneous valid chains (b_i + d_i <= 1): random order and b-ascending order
    a2r = Float64[]; m2r = Float64[]; avr = Float64[]; mvr = Float64[]
    a2s = Float64[]; m2s = Float64[]; avs = Float64[]; mvs = Float64[]
    function mf_from(bi, di)
        T1a, T2a = exact_moments(bi, di)
        bmf = mean(bi); dmf = d_from_T1(bmf, T1a, N)
        isnan(dmf) && return nothing
        return T2a, T2_full(bmf, dmf, N), T1a
    end
    for _ in 1:200
        b0 = 0.45 + 0.3 * rand(); s = rand()
        bi = clamp.(b0 .+ s * 0.4 .* (rand(N - 1) .- 0.5), 0.05, 0.95)
        di = min.(0.5 * rand() .* rand(N - 1), 0.9 .* bi, 1 .- bi)
        r = mf_from(bi, di)
        if r !== nothing
            T2a, T2m, T1a = r
            push!(a2r, T2a); push!(m2r, T2m); push!(avr, T2a - T1a^2); push!(mvr, T2m - T1a^2)
        end
        perm = sortperm(bi)                                # ascending b_i
        r = mf_from(bi[perm], di[perm])
        if r !== nothing
            T2a, T2m, T1a = r
            push!(a2s, T2a); push!(m2s, T2m); push!(avs, T2a - T1a^2); push!(mvs, T2m - T1a^2)
        end
    end

    p2 = scatter(a2r, m2r, ms = 3, label = "random order", xscale = :log10, yscale = :log10,
                 xlabel = "actual <T1^2> (exact)", ylabel = "<T1^2> from MF",
                 title = "T2 prediction (T1 matched)")
    scatter!(p2, a2s, m2s, ms = 3, label = "b ascending")
    lims = extrema(vcat(a2r, m2r, a2s, m2s)); plot!(p2, [lims...], [lims...], c = :black, label = "y = x")

    p3 = scatter(avr, mvr, ms = 3, label = "random order", xscale = :log10, yscale = :log10,
                 xlabel = "actual Var(T1) (exact)", ylabel = "Var from MF",
                 title = "Variance prediction (sharper test)")
    scatter!(p3, avs, mvs, ms = 3, label = "b ascending")
    vv = filter(>(0), vcat(avr, mvr, avs, mvs)); lv = extrema(vv)
    plot!(p3, [lv...], [lv...], c = :black, label = "y = x")

    save(plot(p1, p2, p3, layout = (1, 3), size = (1800, 520)), "fig3_inversion.png")
    println("\nFIG 3: median |T2_MF/T2 - 1|   random = ", median(abs.(m2r ./ a2r .- 1)),
            "   ascending = ", median(abs.(m2s ./ a2s .- 1)))
    println("       median |Var_MF/Var - 1|  random = ", median(abs.(mvr ./ avr .- 1)),
            "   ascending = ", median(abs.(mvs ./ avs .- 1)))
end