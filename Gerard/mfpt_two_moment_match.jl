# =============================================================================
#  Find one (b,d) reproducing BOTH <T1> and Var(T1) of a chain
#  Leading order:  v = b-d = N/T1 ,  s = b+d = (N/T1)^2 (1 + Var/T1)
#  Feasible iff   T1 (T1/N - 1) <= Var <= T1 (T1^2/N^2 - 1)   (0<=d, b+d<=1)
# =============================================================================
ENV["GKSwstype"] = "nul"
using Random, Statistics, LinearAlgebra, Plots, Printf
Random.seed!(7)
default(legend = :topleft, framestyle = :box, left_margin = 12Plots.mm,
        bottom_margin = 10Plots.mm, top_margin = 4Plots.mm, right_margin = 4Plots.mm)
const OUTDIR = joinpath(".", "Gerard", "Figures", "mf_moments"); mkpath(OUTDIR)
save(p, name) = (savefig(p, joinpath(OUTDIR, name)); println("saved ", joinpath(OUTDIR, name)))

# ---- exact moments of an arbitrary chain (reflecting at 1, absorbing at N) ---
function solve_chain(b, d, rhs)
    n = length(b)
    dm = [-(b[i] + d[i]) for i in 1:n]; dm[1] = -b[1]
    Tridiagonal([d[i] for i in 2:n], dm, [b[i] for i in 1:n-1]) \ rhs
end
function exact_moments(b::Vector{Float64}, d::Vector{Float64})
    M = solve_chain(b, d, -ones(length(b)))
    S = solve_chain(b, d, -(2 .* M .- 1))
    return M[1], S[1]
end

# ---- full mean-field formulas -------------------------------------------------
function T1_full(b, d, N)
    x = d / b; (1 - x^N) / (1 - x) |> y -> (N - y) / (b - d)
end
function T2_full(b, d, N)
    x = d / b; xN = x^N
    A = N * (N - 1) / 2
    B = 2 * (x - N * xN + (N - 1) * xN * x) / (1 - x)^2
    C = xN / (1 - x) * (N - 1 - (x - xN) / (1 - x))
    return 2 / (b - d)^2 * (A - B + C) - T1_full(b, d, N)
end
var_full(b, d, N) = T2_full(b, d, N) - T1_full(b, d, N)^2

# ---- leading-order inversion + feasibility window ---------------------------
function bd_leading(T1, V, N)
    v = N / T1; s = v^2 * (1 + V / T1)
    return (s + v) / 2, (s - v) / 2
end
var_window(T1, N) = (T1 * (T1 / N - 1), T1 * (T1^2 / N^2 - 1))   # (min, max)

# ---- Newton refinement on the full formulas (relative residuals) ------------
function match_moments(T1, V, N; maxit = 100)
    b, d = bd_leading(T1, V, N)
    lo, hi = var_window(T1, N)
    feasible_lo = lo <= V <= hi
    valid(b, d) = 0 <= d < b <= 1 && b + d <= 1
    F(p) = [T1_full(p[1], p[2], N) / T1 - 1, var_full(p[1], p[2], N) / V - 1]
    p = [b, d]
    # if leading guess is outside the box, start from a safe interior point
    valid(p[1], p[2]) || (p = [0.6, 0.2])
    converged = false
    for _ in 1:maxit
        f = F(p); norm(f) < 1e-10 && (converged = true; break)
        J = zeros(2, 2); h = 1e-7
        for k in 1:2
            q = copy(p); q[k] += h; J[:, k] = (F(q) - f) / h
        end
        step = -(J \ f); t = 1.0
        while t > 1e-6                                   # damped, stay inside the box
            q = p + t * step
            if valid(q[1], q[2]) && norm(F(q)) < norm(f); p = q; break; end
            t /= 2
        end
        t <= 1e-6 && break
    end
    ok = converged && valid(p[1], p[2])
    return (b = p[1], d = p[2], ok = ok, window = (lo, hi), in_window = feasible_lo,
            b0 = b, d0 = d)
end

# =============================================================================
# TEST 1: round trip.  pick (b,d), get (T1,Var), recover (b,d)
# =============================================================================
println("TEST 1: round trip, N = 200")
@printf("%6s %6s | %9s %9s | %9s %9s | %9s %9s\n", "b", "d", "b lead", "d lead", "b Newton", "d Newton", "err b", "err d")
let N = 200
    for (b, d) in [(0.6, 0.2), (0.5, 0.4), (0.8, 0.1), (0.4, 0.0), (0.55, 0.35)]
        T1 = T1_full(b, d, N); V = var_full(b, d, N)
        r = match_moments(T1, V, N)
        @printf("%6.3f %6.3f | %9.4f %9.4f | %9.5f %9.5f | %9.1e %9.1e\n",
                b, d, r.b0, r.d0, r.b, r.d, r.b - b, r.d - d)
    end
end

# =============================================================================
# TEST 2: feasible region in the (T1, Var) plane, with heterogeneous chains
# =============================================================================
let N = 200
    T1s = range(N * 1.01, N * 6, length = 300)
    lo = [var_window(T, N)[1] for T in T1s]; hi = [var_window(T, N)[2] for T in T1s]
    p = plot(T1s, lo, fillrange = hi, fillalpha = 0.15, lw = 2, label = "feasible (d>=0 ... b+d<=1)",
             xlabel = "T1", ylabel = "Var(T1)", yscale = :log10,
             title = "Which (T1,Var) a single (b,d) can reproduce, N=$N")
    plot!(p, T1s, hi, lw = 2, label = "upper: no-stay (b+d=1)")

    t1r = Float64[]; vr = Float64[]; t1s = Float64[]; vs = Float64[]
    nok = [0, 0]; ntot = [0, 0]; err2 = [Float64[], Float64[]]
    for _ in 1:300
        b0 = 0.45 + 0.3 * rand(); s = rand()
        bi = clamp.(b0 .+ s * 0.4 .* (rand(N - 1) .- 0.5), 0.05, 0.95)
        di = min.(0.5 * rand() .* rand(N - 1), 0.9 .* bi, 1 .- bi)
        for (k, perm) in enumerate((1:N-1, sortperm(bi)))       # random order, b ascending
            b = bi[perm]; d = di[perm]
            T1, T2 = exact_moments(b, d); V = T2 - T1^2
            (k == 1 ? (push!(t1r, T1); push!(vr, V)) : (push!(t1s, T1); push!(vs, V)))
            ntot[k] += 1
            r = match_moments(T1, V, N)
            if r.ok
                nok[k] += 1
                push!(err2[k], abs(T2_full(r.b, r.d, N) / T2 - 1))
            end
        end
    end
    scatter!(p, t1r, vr, ms = 3, label = "random-order chains")
    scatter!(p, t1s, vs, ms = 3, label = "b-ascending chains")
    save(p, "fig4_feasible_region.png")

    println("\nTEST 2: heterogeneous chains (N=$N)")
    for (k, name) in enumerate(("random order", "b ascending"))
        @printf("  %-13s  matched (valid b,d): %d / %d   median |T2_MF/T2 - 1| = %.2e\n",
                name, nok[k], ntot[k], isempty(err2[k]) ? NaN : median(err2[k]))
    end
end
