function calculate_T1_user(b, d)
    N_minus_1 = length(b)
    T1 = 0.0
    for m in 1:N_minus_1
        inner_sum = 0.0
        for i in 1:m
            prod_term = 1.0 / b[i]
            for j in (i+1):m
                prod_term *= (d[j] / b[j])
            end
            inner_sum += prod_term
        end
        T1 += inner_sum
    end
    return T1
end

function calculate_T0_user(b, d)
    N_minus_1 = length(b)
    T0 = 1.0 
    for m in 1:N_minus_1
        inner_sum = 0.0
        prod_term_0 = 1.0
        for j in 1:m
            prod_term_0 *= (d[j] / b[j])
        end
        inner_sum += prod_term_0
        for i in 1:m
            prod_term = 1.0 / b[i]
            for j in (i+1):m
                prod_term *= (d[j] / b[j])
            end
            inner_sum += prod_term
        end
        T0 += inner_sum
    end
    return T0
end

function calculate_paper_ET(b, d)
    len = length(b)
    rho = d ./ b
    sum_contiguous = 0.0
    for m in 1:len
        for i in 1:(len - m + 1)
            prod_term = 1.0
            for j in i:(i + m - 1)
                prod_term *= rho[j]
            end
            sum_contiguous += prod_term
        end
    end
    return (len + 1) + 2 * sum_contiguous
end

# Define b probabilities, then strictly enforce d = 1 - b
b_asc = [0.8, 0.7, 0.6, 0.5, 0.4]
d_asc = 1.0 .- b_asc

# Reverse for the descending arrays
b_desc = reverse(b_asc)
d_desc = reverse(d_asc)

# 1. Original User Formula (Starts at 1)
T1_asc = calculate_T1_user(b_asc, d_asc)
T1_desc = calculate_T1_user(b_desc, d_desc)

# 2. Adjusted User Formula (Starts at 0)
T0_asc = calculate_T0_user(b_asc, d_asc)
T0_desc = calculate_T0_user(b_desc, d_desc)

# 3. Paper Formula (Starts at 0)
ET_asc = calculate_paper_ET(b_asc, d_asc)
ET_desc = calculate_paper_ET(b_desc, d_desc)

println("--- User Formula T1 (Starts at 1) ---")
println("Ascending:  ", round(T1_asc, digits=4))
println("Descending: ", round(T1_desc, digits=4))
println("Symmetric?  ", isapprox(T1_asc, T1_desc))

println("\n--- Adjusted User Formula T0 (Starts at 0) ---")
println("Ascending:  ", round(T0_asc, digits=4))
println("Descending: ", round(T0_desc, digits=4))
println("Symmetric?  ", isapprox(T0_asc, T0_desc))

println("\n--- Paper Formula ET (Starts at 0) ---")
println("Ascending:  ", round(ET_asc, digits=4))
println("Descending: ", round(ET_desc, digits=4))
println("Symmetric?  ", isapprox(ET_asc, ET_desc))