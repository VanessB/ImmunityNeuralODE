using DifferentialEquations, Plots, LaTeXStrings, Printf


function model_rhs(xdot, x, h, p, t)
    beta, K, gamma, b_i, theta_sat, tau, alpha_E = p
    V, E = x
    Vh, Eh = h(p, t - tau)

    xdot[1] = beta * V * (1.0 - V / K) - gamma * V * E
    xdot[2] = b_i * Vh * Eh / (theta_sat + V) - alpha_E * E
end


function make_problem()
    # Parameters.
    # beta, K, gamma, b_i, theta_sat, tau, alpha_E
    p = [4.52, 3.17e6, 4.45e-6, 2.52, 1.34e5, 7.16e-2, 8.62e-2]

    # Initial conditions.
    V_0 = 200.0
    E_0 = 256.0
    x_0 = [V_0, E_0]

    # History
    history(p, t) = [0.0, E_0]

    tspan = (0.0, 10.0)
    model_problem = DDEProblem(model_rhs, x_0, history, tspan, p, constant_lags = [p[6]])

    return model_problem
end


function make_objective(problem, T_obs, X_obs)
    function objective(p)
        problem_p = remake(problem, p=p)
        solution = solve(problem_p, MethodOfSteps(Tsit5()), reltol=1e-15, abstol=1e-15)
        return sum((solution(T_obs[1], idxs=1) - X_obs[1]).^2) + sum((solution(T_obs[2], idxs=2) - X_obs[2]).^2)
    end

    return objective
end


function draw(solution, p, T_obs, V_obs, E_obs, str_title)
    plot(T_obs[1], V_obs, lt=:scatter, marker=:rect, color=:lightgreen,
         label = L"V_{obs}")
    plot!(T_obs[2], E_obs, lt=:scatter, marker=:rect, color=:magenta,
          label = L"E_{obs}")

    tt = 0:0.01:14
    plot!(solution(tt, idxs=1), color=:green, lw=2, label=L"V(t)")
    plot!(solution(tt, idxs=2), color=:darkmagenta, lw=2, label=L"E(t)",
          legend=:bottomleft)

    yaxis!(yscale=:log10, ylims=(1,1e8))
    xaxis!(xticks=0:14)
    xlabel!("t, сутки")
    ylabel!("копии на селезенку")
    title!(str_title)# * ", Φ(p̂) = " * @sprintf("%3.2e", obj(p))) #* ", AICc = " * @sprintf("%4.1f", aic(p)))

    savefig(str_title * ".pdf")
end


function find_optimum()
    T_obs = [[1.0:4.0;6:2:14],
             [2.5;6:2:14]]

    V_obs = ([3.55e4, 5e5, 3.8e6, 3.2e6, 3.7e4, 3.1e4, 2.5e2, 2e2, 10] .+
             [1.2e4, 1.6e6, 3.9e6, 2.1e6, 1.25e5, 2.6e4, 8e4, 7.5e2, 10]) ./2

    E_obs = ([5e3, 8.33e5, 4.75e6, 4.16e6, 3.07e6, 2.22e6] .+
             [5e3, 9.85e5, 4.03e6, 5.8e6, 2.25e6, 2.89e6]) ./2

    X_obs = [V_obs, E_obs]

    problem = make_problem()

    objective = make_objective(problem, T_obs, X_obs)

    # Random search.
    n_steps = 100
    n_guesses = 20
    spread = 0.2
    #best_guess = [4.52, 3.17e6, 4.45e-6, 2.52, 1.34e5, 7.16e-2, 8.62e-2]
    best_guess = [6.0, 5.0e6, 5.0e-6, 5.0, 2.0e5, 8.0e-2, 8.0e-2]
    best_val = objective(best_guess)
    for step in 1:n_steps
        noise = exp.(spread * randn(n_guesses, size(best_guess)...))
        guesses = best_guess' .* noise

        objectives = mapslices(objective, guesses, dims=2)
        min_val, min_index = findmin(objectives)
        if min_val < best_val
            best_guess = guesses[min_index[1],:]
            best_val = min_val
        end
        println("Текущая наименьшая ошибка: ")
        println(best_val)
    end

    println("Оптимум: ")
    display(best_guess)

    problem_p = remake(problem, p=best_guess)
    solution = solve(problem_p, MethodOfSteps(Tsit5()), reltol=1e-15, abstol=1e-15)
    draw(solution, best_guess, T_obs, V_obs, E_obs, "plot")
end

#=
plot(solution, linewidth = 5, title = "Model 2",
    xaxis = "Time (t)", yaxis = "x(t)", yscale = :log10,
    titlefontsize=28,
    guidefontsize=28,
    tickfontsize=26,
    legendfontsize=22,
    size = (1920, 1080), fmt = :png)
savefig("plot.png")
=#
