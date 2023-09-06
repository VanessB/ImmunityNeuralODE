using DiffEqFlux, Optimization, OptimizationFlux, DifferentialEquations, LinearAlgebra, Plots, LaTeXStrings, Printf


function get_observations()
    T_obs = [[1.0:4.0;6:2:14],
             [2.5;6:2:14]]

    V_obs = ([3.55e4, 5e5, 3.8e6, 3.2e6, 3.7e4, 3.1e4, 2.5e2, 2e2, 10] .+
             [1.2e4, 1.6e6, 3.9e6, 2.1e6, 1.25e5, 2.6e4, 8e4, 7.5e2, 10]) ./2

    E_obs = ([5e3, 8.33e5, 4.75e6, 4.16e6, 3.07e6, 2.22e6] .+
             [5e3, 9.85e5, 4.03e6, 5.8e6, 2.25e6, 2.89e6]) ./2

    X_obs = [V_obs, E_obs]

    return T_obs, X_obs
end


function get_initial_conditions()
    # Initial conditions.
    V_0 = 200.0
    E_0 = 256.0
    x_0 = [V_0, E_0]

    # History
    history(p, t) = [0.0, E_0]

    return x_0, history
end


function make_model()
    n_physically_informed_params = 5

    rhs_nn = Chain(Dense(2, 50, tanh), Dense(50, 50, tanh), Dense(50, 2)) |> f64
    p_nn, rhs_nn = Flux.destructure(rhs_nn)
    n_nn_params = length(p_nn)

    function rhs(dot_x, x, params, t)
        beta, K, alpha_E, b_i, gamma = exp.(params[begin:n_physically_informed_params])
        V, E = x

        rhs_nn_output = 0.01 * rhs_nn(x, params[n_physically_informed_params + 1:end])
        dot_x[1] = beta * V * (1.0 - V / K) - gamma * (1.0 + rhs_nn_output[1]) * V * E
        dot_x[2] = b_i * (1.0 + rhs_nn_output[2]) * V * E - alpha_E * E

        return dot_x
    end

    initial_params = zeros(n_physically_informed_params + n_nn_params)
    initial_params[begin:n_physically_informed_params] .= log.([4.5, 2.7e6, 9.3e-2, 9.22e-7, 1.4e-6])
    initial_params[n_physically_informed_params + 1:end] = p_nn

    return rhs, initial_params
end


function make_problem(t_end = 15.0)
    x_0, history = get_initial_conditions()
    t_span = (0.0, t_end)
    rhs, initial_params = make_model()
    problem = ODEProblem{true}(rhs, x_0, t_span)

    return problem, initial_params
end


function make_neural_ode_loss(problem, T_obs, X_obs)
    T = sort(unique(cat(T_obs..., dims=1)))
    T_obs_indexes = [something.(indexin(x, T)) for x in T_obs]

    function loss_adjoint(params)
        solution = Array(solve(problem, Tsit5(), p=params, saveat=T,
                               sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))

        loss = sum(sum(([log(solution[dim,index]) for index in T_obs_indexes[dim]] - log.(X_obs[dim])).^2) for dim=1:size(X_obs)[1])
        return loss
    end

    return loss_adjoint
end


function find_optimal()
    T_obs, X_obs = get_observations()
    problem, initial_params = make_problem()

    loss_function = make_neural_ode_loss(problem, T_obs, X_obs)

    iter = 0
    function callback(params, loss)
        iter += 1
        #if iter % 10 == 0
        #    println(loss)
        #end
        println(loss)
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x,p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, initial_params)
    res1 = Optimization.solve(optprob, Adam(0.05), callback = callback, maxiters = 150)

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Adam(0.001), callback = callback, maxiters = 150)
    opt = res2.u

    solution = solve(problem, Tsit5(), p=opt)
    draw(solution, opt, T_obs, X_obs[1], X_obs[2], "Нейромодель")
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



function _test_model()
    x_0, history = get_initial_conditions()
    rhs, initial_params = make_model()
    display(rhs(zero(x_0), x_0, initial_params, 10.0))
end
