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
    W_0 = 0.0
    x_0 = [V_0, E_0]

    # History
    history(p, t) = [0.0, E_0]

    return x_0, history
end


function make_model()
    n_physically_informed_params = 5

    # Neural network architecture.
    rhs_nn = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, 4)) |> f64
    p_nn, rhs_nn = Flux.destructure(rhs_nn)
    n_nn_params = length(p_nn)

    # Right-hand-side function for differential equation.
    function rhs(dot_x, x, params, t)
        # Get params and current values.
        beta, K, alpha_E, b_i, gamma = exp.(params[begin:n_physically_informed_params])
        V, E = x

        # Get NN output (+ scale and center around 1.0).
        rhs_nn_output = 1.0 .+ 1e-1 * rhs_nn(log.(max.(x, 1e-16)), params[n_physically_informed_params + 1:end])

        # Compute derivative.
        dot_x[1] = beta * V * (1.0 - V / (K * rhs_nn_output[1])) - gamma * V * E * rhs_nn_output[2]
        dot_x[2] = b_i * V * E * rhs_nn_output[3] - alpha_E * E * rhs_nn_output[4]

        return dot_x
    end

    # Assemble initial parameters vector.
    # Physically informed parameters are stored in log scale.
    initial_params = zeros(n_physically_informed_params + n_nn_params)
    initial_params[begin:n_physically_informed_params] .= log.([4.5, 2.7e6, 9.3e-2, 9.22e-7, 1.4e-6])
    initial_params[n_physically_informed_params + 1:end] = p_nn

    return rhs, initial_params, rhs_nn
end


function make_problem(rhs, initial_params, t_end = 15.0)
    x_0, history = get_initial_conditions()
    t_span = (0.0, t_end)
    problem = ODEProblem{true}(rhs, x_0, t_span)

    return problem
end


function make_neural_ode_loss(problem, T_obs, X_obs)
    # Unite time grids.
    T = sort(unique(cat(T_obs..., dims=1)))
    T_obs_indexes = [something.(indexin(x, T)) for x in T_obs]

    # Loss function (MSE in log scale).
    function loss_adjoint(params)
        solution = Array(solve(problem, Tsit5(), p=params, saveat=T,
                               sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))

        loss = sum(sum(([log(max.(solution[dim,index], 1e-16)) for index in T_obs_indexes[dim]] - log.(X_obs[dim])).^2) for dim=1:2)
        return loss
    end

    return loss_adjoint
end


function fit_model(problem, initial_params, T_obs, X_obs)
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
    res1 = Optimization.solve(optprob, Adam(0.01), callback = callback, maxiters = 1000)

    println("Decreasing learning rate")

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Adam(0.001), callback = callback, maxiters = 4000)

    println("Finished")

    return res2
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

    savefig("output/" * str_title * ".pdf")
end


function main()
    # Acquire observations (true data).
    T_obs, X_obs = get_observations()

    # Assemble model.
    rhs, initial_params, rhs_nn = make_model()
    problem = make_problem(rhs, initial_params)

    # Optimize.
    optimization_result = fit_model(problem, initial_params, T_obs, X_obs)
    optimal_params = optimization_result.u

    # Draw solution.
    solution = solve(problem, Tsit5(), p=optimal_params)
    draw(solution, opt, T_obs, X_obs[1], X_obs[2], "Нейромодель")
end



function _test_model()
    x_0, history = get_initial_conditions()
    rhs, initial_params = make_model()
    display(rhs(zero(x_0), x_0, initial_params, 10.0))
end
