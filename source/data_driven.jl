using DiffEqFlux, Optimization, OptimizationFlux, DifferentialEquations, LinearAlgebra
using DataDrivenSparse, DataDrivenDiffEq, ModelingToolkit
using Printf

include("./utils.jl")


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
    # Right-hand-side function for differential equation.
    function rhs(dot_x, x, params, t)
        # Get params and current values.
        beta, K, alpha_E, b_i, gamma = exp.(params[begin:5])
        p11, p12, p13, p14, p15, p21, p22, p23, p31, p32, p33, p34, p35, p41, p42, p43, p44, p45, p46 = params[6:end]

        V, E = x
        log_V, log_E = log(max(1e-16, V)), log(max(1e-16, E))

        # Get NN output (+ scale and center around 1.0).
        phi_1 = 1.0 + 1e-2 * (p11 + log_V * p12 + log_E * p14 + p15 * (log_E)^2 + p13 * log_V^2)
        phi_2 = 1.0 + 1e-2 * (p22 + log_E * p22 + log_E * log_V * p23)
        phi_3 = 1.0 + 1e-2 * (p31 + log_V * p32 + log_E * p34 + p35 * (log_E^2) + p33 * (log_V^2))
        phi_4 = 1.0 + 1e-2 * (p41 + log_V * p42 + log_E * p44 + p46 * (log_E^2) + p43 * (log_V^2) + log_E * log_V * p45)

        # Compute derivative.
        dot_x[1] = beta * V * (1.0 - V / (K * phi_1)) - gamma * V * E * phi_2
        dot_x[2] = b_i * V * E * phi_3 - alpha_E * E * phi_4

        return dot_x
    end

    # Assemble initial parameters vector.
    # Physically informed parameters are stored in log scale.
    initial_params = cat(log.([4.5, 2.7e6, 9.3e-2, 9.22e-7, 1.4e-6]),
                         [-1.0468679161, 2.0887833243, -0.2859564972, -0.1957789809, -0.1060100014,
                          -0.7773401834, -0.5569085076, 0.1345174182,
                          -0.8776003553, 3.1877898572, -0.7624685992, 0.4856839188, -0.4401322301,
                          3.1956055466, 0.7615889755, -0.1970445311, -4.8779260399, 0.2778054162, 0.9146740718], dims=1)

    return rhs, initial_params
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
        if iter % 100 == 0
            println(loss)
        end
        #println(loss)
        return false
    end

    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x,p) -> loss_function(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, initial_params)
    res1 = Optimization.solve(optprob, Adam(0.01), callback = callback, maxiters = 2000)

    println("Decreasing learning rate")

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Adam(0.001), callback = callback, maxiters = 5000)

    println("Finished")

    return res2
end


function main()
    # Acquire observations (true data).
    T_obs, X_obs = get_observations()

    # Assemble model.
    rhs, initial_params = make_model()
    problem = make_problem(rhs, initial_params)

    # Optimize.
    optimization_result = fit_model(problem, initial_params, T_obs, X_obs)
    optimal_params = optimization_result.u
    objective_value = optimization_result.objective
    aic_value = aic(X_obs, optimal_params, objective_value)

    println(objective_value)

    # Draw solution.
    str_title = "Дистилл. модель" *
            ", Φ(p̂) = " * Printf.format(Printf.Format("%3.2e"), objective_value) *
            ", AIC = " * Printf.format(Printf.Format("%4.1f"), aic_value)

    solution = solve(problem, Tsit5(), p=optimal_params)
    draw(solution, optimal_params, T_obs, X_obs[1], X_obs[2], str_title, "data_driven")
end



function _test_model()
    x_0, history = get_initial_conditions()
    rhs, initial_params = make_model()
    display(rhs(zero(x_0), x_0, initial_params, 10.0))
end
