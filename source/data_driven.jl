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
        p11, p12, p13, p31, p32, p33 = params[6:end]

        V, E = x
        log_V, log_E = log(max(1e-16, V)), log(max(1e-16, E))

        # Get NN output (+ scale and center around 1.0).
        phi_1 = 1.0 + 1e-1 * (p11 + p12 * log_V + p13 * log_E)
        phi_3 = 1.0 + 1e-1 * (p31 + p32 * log_V + p33 * log_E)

        # Compute derivative.
        dot_x[1] = beta * V * (1.0 - V / (K * phi_1)) - gamma * V * E
        dot_x[2] = b_i * V * E * phi_3 - alpha_E * E

        return dot_x
    end

    # Assemble initial parameters vector.
    # Physically informed parameters are stored in log scale.
    initial_params = cat(log.([4.5, 2.7e6, 9.3e-2, 9.22e-7, 1.4e-6]),
                         [0.5079776122, 0.1961962255, -0.2007616565,
                          2.9358171279, 0.1563130119, -0.2638260516], dims=1)

    return rhs, initial_params
end


function fit_model(problem, initial_params, T_obs, X_obs)
    loss_function, metrics = make_neural_ode_loss(problem, T_obs, X_obs)

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
    res1 = Optimization.solve(optprob, Adam(0.01), callback = callback, maxiters = 500)

    println("Decreasing learning rate")

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Adam(0.001), callback = callback, maxiters = 3500)

    println("Finished")

    return res2, metrics(res2.u)
end


function main()
    # Acquire observations (true data).
    T_obs, X_obs = get_observations()

    # Assemble model.
    rhs, initial_params = make_model()
    problem = make_problem(rhs, initial_params)

    # Optimize.
    optimization_result, metrics_value = fit_model(problem, initial_params, T_obs, X_obs)
    optimal_params = optimization_result.u
    objective_value = optimization_result.objective
    aic_value = aic(X_obs, optimal_params, metrics_value)

    println("Metrics: ", metrics_value)
    println("Objective: ", objective_value)

    # Draw solution.
    str_title = "Дистилл. модель" *
            ", Φ(p̂) = " * Printf.format(Printf.Format("%.1f"), objective_value) *
            ", AIC = " * Printf.format(Printf.Format("%.1f"), aic_value)

    solution = solve(problem, Tsit5(), p=optimal_params)
    draw(solution, optimal_params, T_obs, X_obs[1], X_obs[2], str_title, "data_driven")
end
