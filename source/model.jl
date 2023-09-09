using DiffEqFlux, Optimization, OptimizationFlux, DifferentialEquations, LinearAlgebra
using DataDrivenSparse, DataDrivenDiffEq, ModelingToolkit
using Printf

include("./utils.jl")


function make_model()
    n_physically_informed_params = 5

    # Neural network architecture.
    nn_output_dim = 4
    rhs_nn = Chain(Dense(2, 10, tanh), Dense(10, 10, tanh), Dense(10, nn_output_dim)) |> f64
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

    return rhs, initial_params, rhs_nn, nn_output_dim, n_physically_informed_params
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
    res1 = Optimization.solve(optprob, Adam(0.005), callback = callback, maxiters = 2000)

    println("Decreasing learning rate")

    optprob2 = Optimization.OptimizationProblem(optf, res1.u)
    res2 = Optimization.solve(optprob2, Adam(0.001), callback = callback, maxiters = 4000)

    println("Decreasing learning rate")

    optprob3 = Optimization.OptimizationProblem(optf, res2.u)
    res3 = Optimization.solve(optprob3, Adam(0.0001), callback = callback, maxiters = 10000)

    println("Finished")

    return res3, metrics(res3.u)
end


function log_sparsify_function(func, use_V=true, n_samples=10000)
    X = 18 * rand(2, n_samples)
    Y = reduce(hcat, map(func, eachcol(X)))

    problem = DirectDataDrivenProblem(X, Y, name = :Test)

    if use_V
        @variables log_V, log_E
        basis = Basis(polynomial_basis([log_V, log_E], 1), [log_V, log_E])
    else
        @variables log_V, log_E
        basis = Basis(polynomial_basis([log_E], 1), [log_V, log_E])
    end

    result = solve(problem, basis, ADMM())

    return result
end

function sparsify_function(func, n_samples=10000)
    X = 18 * rand(2, n_samples)
    Y = reduce(hcat, map(func, eachcol(X)))
    X = exp.(X)

    problem = DirectDataDrivenProblem(X, Y, name = :Test)

    @variables V, E
    basis = Basis(polynomial_basis([V, E], 1), [V, E])

    result = solve(problem, basis, ADMM())

    return result
end



function main()
    # Acquire observations (true data).
    T_obs, X_obs = get_observations()

    # Assemble model.
    rhs, initial_params, rhs_nn, nn_output_dim, n_physically_informed_params = make_model()
    problem = make_problem(rhs, initial_params)

    # Optimize.
    optimization_result, metrics_value = fit_model(problem, initial_params, T_obs, X_obs)
    optimal_params = optimization_result.u
    objective_value = optimization_result.objective

    println("Metrics: ", metrics_value)
    println("Objective: ", objective_value)
    aic_value = aic(X_obs, optimal_params, metrics_value)


    # Draw solution.
    str_title = "Нейромодель" *
            ", Φ(p̂) = " * Printf.format(Printf.Format("%3.2e"), objective_value) *
            ", AIC = " * Printf.format(Printf.Format("%4.1f"), aic_value)

    solution = solve(problem, Tsit5(), p=optimal_params)
    draw(solution, optimal_params, T_obs, X_obs[1], X_obs[2], str_title, "neural")

    # Data-driven sparse regression.
    for dim=1:nn_output_dim
        func = x -> rhs_nn(log.(max.(x, 1e-16)), optimal_params[n_physically_informed_params + 1:end])[dim]

        sparsify_result = log_sparsify_function(func)

        println(get_basis(sparsify_result))
        println(get_parameter_map(get_basis(sparsify_result)))
        println(rss(sparsify_result))
    end

    println("Inverse approximation")
    println("---------------------")
    println("")

    for dim=1:nn_output_dim
        func = x -> 1.0 / rhs_nn(log.(max.(x, 1e-16)), optimal_params[n_physically_informed_params + 1:end])[dim]

        sparsify_result = log_sparsify_function(func)

        println(get_basis(sparsify_result))
        println(get_parameter_map(get_basis(sparsify_result)))
        println(rss(sparsify_result))
    end
end
