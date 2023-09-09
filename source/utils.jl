using Plots, LaTeXStrings, Printf


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

    # Usual MSE metrics.
    function metrics(params)
        solution = Array(solve(problem, Tsit5(), p=params, saveat=T))
        loss = sum(sum(([solution[dim,index] for index in T_obs_indexes[dim]] - X_obs[dim]).^2) for dim=1:2)
        return loss
    end

    return loss_adjoint, metrics
end


function make_problem(rhs, initial_params, t_end = 15.0)
    x_0, history = get_initial_conditions()
    t_span = (0.0, t_end)
    problem = ODEProblem{true}(rhs, x_0, t_span)

    return problem
end



function aic(X_obs, params, objective_value)
    n_obs = sum(length(x) for x in X_obs)
    n_p = length(params) + 1
    log(objective_value) * n_obs + 2.0 * n_p + 2.0 * n_p * (n_p + 1.0) / (n_obs - n_p - 1.0)
end


function draw(solution, p, T_obs, V_obs, E_obs, str_title, file_name)
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
    title!(str_title)

    savefig("output/" * file_name * ".pdf")
end

