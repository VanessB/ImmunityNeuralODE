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


function aic(X_obs, params, objective_value)
    n_obs = length(X_obs)
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

