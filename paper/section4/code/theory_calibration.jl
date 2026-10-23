#!/usr/bin/env julia
# theory_calibration.jl -- calibration objects for Section 4 (A Neoclassical Interpretation).
# Canonical generator; code/theory_calibration.py is a line-for-line mirror (code/check_mirror.sh).
#
# Reads   <INPUTS>/tables/pooled_did_results_table.csv   columns: row_key, coef_table, ...
#         rows used: international, etr, foreign_profit_share
#         <INPUTS>/tables/section4_same_sample.csv        key,value (optional): robustness
#             estimates from build_section4_same_sample.py (x_pre <= 0 restriction,
#             same-sample re-estimates, Table 9 auxiliaries), passed through as macros
#         <INPUTS>/tables/section4_bootstrap_draws.csv    (optional): firm-bootstrap draws of
#             the three betas under the baseline convention (beta_Intl/beta_ETR on the full
#             resample, beta_share on its share subset), pushed through the inversion for
#             percentile intervals
# Writes  <INPUTS>/tables/theory_calibration.csv          key,value  (all scalars used by the text)
#         <INPUTS>/tables/theory_grid.csv                 one row per (tau_h, case) for Table A.2;
#             cases: none, statutory, and (at the text's tau_h) fdii-none / fdii-statutory,
#             where re-booked rent comes home at the 13.125 percent FDII rate
#
# Model (Section 4.3):
#     beta_intl  = -[(1-theta_o)(1-psi) + (1-theta_h^2) psi]
#     beta_share =  (1-theta_h) psi / omega_h
# Statutory inputs at low-tax rate tau_h:
#     omega_h        = 0.35 - tau_h
#     theta_h (none) = (0.21 - tau_h) / omega_h                                GILTI does not reach the rent
#     theta_h (stat) = (0.21 - [tau_h + max(0, 0.105 - 0.8 tau_h)]) / omega_h  statutory residual
# Inversion (closed form):
#     psi     = beta_share * omega_h / (1 - theta_h)
#     theta_o = 1 - [|beta_intl| - (1 - theta_h^2) psi] / (1 - psi)
#     tau_f   = (0.21 - 0.35 theta_o) / (1 - theta_o)     operating rate implied by theta_o
# Worked example: a firm with exposure -10 points, at each end of the theta_h range.
# Usage:  julia theory_calibration.jl [INPUTS_DIR]     (default: Inputs).  No package dependencies.

using Printf

const INPUTS = length(ARGS) >= 1 ? ARGS[1] : "Inputs"

const TAU_D_PRE, TAU_D_POST = 0.35, 0.21
const TAU_H      = 0.05            # low-tax rate for the text's range
const GILTI_RATE = 0.105           # statutory GILTI rate through 2025
const CREDIT     = 0.80            # creditable share of foreign taxes
const TAU_H_GRID = (0.00, 0.05, 0.10)
const EXPOSURE   = 10.0            # worked example: |x_intl_pre| in percentage points
# Hall-Jorgenson check (footnote in "The real coefficients")
const Z_PARTIAL, ETR_LO, ETR_HI, KA, HORIZON, DELTA = 0.8, 0.18, 0.24, 0.5, 5.0, 0.10

function readtable(path::AbstractString; keycol::AbstractString = "row_key")
    lines = filter(l -> !isempty(strip(l)), readlines(path))
    header = String.(strip.(split(lines[1], ',')))
    out = Dict{String,Dict{String,String}}()
    for l in lines[2:end]
        cells = String.(strip.(split(l, ',')))
        row = Dict{String,String}(h => (j <= length(cells) ? cells[j] : "") for (j, h) in enumerate(header))
        out[row[keycol]] = row
    end
    return out
end
num(s::AbstractString) = parse(Float64, replace(s, "\"" => ""))

const FDII_RATE = 0.13125           # domestic rate on foreign-derived intangible income

function theta_h(tau_h, case::String)
    allin = endswith(case, "none") ? tau_h : tau_h + max(0.0, GILTI_RATE - CREDIT * tau_h)
    home  = startswith(case, "fdii") ? FDII_RATE : TAU_D_POST
    return (home - allin) / (TAU_D_PRE - tau_h)
end

function invert(beta_share, beta_intl_abs, omega_h, th)
    psi     = beta_share * omega_h / (1 - th)
    theta_o = 1 - (beta_intl_abs - (1 - th^2) * psi) / (1 - psi)
    tau_f   = (TAU_D_POST - TAU_D_PRE * theta_o) / (1 - theta_o)
    return psi, theta_o, tau_f
end

function example!(out, prefix, beta_share, beta_intl_abs, beta_etr_abs, omega_h, th)
    psi, theta_o, tau_f = invert(beta_share, beta_intl_abs, omega_h, th)
    omega_o = TAU_D_PRE - tau_f
    rent_part, op_part = psi * EXPOSURE, (1 - psi) * EXPOSURE
    s, so = rent_part / omega_h, op_part / omega_o
    sigma = s + so
    omega_o_post, omega_h_post = theta_o * omega_o, th * omega_h
    s_post = th * s
    item_op_pre, item_op_post     = -omega_o * so, -omega_o_post * so
    item_rent_pre, item_rent_post = -omega_h * s, -omega_h_post * s_post
    item_pre, item_post = item_op_pre + item_rent_pre, item_op_post + item_rent_post
    etr_pre, etr_post   = 100 * TAU_D_PRE + item_pre, 100 * TAU_D_POST + item_post
    append!(out, [
        prefix * "Psi" => psi, prefix * "ThetaO" => theta_o, prefix * "TauF" => tau_f, prefix * "TauFPct" => 100 * tau_f,
        prefix * "OmegaOPre" => omega_o, prefix * "OmegaOPost" => omega_o_post,
        prefix * "OmegaHPre" => omega_h, prefix * "OmegaHPost" => omega_h_post,
        prefix * "RentPart" => rent_part, prefix * "OpPart" => op_part,
        prefix * "S" => s, prefix * "SPost" => s_post, prefix * "SChg" => s_post - s,
        prefix * "SigmaO" => so, prefix * "Sigma" => sigma, prefix * "SigmaPost" => sigma - (s - s_post),
        prefix * "SigmaChg" => -(s - s_post), prefix * "SigmaFallPct" => 100 * (s - s_post) / sigma,
        prefix * "Rebooked" => s - s_post, prefix * "RebookedPct" => 100 * (1 - th),
        prefix * "ItemOpPre" => item_op_pre, prefix * "ItemOpPost" => item_op_post, prefix * "ItemOpChg" => item_op_post - item_op_pre,
        prefix * "ItemRentPre" => item_rent_pre, prefix * "ItemRentPost" => item_rent_post, prefix * "ItemRentChg" => item_rent_post - item_rent_pre,
        prefix * "ItemPre" => item_pre, prefix * "ItemPost" => item_post, prefix * "ItemChg" => item_post - item_pre,
        prefix * "EtrPre" => etr_pre, prefix * "EtrPost" => etr_post, prefix * "EtrChg" => etr_post - etr_pre,
        prefix * "EtrRelDomestic" => (etr_post - etr_pre) - 100 * (TAU_D_POST - TAU_D_PRE),
        prefix * "BetaShareX" => beta_share * EXPOSURE, prefix * "BetaIntlX" => beta_intl_abs * EXPOSURE,
        prefix * "BetaEtrX" => beta_etr_abs * EXPOSURE,
    ])
end

function main()
    did = readtable(joinpath(INPUTS, "tables", "pooled_did_results_table.csv"))
    beta_intl  = abs(num(did["international"]["coef_table"]))
    beta_etr   = abs(num(did["etr"]["coef_table"]))
    beta_share = num(did["foreign_profit_share"]["coef_table"])
    ss_path = joinpath(INPUTS, "tables", "section4_same_sample.csv")
    ss = isfile(ss_path) ? readtable(ss_path; keycol = "key") : Dict{String,Dict{String,String}}()
    # Baseline convention: all three coefficients on the foreign-profit-share
    # sample, so the inversion is not mixing projections on different
    # populations.  Falls back to the Table 7 rows if the file is absent.
    beta_intl_full, beta_etr_full = beta_intl, beta_etr
    if haskey(ss, "betaIntlSame")
        beta_intl = abs(num(ss["betaIntlSame"]["value"]))
        beta_etr  = abs(num(ss["betaEtrSame"]["value"]))
    end

    omega_h  = TAU_D_PRE - TAU_H
    residual = max(0.0, GILTI_RATE - CREDIT * TAU_H)
    out = Pair{String,Float64}["betaIntlAbs" => beta_intl, "betaEtrAbs" => beta_etr, "betaShare" => beta_share,
        "betaIntlFullAbs" => beta_intl_full, "betaEtrFullAbs" => beta_etr_full,
        "passthroughEtrBasePct" => 100 * beta_etr / beta_intl,

        "tauH" => TAU_H, "tauHPct" => 100 * TAU_H, "omegaHStat" => omega_h, "omegaHStatPts" => 100 * omega_h, "exposure" => EXPOSURE,
        "giltiRate" => GILTI_RATE, "giltiRatePct" => 100 * GILTI_RATE, "creditShare" => CREDIT, "creditPct" => 100 * CREDIT,
        "giltiThresholdPct" => 100 * GILTI_RATE / CREDIT, "giltiResidual" => residual, "giltiResidualPts" => 100 * residual,
        "rentAllInStatPct" => 100 * (TAU_H + residual), "rentAllInNoGiltiPct" => 100 * TAU_H,
        "omegaHPostStat" => TAU_D_POST - TAU_H - residual, "omegaHPostNoGilti" => TAU_D_POST - TAU_H]
    for k in ("betaIntlSame", "betaIntlSameSe", "betaEtrSame", "betaEtrSameSe",
              "betaIntlSameN", "betaIntlSameG", "betaLogPiSame", "betaLogPiSameSe",
              "betaIntlNeg", "betaIntlNegSe", "betaEtrNeg", "betaEtrNegSe",
              "betaShareNeg", "betaShareNegSe", "betaIntlNegSame", "betaIntlNegSameSe",
              "negDropFirmsSamePct", "negDropTreatVarSamePct",
              "negDropFirmsPct", "negDropObsPct", "negDropTreatVarPct",
              "betaPpeShare", "betaPpeShareSe", "betaPpeShareN", "betaPpeShareG",
              "betaLogRd")
        haskey(ss, k) && push!(out, k => num(ss[k]["value"]))
    end
    th_stat, th_none = theta_h(TAU_H, "statutory"), theta_h(TAU_H, "none")
    for (label, th) in (("Stat", th_stat), ("NoGilti", th_none))
        psi, theta_o, tau_f = invert(beta_share, beta_intl, omega_h, th)
        append!(out, ["thetaH" * label => th, "psiInv" * label => psi, "psiInv" * label * "Pct" => 100 * psi,
                      "thetaOInv" * label => theta_o, "opWedgeFall" * label * "Pct" => 100 * (1 - theta_o),
                      "tauFImplied" * label => tau_f, "tauFImplied" * label * "Pct" => 100 * tau_f,
                      "omegaOPre" * label => TAU_D_PRE - tau_f, "omegaOPost" * label => theta_o * (TAU_D_PRE - tau_f),
                      "rentRebooked" * label * "Pct" => 100 * (1 - th),
                      "rateCutShareOfRentWedgeFall" * label * "Pct" => 100 * (TAU_D_PRE - TAU_D_POST) / (omega_h - th * omega_h)])
    end

    # robustness grid
    grid = NamedTuple[]; P = Float64[]; T = Float64[]; F = Float64[]
    for tau_h in TAU_H_GRID, case in ("none", "statutory")
        th = theta_h(tau_h, case)
        wh = TAU_D_PRE - tau_h
        psi, theta_o, tau_f = invert(beta_share, beta_intl, wh, th)
        # Shifted share of foreign profits, S/(S+pi_f), per unit of |x_pre|:
        # rent share s = psi/omega_h, operating share sigma_o = (1-psi)/omega_o.
        s, so = psi / wh, (1 - psi) / (TAU_D_PRE - tau_f)
        sh_pre, sh_post = s / (s + so), th * s / (th * s + so)
        tau_h_post = endswith(case, "none") ? tau_h : tau_h + max(0.0, GILTI_RATE - CREDIT * tau_h)
        push!(grid, (tau_h = tau_h, gilti_case = case, omega_h = wh, theta_h = th,
                     tau_h_post = tau_h_post, psi = psi, theta_o = theta_o, tau_f = tau_f,
                     shifted_pre = sh_pre, shifted_post = sh_post, rebooked = 1 - th))
        push!(P, psi); push!(T, theta_o); push!(F, tau_f)
        # flat per-cell keys for the transposed table (Table 10 of draft 3)
        # digit-free suffix: LaTeX control sequences cannot contain digits
        tag = (endswith(case, "none") ? "NoGilti" : "Stat") *
              Dict(0 => "Zero", 5 => "Five", 10 => "Ten")[Int(round(100 * tau_h))]
        append!(out, ["gTauHPost" * tag => 100 * tau_h_post, "gTauF" * tag => 100 * tau_f,
                      "gPsi" * tag => 100 * psi, "gShiftedPre" * tag => 100 * sh_pre,
                      "gShiftedPost" * tag => 100 * sh_post])
    end
    append!(out, ["psiInvLo" => minimum(P), "psiInvHi" => maximum(P), "thetaOInvLo" => minimum(T),
                  "thetaOInvHi" => maximum(T), "tauFLoPct" => 100 * minimum(F), "tauFHiPct" => 100 * maximum(F)])
    for case in ("fdii-none", "fdii-statutory")   # FDII scenarios at the text's tau_h (below the grid range)
        th = theta_h(TAU_H, case)
        psi, theta_o, tau_f = invert(beta_share, beta_intl, omega_h, th)
        s, so = psi / omega_h, (1 - psi) / (TAU_D_PRE - tau_f)
        push!(grid, (tau_h = TAU_H, gilti_case = case, omega_h = omega_h, theta_h = th,
                     tau_h_post = FDII_RATE, psi = psi, theta_o = theta_o, tau_f = tau_f,
                     shifted_pre = s / (s + so), shifted_post = th * s / (th * s + so),
                     rebooked = 1 - th))
        key = case == "fdii-none" ? "FdiiNoGilti" : "FdiiStat"
        append!(out, ["psiInv" * key => psi, "psiInv" * key * "Pct" => 100 * psi,
                      "thetaH" * key => th, "tauFImplied" * key * "Pct" => 100 * tau_f])
    end

    # worked examples at the two ends of the range
    example!(out, "ex",   beta_share, beta_intl, beta_etr, omega_h, th_stat)
    example!(out, "exNg", beta_share, beta_intl, beta_etr, omega_h, th_none)

    # bootstrap intervals: push each draw through the inversion at both theta_h ends
    bs_path = joinpath(INPUTS, "tables", "section4_bootstrap_draws.csv")
    if isfile(bs_path)
        lines = filter(l -> !isempty(strip(l)), readlines(bs_path))
        hdr = String.(strip.(split(lines[1], ',')))
        ib, sb = findfirst(==("beta_intl"), hdr), findfirst(==("beta_share"), hdr)
        for (label, th) in (("Stat", th_stat), ("NoGilti", th_none))
            Pb = Float64[]; Tb = Float64[]; Fb = Float64[]
            for l in lines[2:end]
                c = split(l, ',')
                psi, theta_o, tau_f = invert(num(String(c[sb])), abs(num(String(c[ib]))), omega_h, th)
                push!(Pb, psi); push!(Tb, theta_o); push!(Fb, tau_f)
            end
            q(v, p) = sort(v)[clamp(round(Int, p * length(v)), 1, length(v))]
            append!(out, ["psiCiLo" * label => q(Pb, 0.025), "psiCiHi" * label => q(Pb, 0.975),
                          "psiCiLo" * label * "Pct" => 100 * q(Pb, 0.025), "psiCiHi" * label * "Pct" => 100 * q(Pb, 0.975),
                          "thetaOCiLo" * label => q(Tb, 0.025), "thetaOCiHi" * label => q(Tb, 0.975),
                          "tauFCiLo" * label * "Pct" => 100 * q(Fb, 0.025), "tauFCiHi" * label * "Pct" => 100 * q(Fb, 0.975)])
        end
    end

    # Hall-Jorgenson check
    hj(tau, z) = (1 - tau * z) / (1 - tau)
    rise = 100 * (hj(ETR_HI, Z_PARTIAL) / hj(ETR_LO, Z_PARTIAL) - 1)
    append!(out, ["hjUserCostRisePct" => rise, "capexImpliedTransitionPp" => rise * KA / HORIZON,
                  "capexImpliedSteadyPp" => DELTA * rise * KA])

    mkpath(joinpath(INPUTS, "tables"))
    open(joinpath(INPUTS, "tables", "theory_calibration.csv"), "w") do io
        println(io, "key,value")
        for (k, v) in out
            @printf(io, "%s,%.6f\n", k, v)
        end
    end
    open(joinpath(INPUTS, "tables", "theory_grid.csv"), "w") do io
        println(io, "tau_h,gilti_case,omega_h,theta_h,tau_h_post,psi,theta_o,tau_f,shifted_pre,shifted_post,rebooked")
        for g in grid
            @printf(io, "%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    g.tau_h, g.gilti_case, g.omega_h, g.theta_h, g.tau_h_post, g.psi,
                    g.theta_o, g.tau_f, g.shifted_pre, g.shifted_post, g.rebooked)
        end
    end
    println("wrote theory_calibration.csv (", length(out), " keys) and theory_grid.csv (", length(grid), " rows)")
end

main()
