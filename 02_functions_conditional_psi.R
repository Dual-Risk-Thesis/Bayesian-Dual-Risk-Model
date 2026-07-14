# ============================================================
# 02_functions_conditional_psi.R.R
# Minimal helper functions for Bayesian single-path calibration
# in the classical continuous-time dual risk model
# ============================================================
#
# This file is intentionally a clean toolbox:
#   - no read.csv()
#   - no write.csv()
#   - no saveRDS()
#   - no ggsave()
#   - no set.seed()
#   - no automatic analysis
#
# The goal is to keep only the model-specific logic here and let 03_run_results_conditional_psi.R handle the actual analysis.
# ============================================================


# ============================================================
# 1. MODEL-SPECIFIC MATHEMATICAL FUNCTIONS
# ============================================================

net_profit_exp <- function(lambda, beta, c_rate) {
  # Dual risk net profit condition for exponential gains: lambda * E[X] > c, and E[X] = 1 / beta.
  lambda / beta > c_rate
}

lundberg_root_exp <- function(lambda, beta, c_rate) {
  # Raw Lundberg root for exponential gains: R_raw = lambda / c - beta.
  lambda / c_rate - beta
}

ruin_prob_exp <- function(lambda, beta, c_rate, u) {
  # Ultimate ruin probability for exponential gains.
  #
  # For each posterior draw, there are two cases:
  #   1. Net profit condition holds:
  #        lambda * E[X] > c, with E[X] = 1 / beta.
  #        Then R = lambda / c - beta > 0 and psi(u) = exp(-R*u).
  #   2. Net profit condition fails:
  #        Ultimate ruin is certain, so psi(u) = 1.
  #
  # We keep two versions of psi:
  #   - psi: unconditional posterior draw, including psi = 1 when net profit fails.
  #   - psi_given_net_profit_holds: conditional draw, set to NA when net profit fails.
  #
  # The second column is useful when reporting credible intervals for the continuous
  # part of the posterior distribution, without mixing in the point mass at psi = 1.

  R_raw <- lundberg_root_exp(lambda, beta, c_rate)
  net_profit_holds <- net_profit_exp(lambda, beta, c_rate)

  psi_given_net_profit_holds <- ifelse(
    net_profit_holds & R_raw > 0,
    exp(-R_raw * u),
    NA_real_
  )

  psi <- ifelse(
    net_profit_holds & R_raw > 0,
    psi_given_net_profit_holds,
    1
  )

  # Numerical safety: probabilities should stay inside [0, 1].
  psi <- pmin(pmax(psi, 0), 1)
  psi_given_net_profit_holds <- pmin(pmax(psi_given_net_profit_holds, 0), 1)

  data.frame(
    R_raw = R_raw,
    net_profit_holds = net_profit_holds,
    psi = psi,
    psi_given_net_profit_holds = psi_given_net_profit_holds
  )
}


# ============================================================
# 2. DATA EXTRACTION
# ============================================================

extract_sufficient_stats <- function(events, summary, gain_col = "gain_size") {
  # Sufficient statistics for the conjugate model:
  #   lambda | data uses n and full observation horizon T.
  #   beta | data uses n and sum of observed gains.
  #
  # Important: T is the full observation horizon, not the time of the last gain.

  T_obs <- if ("T_for_lambda_likelihood" %in% names(summary)) {
    summary$T_for_lambda_likelihood[1]
  } else {
    summary$observation_horizon[1]
  }

  n_events <- nrow(events)
  gains <- if (n_events > 0) events[[gain_col]] else numeric(0)
  sum_x <- if (n_events > 0) sum(gains) else 0

  data.frame(
    Case = if ("Case" %in% names(summary)) summary$Case[1] else NA_character_,
    n = n_events,
    T = T_obs,
    sum_x = sum_x,
    lambda_mle = n_events / T_obs,
    beta_mle = ifelse(n_events > 0 && sum_x > 0, n_events / sum_x, NA_real_),
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 3. PRIORS AND POSTERIORS
# ============================================================

gamma_prior_from_mean_concentration <- function(mean, concentration) {
  # R uses the parameters: theta ~ Gamma(shape, rate).
  # This parametrisation gives:
  #   E[theta] = shape / rate = mean
  # when:
  #   shape = concentration
  #   rate = concentration / mean
  #
  # Simply saying, mean decides the center and 
  # concentration decides how hard the prior is holding at the center
  list(
    shape = concentration,
    rate = concentration / mean
  )
}

make_prior_specs <- function(lambda_mean,
                             lambda_concentration,
                             beta_mean,
                             beta_concentration,
                             prior_name = "main") {
  lambda_prior <- gamma_prior_from_mean_concentration(lambda_mean, lambda_concentration) # lambda ~ Gamma(shape, rate)
  beta_prior <- gamma_prior_from_mean_concentration(beta_mean, beta_concentration) # beta ~ Gamma(shape, rate)

  list(
    prior_name = prior_name,
    lambda_shape = lambda_prior$shape,
    lambda_rate = lambda_prior$rate,
    beta_shape = beta_prior$shape,
    beta_rate = beta_prior$rate,
    lambda_prior_mean = lambda_mean,
    beta_prior_mean = beta_mean,
    lambda_prior_concentration = lambda_concentration,
    beta_prior_concentration = beta_concentration
  )
}

 # Function (prior_specs_to_table) will be used mainly to store prior-specifications in a table, rather than a list
prior_specs_to_table <- function(prior_specs, case_name = NA_character_) {
  data.frame(
    Case = case_name,
    prior_name = prior_specs$prior_name,
    parameter = c("lambda", "beta"),
    shape = c(prior_specs$lambda_shape, prior_specs$beta_shape),
    rate = c(prior_specs$lambda_rate, prior_specs$beta_rate),
    mean = c(
      prior_specs$lambda_shape / prior_specs$lambda_rate,
      prior_specs$beta_shape / prior_specs$beta_rate
    ),
    sd = c( # Formula for sd from Gamma-distribution
      sqrt(prior_specs$lambda_shape) / prior_specs$lambda_rate, 
      sqrt(prior_specs$beta_shape) / prior_specs$beta_rate
    ),
    concentration = c(
      prior_specs$lambda_prior_concentration,
      prior_specs$beta_prior_concentration
    ),
    stringsAsFactors = FALSE
  )
}

# posterior_params_gamma is used for Bayesian update.
# Argument stats contains sufficient statistics from data.
posterior_params_gamma <- function(stats, prior_specs) {
  # Conjugate posterior parameters:
  #   lambda | data ~ Gamma(a_lambda + n, b_lambda + T)
  #   beta | data ~ Gamma(a_beta + n, b_beta + sum_x)
  data.frame(
    Case = stats$Case[1],
    prior_name = prior_specs$prior_name,
    lambda_shape_post = prior_specs$lambda_shape + stats$n[1],
    lambda_rate_post = prior_specs$lambda_rate + stats$T[1],
    beta_shape_post = prior_specs$beta_shape + stats$n[1],
    beta_rate_post = prior_specs$beta_rate + stats$sum_x[1],
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 4. POSTERIOR DRAWS
# ============================================================

# The function (make_draws_df) takes the posterior Gamma parameters and simulates S posterior draws of \lambda and \beta.
make_draws_df <- function(posterior_params, S, case_name = NULL) {
  case_name <- if (is.null(case_name)) posterior_params$Case[1] else case_name

  data.frame(
    Case = case_name,
    prior_name = posterior_params$prior_name[1],
    draw_id = seq_len(S),
    lambda = rgamma(
      n = S,
      shape = posterior_params$lambda_shape_post[1],
      rate = posterior_params$lambda_rate_post[1]
    ),
    beta = rgamma(
      n = S,
      shape = posterior_params$beta_shape_post[1],
      rate = posterior_params$beta_rate_post[1]
    ),
    stringsAsFactors = FALSE
  )
}

# The function (add_ruin_quantities) takes the posterior draws of \lambda and \beta, then computes ruin-related quantities for every draw.
add_ruin_quantities <- function(draws, u, c_rate) {
  ruin_quantities <- ruin_prob_exp(
    lambda = draws$lambda,
    beta = draws$beta,
    c_rate = c_rate,
    u = u
  )

  cbind(draws, ruin_quantities)
}

# The function (run_case_analysis) runs the whole analysis for one case.
run_case_analysis <- function(events,
                              summary,
                              prior_specs,
                              u,
                              c_rate,
                              S,
                              case_name = NULL) {
  stats <- extract_sufficient_stats(events, summary) # This step turns raw data into the sufficient statistics needed for the conjugate Bayesian update
  if (!is.null(case_name)) stats$Case <- case_name

  posterior_params <- posterior_params_gamma(stats, prior_specs) # Bayesian update
  draws <- make_draws_df(posterior_params, S, case_name = stats$Case[1])
  draws <- add_ruin_quantities(draws, u, c_rate)

  diagnostics <- data.frame(
    Case = stats$Case[1],
    prior_name = prior_specs$prior_name,
    S = S,
    n = stats$n[1],
    T = stats$T[1],
    sum_x = stats$sum_x[1],
    lambda_mle = stats$lambda_mle[1],
    beta_mle = stats$beta_mle[1],
    posterior_pr_net_profit_holds = mean(draws$net_profit_holds), # Posterior probability that NPC holds
    posterior_pr_net_profit_fails = mean(!draws$net_profit_holds), # Since net_profit_holds is TRUE/FALSE -> mean() can be used to calculate the proportion of TRUE values
    posterior_pr_psi_equals_one = mean(draws$psi == 1), # The proportion of posterior draws that has ultimate ruin probability = 100%
    stringsAsFactors = FALSE
  )

  # Return all intermediate and final objects.
  list(
    stats = stats,
    prior_specs = prior_specs,
    posterior_params = posterior_params,
    draws = draws,
    diagnostics = diagnostics
  )
}


# ============================================================
# 5. OPTIONAL PLOT HELPERS
# ============================================================

# The function (plot_density) plots a posterior density for some variable in the posterior draws table
plot_density <- function(draws, variable, case_col = "Case", title = NULL, x_label = NULL) {
  if (is.null(title)) title <- paste("Posterior density:", variable)
  if (is.null(x_label)) x_label <- variable

  # .data[[variable]] = Inside this data frame, take the column whose name is stored in the object variable.
  # Example: variable <- "psi" =>  .data[[variable]] = .data[["psi]] => draws$psi
  ggplot2::ggplot(draws, ggplot2::aes(x = .data[[variable]], linetype = .data[[case_col]])) +
    # This part adds a kernel density estimate, where missing values are ignored
    ggplot2::geom_density(linewidth = 0.8, na.rm = TRUE) + 
    ggplot2::labs(
      title = title,
      x = x_label,
      y = "Density",
      linetype = "Case"
    ) +
    ggplot2::theme_minimal()
}

# This function plots a normalized likelihood for either \lambda or \beta, using the sufficient statistics.
# Note that this is not the posterior. It is the likelihood shape, normalized so it can be plotted like a density.
# stats = sufficient-statistics table from extract_sufficient_stats() 
# grid_length = number of x-values used to draw the curve
plot_likelihood_gamma <- function(stats, parameter = c("lambda", "beta"), grid_length = 500) {
  parameter <- match.arg(parameter) # Stop the function if parameter is wrongfully written.

### Why the normalized likelihood has a Gamma shape:
# For lambda, the Poisson likelihood is L(lambda) ∝ lambda^n * exp(-lambda*T).
# This has the same mathematical form as a Gamma density, since Gamma(shape, rate) ∝ x^(shape - 1) * exp(-rate*x).
# Therefore, after normalizing the likelihood so that it integrates to 1, it can be plotted as Gamma(n + 1, T).
#
# For beta, the exponential gain-size likelihood is L(beta) ∝ beta^n * exp(-beta*sum_x).
# This also has the same Gamma form, so the normalized likelihood can be plotted as Gamma(n + 1, sum_x).
# This is only a normalized likelihood used for visualization. It is not the posterior distribution, because it does not include the prior.
  if (parameter == "lambda") {
    shape <- stats$n[1] + 1
    rate <- stats$T[1]
    x_label <- "lambda"
  } else {
    shape <- stats$n[1] + 1
    rate <- stats$sum_x[1]
    x_label <- "beta"
  }
  
  
  # Handle invalid likelihoods
  # For \lambda, the rate is stats$T[1] and for \beta, the rate is stats$sum_x[1].
  # These two should not be <= 0 or is infinite endpoint to create evenly spaced grid points.
  if (!is.finite(rate) || rate <= 0) { 
    return(
      ggplot2::ggplot() +
        ggplot2::annotate("text", x = 0, y = 0, label = "No normalized likelihood available") +
        ggplot2::theme_void()
    )
  }

  # Create the x-grid. We have x-values from 0 up to the 99.9% quantile of the normalized Gamma likelihood.
  # We cannot use Inf as the upper plotting limit because seq() needs a finite
  x_grid <- seq(0, stats::qgamma(0.999, shape = shape, rate = rate), length.out = grid_length)
  
  # Create the plotting data
  plot_data <- data.frame(x = x_grid, density = stats::dgamma(x_grid, shape = shape, rate = rate))

  # Plotting the normalized likelihood
  ggplot2::ggplot(plot_data, ggplot2::aes(x = x, y = density)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::labs(
      title = paste("Normalized likelihood for", parameter),
      x = x_label,
      y = "Normalized likelihood"
    ) +
    ggplot2::theme_minimal()
}

# ============================================================
# End of 02_functions.R
# ============================================================
