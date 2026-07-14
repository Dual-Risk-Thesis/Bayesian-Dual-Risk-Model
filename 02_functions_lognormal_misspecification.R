# ============================================================
# 02_functions_lognormal_misspecification.R
# Helper functions for gain-size misspecification analysis
# in the classical continuous-time dual risk model
# ============================================================
#
# Model:
#   U(t) = u - c*t + sum_{i=1}^{N(t)} X_i
#
# Misspecified gain-size model:
#   X_i ~ Lognormal(mu, sigma^2)
#   Y_i = log(X_i) ~ Normal(mu, sigma^2)
#
# This file is intentionally a clean toolbox:
#   - no read.csv()
#   - no write.csv()
#   - no saveRDS()
#   - no ggsave()
#   - no set.seed()
#   - no automatic analysis
#
# The analysis is run from 05_model_misspecification.R.
# ============================================================


# ============================================================
# 1. DATA EXTRACTION FOR LOG-GAINS
# ============================================================

extract_log_gain_stats <- function(events,
                                   summary,
                                   gain_col = "gain_size") {
  # We use the same arrival-process sufficient statistics as the baseline: lambda | data uses n and the full observation horizon T.
  # For the lognormal gain-size model we additionally use: y_i = log(x_i), ybar, and s_y^2.
  # events: table with observed gain events.
  # summary: table with information about observation horizon and case.
  # gain_col: the column that contains gain sizes. Standard is "gain_size".

  # Same function from 02_functions_conditional_psi
  base_stats <- extract_sufficient_stats(
    events = events,
    summary = summary,
    gain_col = gain_col
  ) 

  n_events <- base_stats$n[1]
  gains <- if (n_events > 0) events[[gain_col]] else numeric(0) # Get the gain sizes X_i
  log_gains <- if (n_events > 0) log(gains) else numeric(0) # log(X_i)

  # Sample mean of log-gains.
  # This is also the MLE of mu under the normal model for log(X).
  ybar <- if (n_events > 0) mean(log_gains) else NA_real_ 
  
  sum_y <- if (n_events > 0) sum(log_gains) else 0 # Sum of log-gains
  
  
  # Within-sample sum of squared deviations on the log scale:
  #   sum_i (y_i - ybar)^2
  #
  # This quantity is needed for the posterior update of sigma^2 in the normal-inverse-chi-square model.
  sum_y_minus_ybar2 <- if (n_events >= 2) { 
    sum((log_gains - ybar)^2)
  } else {
    0
  }
  
  # Usual sample variance of log-gains, using denominator n - 1 
  # because the sample mean ybar has already been estimated from the same data, which uses one degree of freedom.
  # This is mainly used for reporting and for recovering the within-sample sum of squares.
  sy2 <- if (n_events >= 2) {
    sum_y_minus_ybar2 / (n_events - 1)
  } else {
    NA_real_
  }
  
  # Maximum likelihood estimate of sigma^2 for:
  #   Y_i = log(X_i) ~ Normal(mu, sigma^2)
  #
  # The MLE uses denominator n rather than n - 1. This estimator is biased downward for finite n when mu is unknown,
  # but it is consistent as n becomes large.
  sigma2_mle <- if (n_events >= 2) {
    sum_y_minus_ybar2 / n_events
  } else {
    NA_real_
  }

  # Return all statistics needed for the lognormal misspecification
  # analysis. Some baseline quantities, such as lambda_mle and beta_mle,
  # are kept for comparison and diagnostics, even though beta is not a
  # parameter in the lognormal gain-size model.
  data.frame(
    Case = base_stats$Case[1],
    n = n_events,
    T = base_stats$T[1],
    sum_x = base_stats$sum_x[1],
    sum_y = sum_y,
    ybar = ybar,
    sy2 = sy2,
    sum_y_minus_ybar2 = sum_y_minus_ybar2,
    lambda_mle = base_stats$lambda_mle[1],
    beta_mle = base_stats$beta_mle[1],
    mu_mle = ybar,
    sigma2_mle = sigma2_mle,
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 2. PRIOR SPECIFICATION
# ============================================================

make_lognormal_misspec_prior <- function(true_row,
                                         lambda_concentration = 4,
                                         kappa0 = 4,
                                         nu0 = 4) {
  # Construct prior hyperparameters for the misspecified lognormal gain-size model.
  #
  # In the simulation study, the true data-generating gain distribution is exponential. 
  # Therefore, there are no true lognormal parameters mu and sigma^2.
  #
  # To make the lognormal misspecification analysis comparable to the
  # exponential baseline, we choose a lognormal prior that matches the
  # true exponential gain distribution in terms of:
  #   1. mean gain size
  #   2. coefficient of variation
  #
  # This does not make the lognormal distribution equal to the exponential
  # distribution. It only gives the misspecified model a comparable prior
  # location before observing the data.
  
  # Prior mean for the gain-arrival intensity lambda. Since this is a simulation study, we center the prior at the true simulated lambda value.
  lambda_mean <- true_row$lambda_true[1]
  
  # The true exponential gain-size rate parameter from the simulation.
  beta_true <- true_row$beta_true[1]

  # For exponential gains, the mean gain size is E[X] = 1 / beta.
  mean_gain_true <- 1 / beta_true

  # For a lognormal random variable:
  #   E[X]  = exp(mu + sigma^2 / 2)
  #   CV^2 = exp(sigma^2) - 1
  #
  # The exponential distribution has CV = 1, so CV^2 = 1.
  # Matching CV^2 = 1 gives:
  #   exp(sigma^2) - 1 = 1
  #   sigma^2 = log(2)
  sigma20 <- log(2)
  
  # Choose mu0 so that the lognormal sampling model, evaluated at the prior central values (mu0, sigma20), implies the same mean
  # gain on the original scale as the true exponential model.
  #
  #   mean_gain_true = exp(mu0 + sigma20 / 2)
  #
  # Solving for mu0 gives:
  #
  #   mu0 = log(mean_gain_true) - sigma20 / 2
  mu0 <- log(mean_gain_true) - sigma20 / 2

  # Return all prior hyperparameters in a list.
  #
  # For lambda we use a Gamma(shape, rate) prior. With:
  #
  #   shape = lambda_concentration
  #   rate  = lambda_concentration / lambda_mean
  #
  # the prior mean becomes:
  #
  #   E[lambda] = shape / rate = lambda_mean
  #
  # For the lognormal gain-size model, mu0, kappa0, nu0 and sigma20
  # define the normal-inverse-chi-square prior used later in the
  # posterior update.
  list(
    prior_name = "moment_matched_lognormal_misspecification",

    lambda_shape = lambda_concentration,
    lambda_rate = lambda_concentration / lambda_mean,
    lambda_prior_mean = lambda_mean,
    lambda_prior_concentration = lambda_concentration,

    mu0 = mu0,
    kappa0 = kappa0,
    nu0 = nu0,
    sigma20 = sigma20,

    implied_prior_mean_gain = exp(mu0 + sigma20 / 2),
    implied_prior_cv_gain = sqrt(exp(sigma20) - 1)
  )
}

# Convert the prior specification list into a data frame.
# This is used for saving and reporting the prior assumptions in the misspecification analysis. Keeping this as a separate table
# makes the analysis more transparent and reproducible :)
lognormal_misspec_prior_to_table <- function(prior_specs,
                                             case_name = NA_character_) {
  data.frame(
    Case = case_name,
    prior_name = prior_specs$prior_name,
    lambda_prior_mean = prior_specs$lambda_prior_mean,
    lambda_prior_concentration = prior_specs$lambda_prior_concentration,
    lambda_shape = prior_specs$lambda_shape,
    lambda_rate = prior_specs$lambda_rate,
    mu0 = prior_specs$mu0,
    kappa0 = prior_specs$kappa0,
    nu0 = prior_specs$nu0,
    sigma20 = prior_specs$sigma20,
    implied_prior_mean_gain = prior_specs$implied_prior_mean_gain,
    implied_prior_cv_gain = prior_specs$implied_prior_cv_gain,
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 3. POSTERIOR HYPERPARAMETERS
# ============================================================

normal_inverse_chisq_posterior <- function(log_stats, 
                                           prior_specs) {
  # Compute posterior hyperparameters for the lognormal gain-size misspecification model.
  #
  # The gain-size model is written on the log scale:
  #   Y_i = log(X_i)
  #   Y_i | mu, sigma^2 ~ Normal(mu, sigma^2)
  #
  # The prior for the lognormal parameters is a normal-inverse-chi-square prior:
  #   sigma^2 ~ Inv-chi-square(nu0, sigma20)
  #   mu | sigma^2 ~ Normal(mu0, sigma^2 / kappa0)
  #
  # This prior is conjugate for normal data with unknown mean and variance. Therefore, the posterior has the same form:
  #   sigma^2 | y ~ Inv-chi-square(nu_n, sigma_n2)
  #   mu | sigma^2, y ~ Normal(mu_n, sigma^2 / kappa_n)

  # Extract the number of observed gains and the sample mean of the log-transformed gains.
  n <- log_stats$n[1]
  ybar <- if (n > 0) log_stats$ybar[1] else 0
  
  # Usual sample variance of log-gains. This is mainly kept for reporting and diagnostics. 
  # The posterior update below uses the within-sample sum of squares directly.
  sy2 <- if (n >= 2) log_stats$sy2[1] else NA_real_

  # Prior hyperparameters.
  kappa0 <- prior_specs$kappa0 # kappa0 controls the prior strength for mu0.
  mu0 <- prior_specs$mu0 # mu0 is the prior center for the log-scale mean mu.
  nu0 <- prior_specs$nu0 # nu0 controls the prior strength for sigma20.
  sigma20 <- prior_specs$sigma20 # sigma20 is the prior scale/center for sigma^2.

  kappa_n <- kappa0 + n # Posterior strength for mu. The data add n units of information to the prior strength kappa0.
  
  # Posterior mean for mu.
  # This is a weighted average of the prior center mu0 and the sample mean ybar, with weights kappa0 and n.
  mu_n <- (kappa0 * mu0 + n * ybar) / kappa_n
  
  # Posterior degrees of freedom for sigma^2. The data add n observations to the prior degrees of freedom nu0.
  nu_n <- nu0 + n

  # Within-sample sum of squares:
  #   sum_i (y_i - ybar)^2
  #
  # This measures the variation of the observed log-gains around
  # their own sample mean.
  within_sample_term <- if (n >= 2) {
    log_stats$sum_y_minus_ybar2[1]
  } else {
    0
  }
  
  # Prior-data conflict term: (kappa0 * n / (kappa0 + n)) * (ybar - mu0)^2
  # This term increases the posterior variance scale when the sample mean ybar is far from the prior center mu0.
  prior_data_conflict_term <- if (n > 0) {(kappa0 * n / (kappa0 + n)) * (ybar - mu0)^2} else {0}

  # Posterior scale parameter for sigma^2.
  #   1. prior variance information: nu0 * sigma20
  #   2. within-sample variation in the log-gains
  #   3. disagreement between the prior center and the data mean
  sigma_n2 <- (nu0 * sigma20 + within_sample_term + prior_data_conflict_term) / nu_n

  # The arrival intensity lambda is updated separately using the same Gamma-Poisson conjugacy as in the exponential baseline model:
  #   lambda | n, T ~ Gamma(lambda_shape + n, lambda_rate + T)
  #                         
  # The lognormal misspecification only changes the gain-size model, not the arrival process.
  data.frame(
    Case = log_stats$Case[1],
    prior_name = prior_specs$prior_name,

    lambda_shape_post = prior_specs$lambda_shape + log_stats$n[1],
    lambda_rate_post = prior_specs$lambda_rate + log_stats$T[1],

    mu0 = mu0,
    kappa0 = kappa0,
    nu0 = nu0,
    sigma20 = sigma20,

    n = n,
    ybar = if (n > 0) log_stats$ybar[1] else NA_real_,
    sy2 = sy2,

    kappa_n = kappa_n,
    mu_n = mu_n,
    nu_n = nu_n,
    sigma_n2 = sigma_n2,
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 4. POSTERIOR DRAWS
# ============================================================

# Generate posterior draws for the parameters in the lognormal misspecification model.
#
# The gain-size model is: Y_i = log(X_i) and Y_i | mu, sigma^2 ~ Normal(mu, sigma^2).
#
# From the normal-inverse-chi-square posterior, we draw:
#   sigma^2 | y ~ Inv-chi-square(nu_n, sigma_n2)
#   mu | sigma^2, y ~ Normal(mu_n, sigma^2 / kappa_n)
#
# The arrival intensity lambda is drawn separately from its Gamma posterior:
#   lambda | n, T ~ Gamma(lambda_shape_post, lambda_rate_post)
#
# Each row in the returned data frame is one posterior draw of (lambda, mu, sigma^2).
draw_normal_inverse_chisq <- function(posterior_params,
                                      S,
                                      case_name = NULL) {
  case_name <- if (is.null(case_name)) posterior_params$Case[1] else case_name

  # Draw sigma^2 from the scaled inverse-chi-square posterior.
  # If Q ~ Chi-square(nu_n), then: sigma^2 = nu_n * sigma_n2 / Q
  # has a scaled inverse-chi-square distribution with degrees of freedom nu_n and scale sigma_n2.
  sigma2_draw <- posterior_params$nu_n[1] * posterior_params$sigma_n2[1] /
    stats::rchisq(S, df = posterior_params$nu_n[1])

  # Draw mu conditional on each sigma^2 draw.
  #
  # The posterior conditional distribution is: mu | sigma^2, y ~ Normal(mu_n, sigma^2 / kappa_n)
  #
  # Since sigma2_draw is a vector, each posterior draw of mu uses its corresponding posterior draw of sigma^2.
  mu_draw <- stats::rnorm(
    S,
    mean = posterior_params$mu_n[1],
    sd = sqrt(sigma2_draw / posterior_params$kappa_n[1])
  )

  data.frame(
    Case = case_name,
    prior_name = posterior_params$prior_name[1],
    draw_id = seq_len(S),
    lambda = stats::rgamma(n = S, shape = posterior_params$lambda_shape_post[1], # Posterior draws for the Poisson gain-arrival intensity.
                           rate = posterior_params$lambda_rate_post[1]),
    
    # Posterior draws for the lognormal gain-size parameters.
    mu = mu_draw,
    sigma2 = sigma2_draw,
    sigma = sqrt(sigma2_draw), # sigma is included as sqrt(sigma^2) because it is easier to interpret than the variance on the log scale.
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 5. LOGNORMAL LAPLACE TRANSFORM AND LUNDBERG EQUATION
# ============================================================

laplace_lognormal_z <- function(R, mu, sigma, rel.tol = 1e-8) {
  # Compute the Laplace transform of the lognormal gain-size distribution: L_X(R) = E[exp(-R X)]
  # Under the lognormal model: X = exp(mu + sigma*Z), Z ~ Normal(0, 1)
  #
  # Therefore: L_X(R) = E[exp(-R exp(mu + sigma Z))] = integral exp(-R exp(mu + sigma z)) phi(z) dz,
  # where phi(z) is the standard normal density (Z ~ Normal(0, 1)).
  #
  # The lognormal Laplace transform has no simple closed-form expression, so it is evaluated numerically.

  # At R = 0, the Laplace transform is E[exp(0)] = 1. Returning this directly avoids unnecessary numerical integration.
  if (R == 0) return(1)

  # Integrand in the standard-normal representation of the lognormal Laplace transform.
  integrand <- function(z) {
    exp(-R * exp(mu + sigma * z)) * stats::dnorm(z)
  }

  # Numerically integrate over the standard-normal variable z. 
  # This is easier to explain in our thesis and often more stable than integrating directly over x with the lognormal density.
  stats::integrate(integrand,
                   lower = -Inf,
                   upper = Inf,
                   rel.tol = rel.tol)$value
  }

lundberg_function_lognormal <- function(R, lambda, mu, sigma2, c_rate) {
  # Evaluate the Lundberg equation function under lognormal gains: f(R) = lambda * (L_X(R) - 1) + c * R
  # A positive non-trivial root of f(R) = 0 is used as the Lundberg root for the posterior draw.
  # NOTE: Unlike the exponential gain-size model, the lognormal model does not give a closed-form root 
  # because L_X(R) is evaluated numerically.
  
  sigma <- sqrt(sigma2)
  L_R <- laplace_lognormal_z(R = R, mu = mu, sigma = sigma) # Laplace transform of the lognormal gain-size distribution evaluated at R.

  lambda * (L_R - 1) + c_rate * R # Lundberg equation written as a root-finding function
}

# Find the positive non-trivial Lundberg root for one posterior draw.
find_lundberg_root_lognormal <- function(lambda, mu, sigma2, c_rate,
                                         lower = 1e-12, initial_upper = 0.01,
                                         expansion_factor = 2,
                                         max_expansions = 30) {
  # R = 0 is always a trivial root. We therefore start just above zero.
  # Under the net profit condition, f(R) initially moves below zero and
  # then eventually becomes positive because c*R dominates for large R.

  implied_mean_gain <- exp(mu + sigma2 / 2)
  expected_gain_rate <- lambda * implied_mean_gain # lambda * E[X]

  # Check the net profit condition: lambda * E[X] > c
  # If this condition fails, ultimate ruin is treated as certain later in the analysis, 
  # and no positive Lundberg root is searched for.
  if (expected_gain_rate <= c_rate) {
    return(list(R_raw = NA_real_, root_status = "net_profit_fails"))
  }

  # Define the Lundberg function for this posterior draw.
  # Only R varies here, while lambda, mu, sigma2 and c_rate are fixed from the current posterior draw.
  f_eval <- function(R) {
    lundberg_function_lognormal(R = R, lambda = lambda, mu = mu, sigma2 = sigma2, c_rate = c_rate)
  }
  
  # Evaluate f(R) slightly above zero.
  # tryCatch prevents the full analysis from stopping if numerical integration fails for this posterior draw.
  f_lower <- tryCatch(f_eval(lower), error = function(e) NA_real_)

  # If the function cannot be evaluated near zero, record this as a numerical integration failure.
  if (!is.finite(f_lower)) {
    return(list(R_raw = NA_real_, root_status = "integration_failed_at_lower"))
  }

  # Choose an initial upper bound for the root search.
  # Gain size X >= 0 and a positive root R > 0. Therefore, 0 <= L_X(R) = E[exp(-R*X)] <= 1
  # Since 0 <= L_X(R) <= 1, any positive root satisfies approximately: R <= lambda / c (We can solve this from Lundberg equation)
  # so 2 * lambda / c_rate is used as a conservative initial upper bound.
  upper <- max(initial_upper, 2 * lambda / c_rate)
  f_upper <- tryCatch(f_eval(upper), error = function(e) NA_real_) # Evaluate f(R) at the upper bound.

  # uniroot() requires a sign change over the interval: f(lower) * f(upper) < 0
  # If this is not satisfied, expand the upper bound repeatedly.
  # NOTE: We expand the upper bound while we still have expansion attempts left and the current upper bound is not usable for uniroot().
  # This happens if f_upper is non-finite, or  
  # if f_lower and f_upper still have the same sign, meaning no bracketing interval has been found.
  expansions <- 0
  while (
    expansions < max_expansions &&
      (!is.finite(f_upper) || f_lower * f_upper > 0)
  ) {
    upper <- upper * expansion_factor
    f_upper <- tryCatch(f_eval(upper), error = function(e) NA_real_)
    expansions <- expansions + 1
  }

  # If no valid sign change was found, the root cannot be located with this bracketing method.
  if (!is.finite(f_upper) || f_lower * f_upper > 0) {
    return(list(R_raw = NA_real_, root_status = "no_sign_change"))
  }

  # Solve f(R) = 0 on the bracketing interval.
  root <- tryCatch(
    stats::uniroot(
      f_eval,
      interval = c(lower, upper),
      tol = 1e-10 # Numerical tolerance for the root-finding algorithm. Stop when the root is found to high numerical precision.
    )$root,
    error = function(e) NA_real_
  )

  # Keep only finite positive roots. A non-positive or non-finite result is treated as a numerical root-finding failure.
  if (!is.finite(root) || root <= 0) {
    return(list(R_raw = NA_real_, root_status = "uniroot_failed"))
  }

  list(R_raw = root, root_status = "found") # Successful root-finding.

}
################################################################################
# Compute ruin-related quantities for posterior draws under the lognormal gain-size model.
# For each posterior draw, the function:
#   1. computes the implied mean gain E[X] = exp(mu + sigma^2 / 2),
#   2. checks the net profit condition lambda * E[X] > c,
#   3. finds the positive Lundberg root when the condition holds,
#   4. computes psi(u) = exp(-R*u) when a root is found,
#   5. sets psi(u) = 1 when the net profit condition fails or when root-finding fails.
#
# The variable psi is the full posterior ruin probability, 
# while psi_given_net_profit_holds stores the conditional ruin probability
# only for draws where a finite Lundberg root is found.

ruin_prob_lognormal <- function(lambda, mu, sigma2, c_rate, u) {
  S <- length(lambda) # 10000 posterior draws

  implied_mean_gain <- exp(mu + sigma2 / 2)
  expected_gain_rate <- lambda * implied_mean_gain
  net_profit_holds <- expected_gain_rate > c_rate

  R_raw <- rep(NA_real_, S)
  psi <- rep(1, S)
  psi_given_net_profit_holds <- rep(NA_real_, S)
  root_status <- rep(NA_character_, S)

  for (s in seq_len(S)) {
    if (!net_profit_holds[s]) {
      root_status[s] <- "net_profit_fails"
    } else {
      root_result <- find_lundberg_root_lognormal(
        lambda = lambda[s],
        mu = mu[s],
        sigma2 = sigma2[s],
        c_rate = c_rate
      )

      R_raw[s] <- root_result$R_raw
      root_status[s] <- root_result$root_status

      if (identical(root_result$root_status, "found")) {
        psi_given_net_profit_holds[s] <- exp(-R_raw[s] * u)
        psi[s] <- psi_given_net_profit_holds[s]
      } else {
        psi[s] <- 1
      }
    }
  }

  psi <- pmin(pmax(psi, 0), 1)
  psi_given_net_profit_holds <- pmin(pmax(psi_given_net_profit_holds, 0), 1)

  data.frame(
    implied_mean_gain = implied_mean_gain,
    expected_gain_rate = expected_gain_rate,
    net_profit_holds = net_profit_holds,
    R_raw = R_raw,
    psi = psi,
    psi_given_net_profit_holds = psi_given_net_profit_holds,
    root_status = root_status,
    root_failed = net_profit_holds & root_status != "found",
    stringsAsFactors = FALSE
  )
}

##################################################################
# Add lognormal ruin quantities to the posterior draw table.
# This keeps the original posterior parameter draws and appends 
# implied gain rates, net profit indicators, Lundberg roots,
# ruin probabilities, and root-finding diagnostics.
add_lognormal_ruin_quantities <- function(draws, u, c_rate) {
  ruin_quantities <- ruin_prob_lognormal(
    lambda = draws$lambda,
    mu = draws$mu,
    sigma2 = draws$sigma2,
    c_rate = c_rate,
    u = u
  )
  cbind(draws, ruin_quantities)
}


# ============================================================
# 6. FULL CASE RUNNER                                 
# ============================================================
# Run the complete lognormal misspecification analysis for one case.
# This function connects the previous helper functions: 
#   1. extract sufficient statistics for log-gains, 
#   2. construct the moment-matched lognormal prior, 
#   3. compute posterior hyperparameters, 
#   4. simulate posterior draws, 
#   5. transform posterior draws into ruin-related quantities, 
#   6. collect compact diagnostics for reporting.
run_lognormal_misspecification_case <- function(events,
                                                summary,
                                                true_row,
                                                u,
                                                c_rate,
                                                S,
                                                case_name = NULL,
                                                lambda_concentration = 4,
                                                kappa0 = 4,
                                                nu0 = 4) {
  
  # Extract sufficient statistics from the observed gain events.
  log_stats <- extract_log_gain_stats(events = events, summary = summary)
  if (!is.null(case_name)) log_stats$Case <- case_name

  # Construct the prior for the lognormal misspecification model.
  prior_specs <- make_lognormal_misspec_prior(
    true_row = true_row,
    lambda_concentration = lambda_concentration,
    kappa0 = kappa0,
    nu0 = nu0
  )

  # Combine the prior and sufficient statistics to obtain posterior hyperparameters. 
  # This step does not simulate posterior draws yet. It only computes the parameters of the Gamma posterior for lambda 
  # and the normal-inverse-chi-square posterior for (mu, sigma^2).
  posterior_params <- normal_inverse_chisq_posterior(
    log_stats = log_stats,
    prior_specs = prior_specs
  )

  # Draw from the posterior distribution of the model parameters.
  # Each row in draws corresponds to one posterior draw of: lambda, mu, and sigma^2
  draws <- draw_normal_inverse_chisq(
    posterior_params = posterior_params,
    S = S,
    case_name = log_stats$Case[1]
  )

  # For each posterior draw, compute implied gain quantities,
  # check the net profit condition, solve for the Lundberg root when possible, 
  # and compute the corresponding ruin probability.
  draws <- add_lognormal_ruin_quantities(
    draws = draws,
    u = u,
    c_rate = c_rate
  )

  # Collect one-row diagnostics for this case.
  # The diagnostics combine: 
  #   - observed data summaries, 
  #   - simple MLE reference values, 
  #   - posterior probabilities for the net profit condition, 
  #   - posterior mass at psi = 1, 
  #   - numerical root-finding diagnostics.
  diagnostics <- data.frame(
    Case = log_stats$Case[1],
    prior_name = prior_specs$prior_name,
    S = S,
    n = log_stats$n[1],
    T = log_stats$T[1],
    sum_x = log_stats$sum_x[1],
    sum_y = log_stats$sum_y[1],
    ybar = log_stats$ybar[1],
    sy2 = log_stats$sy2[1],
    lambda_mle = log_stats$lambda_mle[1],
    mu_mle = log_stats$mu_mle[1],
    sigma2_mle = log_stats$sigma2_mle[1],
    posterior_pr_net_profit_holds = mean(draws$net_profit_holds),
    posterior_pr_net_profit_fails = mean(!draws$net_profit_holds),
    posterior_pr_psi_equals_one = mean(draws$psi == 1),
    root_failure_rate_unconditional = mean(draws$root_failed),
    root_failure_rate_given_net_profit_holds = ifelse(
      any(draws$net_profit_holds),
      mean(draws$root_failed[draws$net_profit_holds]),
      NA_real_
    ),
    stringsAsFactors = FALSE
  )

  # Return all intermediate and final objects.
  list(
    stats = log_stats,
    prior_specs = prior_specs,
    posterior_params = posterior_params,
    draws = draws,
    diagnostics = diagnostics
  )
}


# ============================================================
# End of 02_functions_lognormal_misspecification.R
# ============================================================
