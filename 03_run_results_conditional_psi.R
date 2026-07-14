# ============================================================
# 03_run_results_conditional_psi.R
# Baseline results for Bayesian single-path calibration in the classical continuous-time dual risk model
# ============================================================
#
# Model:
#   U(t) = u - c*t + sum_{i=1}^{N(t)} X_i
#
# Main focus in this result script:
#   - Compare sparse and rich single-path information.
#   - Use correct model specification.
#   - Use truth-centered baseline priors in the simulation study.
#   - Keep only thesis-relevant result figures.
#
# Bayesian model:
#   lambda ~ Gamma(a_lambda, b_lambda)
#   beta   ~ Gamma(a_beta, b_beta)
#
# Conjugate updates:
#   lambda | data ~ Gamma(a_lambda + n, b_lambda + T)
#   beta   | data ~ Gamma(a_beta + n, b_beta + sum(x_i))
#
# Important:
#   T is the full observation horizon, not the time of the last gain.
#   Posterior draws where the net profit condition fails are kept.
#   These draws are assigned psi(u) = 1.
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================
# Load packages used for plotting, data manipulation, and posterior summaries.
library(ggplot2)
library(dplyr)
library(posterior)

# ============================================================
# 1. SOURCE FUNCTIONS
# ============================================================
# Load model-specific helper functions.
# This script handles the analysis workflow, while the mathematical and Bayesian helper functions are kept in 02_functions_conditional_psi.R.
source("02_functions_conditional_psi.R")

# ============================================================
# 2. USER SETTINGS
# ============================================================
seed <- 999
set.seed(seed)

S <- 10000 # The number of posterior draws used to approximate the posterior distribution of the parameters and the induced ruin probability.

# Fixed model constants from the simulation design.
u <- 207.28 # The initial surplus
c_rate <- 91.65 / 12 # The deterministic monthly expense rate.

case_sparse <- "sparse_information"
case_rich <- "rich_information"
case_names <- c(case_sparse, case_rich)

simulation_dir_name <- "simulated_dual_risk_data"
results_dir_name <- "results"

analysis_label <- "baseline_truth_centered_correct_specification"

# Baseline prior before the real prior sensitivity analysis.
# The prior centers are set to the true simulation parameters for each case.
# Only the prior strength below is common across cases.
baseline_prior_name <- "truth_centered_baseline"
baseline_lambda_concentration <- 4
baseline_beta_concentration <- 4


# ============================================================
# 3. INPUT AND OUTPUT FOLDERS
# ============================================================
# Define input and output folders.
# The script is mainly designed to be run from the main project folder.
current_folder <- basename(normalizePath(getwd(), winslash = "/", mustWork = TRUE))

# If it is run from inside the simulation or results folder, "." is used to refer to the current working directory.
input_dir <- if (identical(current_folder, simulation_dir_name)) "." else simulation_dir_name 
results_dir <- if (identical(current_folder, results_dir_name)) "." else results_dir_name

# Create result subfolders for tables, posterior draws, and figures amd showWarnings = FALSE avoids warnings if the folders already exist.
tables_dir <- file.path(results_dir, "tables")
draws_dir <- file.path(results_dir, "draws")
figures_dir <- file.path(results_dir, "figures")
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(draws_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Just some comments
cat("Reading simulation files from:", normalizePath(input_dir, winslash = "/"), "\n")
cat("Saving results in:", normalizePath(results_dir, winslash = "/"), "\n")


# ============================================================
# 4. READ SIMULATION DATA
# ============================================================
# Read the simulated event logs and summary files for each information case.
# Each case is stored as a list with two components:
#   events - observed gain arrivals and gain sizes,
#   summary - case-level information such as the observation horizon.
read_case_files <- function(case_name) {
  list(
    events = read.csv(file.path(input_dir, paste0("events_", case_name, ".csv")), stringsAsFactors = FALSE),
    summary = read.csv(file.path(input_dir, paste0("summary_", case_name, ".csv")), stringsAsFactors = FALSE)
  )
}

sparse_data <- read_case_files(case_sparse)
rich_data <- read_case_files(case_rich)

# Read the true simulation parameters.
# These are used to construct truth-centered priors and to compare estimates against the known data-generating values
true_parameters <- read.csv(file.path(input_dir, "true_parameters_two_cases.csv"), stringsAsFactors = FALSE)

# ============================================================
# 5. TRUTH-CENTERED BASELINE PRIORS
# ============================================================
# Construct truth-centered baseline priors. Since this is a simulation study, the true parameter values are known.
# The baseline prior is therefore centered at the true lambda and beta values for each case.
get_true_row <- function(case_name) {
  true_parameters[true_parameters$Case == case_name, , drop = FALSE]
}

make_truth_centered_prior <- function(case_name) {
  true_i <- get_true_row(case_name)

  make_prior_specs(
    prior_name = baseline_prior_name,
    lambda_mean = true_i$lambda_true[1],
    lambda_concentration = baseline_lambda_concentration,
    beta_mean = true_i$beta_true[1],
    beta_concentration = baseline_beta_concentration
  )
}

# This is useful as a controlled reference analysis before studying prior sensitivity. 
# In a real empirical application, the true parameters would not be known and the prior would need to be justified from external information or domain knowledge.
prior_sparse <- make_truth_centered_prior(case_sparse)
prior_rich <- make_truth_centered_prior(case_rich)

prior_table <- dplyr::bind_rows(
  prior_specs_to_table(prior_sparse, case_sparse),
  prior_specs_to_table(prior_rich, case_rich)
)

# Save the prior table for transparency and reproducibility.
write.csv(prior_table, file.path(tables_dir, "prior_table_baseline_truth_centered.csv"), row.names = FALSE)

# ============================================================
# 6. RUN BASELINE BAYESIAN CALIBRATION
# ============================================================
# Run the baseline Bayesian calibration separately for the sparse and rich cases.
# run_case_analysis() extracts sufficient statistics, updates the Gamma posteriors, simulates posterior draws, and computes the induced ruin probability for each draw.
result_sparse <- run_case_analysis(
  events = sparse_data$events,
  summary = sparse_data$summary,
  prior_specs = prior_sparse,
  u = u,
  c_rate = c_rate,
  S = S,
  case_name = case_sparse
)

result_rich <- run_case_analysis(
  events = rich_data$events,
  summary = rich_data$summary,
  prior_specs = prior_rich,
  u = u,
  c_rate = c_rate,
  S = S,
  case_name = case_rich
)

# Combine posterior draws from both cases into one object for later summaries and plots.
posterior_draws_all <- dplyr::bind_rows( 
  result_sparse$draws,
  result_rich$draws
)

# Save posterior draws as RDS files so that later scripts can reuse the exact posterior samples without rerunning the calibration.
saveRDS(result_sparse$draws, file.path(draws_dir, paste0("posterior_draws_", case_sparse, ".rds")))
saveRDS(result_rich$draws, file.path(draws_dir, paste0("posterior_draws_", case_rich, ".rds")))
saveRDS(posterior_draws_all, file.path(draws_dir, "posterior_draws_all_cases_baseline.rds"))

# ============================================================
# 7. DATA DIAGNOSTICS AND MLE BASELINE
# ============================================================
# Build a diagnostic table comparing the observed simulated data with the known true simulation parameters
stats_all <- dplyr::bind_rows(
  result_sparse$stats,
  result_rich$stats
)

# The table checks how informative or unusual each observed path is:
#   - how many gains were observed compared with the expected number,
#   - how the MLE of lambda compares with the true lambda,
#   - how the MLE of beta compares with the true beta.
data_diagnostics_table <- stats_all %>%
  dplyr::left_join(true_parameters, by = "Case") %>%
  dplyr::mutate(
    observed_minus_expected_gains = n - expected_number_of_gains,
    observed_over_expected_gains = n / expected_number_of_gains,
    lambda_mle_over_true = lambda_mle / lambda_true,
    beta_mle_over_true = beta_mle / beta_true
  ) %>%
  dplyr::select(Case, T,
                expected_number_of_gains, n, observed_minus_expected_gains, observed_over_expected_gains, 
                lambda_true, lambda_mle, lambda_mle_over_true,
                beta_true, beta_mle, beta_mle_over_true,
                sum_x, psi_true_ultimate)

write.csv(data_diagnostics_table, file.path(tables_dir, "data_diagnostics_and_mle_table.csv"), row.names = FALSE)

# ============================================================
# 8. POSTERIOR SUMMARIES
# ============================================================
# This section converts posterior draws into summary tables.
# We report posterior summaries for the model parameters, the Lundberg root, and the induced ultimate ruin probability.
#
# Two versions of psi are summarized:
#   psi: Unconditional posterior ruin probability. This includes draws where the net profit condition fails, in which case ultimate ruin is set to 1.
#   psi_given_net_profit_holds: Conditional posterior ruin probability. This uses only draws where the net profit condition holds. Draws where the condition fails are NA.

############## Helper functions for posterior summaries ############## 
# These functions return NA if all values are missing.
# This is important for conditional quantities such as psi_given_net_profit_holds, where draws that violate the net profit condition are stored as NA.
safe_mean <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_median <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::median(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::sd(x, na.rm = TRUE)
}

q025 <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  unname(stats::quantile(x, 0.025, na.rm = TRUE))
}

q975 <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  unname(stats::quantile(x, 0.975, na.rm = TRUE))
}

############## Summarize posterior draws for one case ############## 
# The input should be the posterior draws for either the sparse or rich case.
# The output is a long-format table with one row per summarized variable.

summarise_case_draws <- function(draws) {
  vars <- c("lambda", "beta", "R_raw", "psi", "psi_given_net_profit_holds") # Variables to summarize.
  # lambda: Posterior draws of the gain arrival intensity.
  #
  # beta: Posterior draws of the exponential gain-size rate parameter. Since E[X] = 1 / beta, a larger beta implies smaller expected gains.
  #
  # R_raw: Posterior draws of the Lundberg root before any additional filtering.
  #
  # psi: Unconditional posterior ruin probability. This includes psi = 1 for draws where the net profit condition fails.
  #
  # psi_given_net_profit_holds: Conditional ruin probability, using only draws where the net profit condition holds. It is NA when the net profit condition fails.

  # Loop over the selected variables and compute posterior summary statistics. The result is stored in long format: one row per variable.
  out <- dplyr::bind_rows(
    lapply(vars, function(v) {
      x <- draws[[v]]
      data.frame(
        variable = v,
        mean = safe_mean(x),
        median = safe_median(x),
        sd = safe_sd(x),
        q025 = q025(x),
        q975 = q975(x),
        stringsAsFactors = FALSE
      )
    })
  )

  # Add case and prior identifiers to the summary table.
  # Each call to this function should contain draws from one case and one prior.
  out$Case <- unique(draws$Case)
  out$prior_name <- unique(draws$prior_name)
  out[, c("Case", "prior_name", "variable", "mean", "median", "sd", "q025", "q975")] # Reorder columns to make the table easier to read.

}

# Apply the summary function to both information cases and combine the results.
posterior_summary_long <- dplyr::bind_rows(
  summarise_case_draws(result_sparse$draws),
  summarise_case_draws(result_rich$draws)
)

# Save the posterior summary table.
# This table is useful for reporting posterior means, medians, standard deviations, and 95% credible intervals for parameters and ruin quantities.
write.csv(
  posterior_summary_long,
  file.path(tables_dir, "posterior_summary_long_baseline.csv"),
  row.names = FALSE
)

# Collect posterior Gamma parameters for lambda and beta. These document the conjugate posterior updates used in the baseline analysis.
posterior_params_all <- dplyr::bind_rows(
  result_sparse$posterior_params,
  result_rich$posterior_params
)

# Collect posterior diagnostics related to the net profit condition. These diagnostics show how often posterior draws imply certain ultimate ruin.
diagnostics_all <- dplyr::bind_rows(
  result_sparse$diagnostics,
  result_rich$diagnostics
)

# Save net profit diagnostics. 
# This table is important because posterior mass at psi = 1 is mainly driven by posterior draws where the net profit condition fails.

write.csv(
  diagnostics_all,
  file.path(tables_dir, "net_profit_posterior_diagnostics.csv"),
  row.names = FALSE
)

# Save the posterior Gamma parameters.
write.csv(
  posterior_params_all,
  file.path(tables_dir, "posterior_gamma_parameters.csv"),
  row.names = FALSE
)


# ============================================================
# 9. THESIS-FRIENDLY TABLES
# ============================================================
# This section reorganizes previous results into compact tables for reporting.

# Helper function for extracting one posterior summary statistic for a given case and variable.
# posterior_summary_long is created with dplyr and may behave like a tibble.
# [[statistic_name]] extracts the requested summary column as a plain vector.
get_summary_value <- function(case_name, variable_name, statistic_name) {
  row_id <- posterior_summary_long$Case == case_name & posterior_summary_long$variable == variable_name #   # Identify the row matching the requested case and posterior variable.
  as.numeric(posterior_summary_long[[statistic_name]][row_id][1]) # Extract the requested summary statistic and return it as a single numeric value.
}

# Helper function for extracting one case-level value from a given table.
# Use [[column_name]] so the result is a plain vector, not a one-column data frame/tibble.
get_table_value <- function(case_name, data, column_name) {
  as.numeric(data[[column_name]][data$Case == case_name][1]) # Select the requested column and return the value for the matching case.
}

# Create a case-level table focused on posterior summaries of the ruin probability.
ruin_summary_table <- data.frame(
  Case = case_names,

  # True ultimate ruin probability from the simulation design.
  # vapply() is used istead of these two code:
  #   1. get_table_value("sparse_information", true_parameters, "psi_true_ultimate")
  #   2. get_table_value("rich_information", true_parameters, "psi_true_ultimate")
  true_psi = vapply(
    case_names,
    get_table_value,
    numeric(1),
    data = true_parameters,
    column_name = "psi_true_ultimate"
  ),

  # Posterior mass where ultimate ruin is certain because the net profit condition fails.
  posterior_pr_net_profit_fails = vapply(
    case_names,
    get_table_value,
    numeric(1),
    data = diagnostics_all,
    column_name = "posterior_pr_net_profit_fails"
  ),

  # Posterior probability that the ruin probability is exactly one.
  posterior_pr_psi_equals_one = vapply(
    case_names,
    get_table_value,
    numeric(1),
    data = diagnostics_all,
    column_name = "posterior_pr_psi_equals_one"
  ),

  # Unconditional posterior summaries include the point mass at psi = 1.
  # These are useful diagnostics, but the unconditional 95% interval may be uninformative
  # when enough posterior mass is located exactly at psi = 1 (We can see it in the thesis).
  posterior_mean_psi_unconditional = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi",
    statistic_name = "mean"
  ),

  # Posterior median of the unconditional ruin probability.
  posterior_median_psi_unconditional = vapply( 
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi",
    statistic_name = "median"
  ),
  
### 95% credible interval for the unconditional posterior ruin probability ### 
  psi_unconditional_ci_lower = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi",
    statistic_name = "q025"
  ),

  psi_unconditional_ci_upper = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi",
    statistic_name = "q975"
  ),

### Conditional posterior summaries exclude the net-profit-failure draws ### 
  # These describe the continuous part of the posterior distribution of psi.
  posterior_mean_psi_given_net_profit_holds = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi_given_net_profit_holds",
    statistic_name = "mean"
  ),

  posterior_median_psi_given_net_profit_holds = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi_given_net_profit_holds",
    statistic_name = "median"
  ),

  psi_given_net_profit_holds_ci_lower = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi_given_net_profit_holds",
    statistic_name = "q025"
  ),

  psi_given_net_profit_holds_ci_upper = vapply(
    case_names,
    get_summary_value,
    numeric(1),
    variable_name = "psi_given_net_profit_holds",
    statistic_name = "q975"
  ),

  # Practical risk summaries that we choose to remain unconditional.
  posterior_pr_psi_gt_095_unconditional = vapply(
    case_names,
    function(case_name) {
      mean(posterior_draws_all$psi[posterior_draws_all$Case == case_name] > 0.95)
    },
    numeric(1)
  ),

  # Posterior probability of near-certain ruin under a stricter threshold.
  posterior_pr_psi_gt_099_unconditional = vapply(
    case_names,
    function(case_name) {
      mean(posterior_draws_all$psi[posterior_draws_all$Case == case_name] > 0.99)
    },
    numeric(1)
  ),

  stringsAsFactors = FALSE # Keep character columns as character strings. Just more defensive
)

# Save the main ruin probability summary table.
write.csv(ruin_summary_table, file.path(tables_dir, "ruin_probability_summary_table.csv"),
  row.names = FALSE) 

# Add true simulation values to the posterior summary table.
# left_join(x, y) = keep all rows from x and add the matching columns from y
parameter_summary_table <- posterior_summary_long %>%
  dplyr::left_join( 
    dplyr::bind_rows(
      # Convert true simulation values to long format so they can be joined to posterior summaries by Case and variable.
      data.frame(Case = true_parameters$Case, variable = "lambda", true_value = true_parameters$lambda_true),
      data.frame(Case = true_parameters$Case, variable = "beta", true_value = true_parameters$beta_true),
      data.frame(Case = true_parameters$Case, variable = "R_raw", true_value = true_parameters$R_raw_true),
      data.frame(Case = true_parameters$Case, variable = "psi", true_value = true_parameters$psi_true_ultimate),
      data.frame(Case = true_parameters$Case, variable = "psi_given_net_profit_holds", true_value = true_parameters$psi_true_ultimate)
    ),
    by = c("Case", "variable")
  ) %>%
  dplyr::select(Case, variable, true_value, mean, median, sd, q025, q975)

# Save posterior summaries together with true simulation values.
write.csv(
  parameter_summary_table,
  file.path(tables_dir, "parameter_posterior_summary_table.csv"),
  row.names = FALSE
)

#### Create a compact one-row-per-case table with the main thesis results ####
## This might be a good idea because we then have all the relevant informations in one placefor the thesis ##
thesis_friendly_result_table <- data.frame(
  Case = case_names,
  observed_gains_n = vapply(case_names, get_table_value, numeric(1), data = data_diagnostics_table, column_name = "n"),
  expected_gains_true = vapply(case_names, get_table_value, numeric(1), data = data_diagnostics_table, column_name = "expected_number_of_gains"),
  lambda_mle = vapply(case_names, get_table_value, numeric(1), data = data_diagnostics_table, column_name = "lambda_mle"),
  beta_mle = vapply(case_names, get_table_value, numeric(1), data = data_diagnostics_table, column_name = "beta_mle"),
  posterior_median_lambda = vapply(case_names, get_summary_value, numeric(1), variable_name = "lambda", statistic_name = "median"),
  posterior_median_beta = vapply(case_names, get_summary_value, numeric(1), variable_name = "beta", statistic_name = "median"),

  true_psi = vapply(case_names, get_table_value, numeric(1), data = true_parameters, column_name = "psi_true_ultimate"),
  posterior_pr_net_profit_fails = vapply(case_names, get_table_value, numeric(1), data = diagnostics_all, column_name = "posterior_pr_net_profit_fails"),
  posterior_pr_psi_equals_one = vapply(case_names, get_table_value, numeric(1), data = diagnostics_all, column_name = "posterior_pr_psi_equals_one"),

  posterior_median_psi_unconditional = vapply(case_names, get_summary_value, numeric(1), variable_name = "psi", statistic_name = "median"),
  posterior_median_psi_given_net_profit_holds = vapply(case_names, get_summary_value, numeric(1), variable_name = "psi_given_net_profit_holds", statistic_name = "median"),
  psi_given_net_profit_holds_ci_lower = vapply(case_names, get_summary_value, numeric(1), variable_name = "psi_given_net_profit_holds", statistic_name = "q025"),
  psi_given_net_profit_holds_ci_upper = vapply(case_names, get_summary_value, numeric(1), variable_name = "psi_given_net_profit_holds", statistic_name = "q975"),

  posterior_pr_psi_gt_095_unconditional = vapply(
    case_names,
    function(case_name) {
      mean(posterior_draws_all$psi[posterior_draws_all$Case == case_name] > 0.95)
    },
    numeric(1)
  ),

  posterior_pr_psi_gt_099_unconditional = vapply(
    case_names,
    function(case_name) {
      mean(posterior_draws_all$psi[posterior_draws_all$Case == case_name] > 0.99)
    },
    numeric(1)
  ),

  stringsAsFactors = FALSE
)

# Save the compact thesis-friendly result table.
write.csv(
  thesis_friendly_result_table,
  file.path(tables_dir, "thesis_friendly_result_table.csv"),
  row.names = FALSE
)

# Create a wider audit table that combines diagnostics and posterior settings.
full_result_table <- data_diagnostics_table %>%
  dplyr::left_join(diagnostics_all, by = c("Case", "n", "T", "sum_x", "lambda_mle", "beta_mle")) %>% # Add posterior diagnostics to the data diagnostic table.
  dplyr::left_join(posterior_params_all, by = c("Case", "prior_name")) %>% # Add posterior Gamma parameters for lambda and beta.
  dplyr::mutate( # Add analysis metadata for reproducibility.
    analysis_label = analysis_label,
    posterior_draws_S = S,
    seed = seed
  )

# Save the full audit table with diagnostics, posterior parameters, and metadata.
write.csv(
  full_result_table,
  file.path(tables_dir, "full_result_table.csv"),
  row.names = FALSE
)

# ============================================================
# 10. FIGURE HELPERS
# ============================================================
# save_plot_both() = Save a ggplot object in both PNG and PDF format using the same base filename.
save_plot_both <- function(plot, filename_base, width = 8, height = 5) {
  ggplot2::ggsave( # Save as PNG-file
    file.path(figures_dir, paste0(filename_base, ".png")),
    plot = plot,
    width = width,
    height = height,
    dpi = 300 # High solution (Might be good for our thesis)
  )

  ggplot2::ggsave( # Save as PDF-file
    file.path(figures_dir, paste0(filename_base, ".pdf")),
    plot = plot,
    width = width,
    height = height
  )
}

# Map internal case names to cleaner labels for figures.
case_label_map <- c(
  "sparse_information" = "Sparse information",
  "rich_information" = "Rich information"
)

# Convert internal case names into readable labels. Look up the display label and remove the original vector name.
make_case_label <- function(x) {
  out <- unname(case_label_map[x])
  ifelse(is.na(out), x, out) # If no label is found, keep the original case name.
}

# Build plotting data for one simulated surplus path.
make_path_data <- function(events, summary) {
  case_name <- summary$Case[1] # Extract the case name from the case-level summary table.
  
  # Use T_for_lambda_likelihood if available; otherwise use observation_horizon.
  # This keeps the code compatible with summary files that use the older column name.
  T_obs <- if ("T_for_lambda_likelihood" %in% names(summary)) {
    summary$T_for_lambda_likelihood[1]
  } else {
    summary$observation_horizon[1]
  }

  # Start the path at the initial surplus u at time zero.
  rows <- list(data.frame(Case = case_name, time = 0, surplus = u, point_type = "start"))

  # Add jump points only if at least one gain event was observed.
  if (nrow(events) > 0) { 
    for (i in seq_len(nrow(events))) { # Loop over each observed gain event.
      # Add the surplus level immediately before the gain jump.
      rows[[length(rows) + 1]] <- data.frame(
        Case = case_name,
        time = events$arrival_time[i],
        surplus = events$surplus_before_gain[i],
        point_type = "before_gain"
      )
      # Add the surplus level immediately after the gain jump at the same time.
      rows[[length(rows) + 1]] <- data.frame(
        Case = case_name,
        time = events$arrival_time[i],
        surplus = events$surplus_after_gain[i],
        point_type = "after_gain"
      )
    }
  }

  # End the path at the full observation horizon using the final surplus.
  rows[[length(rows) + 1]] <- data.frame(
    Case = case_name,
    time = T_obs,
    surplus = summary$final_surplus[1],
    point_type = "end"
  )

  # Combine all path points into one data frame for plotting.
  # ?do.call for extra information
  do.call(rbind, rows)
}

# Store true simulation values in long format for use in figures.
true_values_long <- dplyr::bind_rows(
  data.frame(Case = true_parameters$Case, variable = "lambda", true_value = true_parameters$lambda_true),
  data.frame(Case = true_parameters$Case, variable = "beta", true_value = true_parameters$beta_true),
  data.frame(Case = true_parameters$Case, variable = "R_raw", true_value = true_parameters$R_raw_true),
  data.frame(Case = true_parameters$Case, variable = "psi", true_value = true_parameters$psi_true_ultimate)
)


# ============================================================
# 11. FIGURES
# ============================================================

# ============================================================
# 11.1 Observed simulated paths
# ============================================================
# Plot the observed simulated surplus paths used as input data for the Bayesian calibration.

# Build one plotting data frame containing the observed paths for both sparse and rich information cases.
path_data <- dplyr::bind_rows(
  make_path_data(sparse_data$events, sparse_data$summary),
  make_path_data(rich_data$events, rich_data$summary)
)

# Create facet labels that show the case name, observation horizon and number of observed gains.
path_labels <- data_diagnostics_table %>%
  dplyr::mutate(
    Case_label = paste0(make_case_label(Case)," (T = ", T, ", n = ", n,")")) %>% # Example: Sparse information (T = 36, n = 8)
  dplyr::select(Case, Case_label)

# Attach the readable facet label to every path point.
path_data <- path_data %>%
  dplyr::left_join(path_labels, by = "Case")

# Plot surplus against time, connecting points within each case.
p_paths <- ggplot(path_data, aes(x = time, y = surplus, group = Case)) +
  geom_path(linewidth = 0.75) + # Use geom_path() to connect points in the constructed event order, including vertical jumps at gain times.
  geom_point( # Mark the post-gain surplus levels to show where positive jumps occurred.
    data = path_data[path_data$point_type == "after_gain", ],
    size = 1.25
  ) +
  facet_wrap(~ Case_label, scales = "free_x") + # Use separate x-axis scales because the two cases have different time horizons.s
  labs(
    title = "Observed single paths from the simulated dual risk model",
    subtitle = "Panels use different time horizons: sparse T = 36 months, rich T = 180 months",
    x = "Time in months",
    y = "Surplus U(t)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# Save the observed path figure in both PNG and PDF format.
save_plot_both(
  p_paths,
  "observed_single_paths",
  width = 9,
  height = 4.5
)


# ============================================================
# 11.2 Prior, likelihood, and posterior for lambda and beta
# ============================================================

# Build plotting data for prior, normalized likelihood, and posterior curves for either lambda or beta.
make_prior_likelihood_posterior_data <- function(parameter_name,
                                                 prior_table,
                                                 stats_all,
                                                 posterior_params_all,
                                                 grid_length = 800) {
  rows <- list() 

  # Create curves separately for each information case.
  for (case_i in unique(stats_all$Case)) {
    # Select the prior parameters for the current case and parameter.
    prior_i <- prior_table %>%
      dplyr::filter(
        Case == case_i,
        parameter == parameter_name
      )

    # Extract Gamma prior shape and rate parameters.
    prior_shape <- prior_i$shape[1]
    prior_rate <- prior_i$rate[1]

    # Select observed sufficient statistics for the current case.
    stats_i <- stats_all %>%
      dplyr::filter(Case == case_i)

    # The likelihood is normalized only for plotting; it is not a prior.
    if (parameter_name == "lambda") {
      # N(T) | lambda ~ Poisson(lambda*T)
      # L(lambda) ∝ lambda^n exp(-T*lambda)
      # Normalized as Gamma(n + 1, T).
      likelihood_shape <- stats_i$n[1] + 1
      likelihood_rate <- stats_i$T[1]
    }

    # The normalized likelihood is used as a visual summary of the information in the data.
    if (parameter_name == "beta") {
      # X_i | beta ~ Exponential(beta)
      # L(beta) ∝ beta^n exp(-beta*sum_x)
      # Normalized as Gamma(n + 1, sum_x).
      likelihood_shape <- stats_i$n[1] + 1
      likelihood_rate <- stats_i$sum_x[1]
    }

    # Select posterior parameters for the current case.
    post_i <- posterior_params_all %>% dplyr::filter(Case == case_i)

    # Extract Gamma posterior parameters for lambda.
    if (parameter_name == "lambda") {
      posterior_shape <- post_i$lambda_shape_post[1]
      posterior_rate <- post_i$lambda_rate_post[1]
    }

    # Extract Gamma posterior parameters for beta.
    if (parameter_name == "beta") {
      posterior_shape <- post_i$beta_shape_post[1]
      posterior_rate <- post_i$beta_rate_post[1]
    }

    # Use the largest 99.9% quantile so the grid covers nearly all mass of the prior, likelihood, and posterior curves.
    x_max <- max(
      qgamma(0.999, shape = prior_shape, rate = prior_rate),
      qgamma(0.999, shape = likelihood_shape, rate = likelihood_rate),
      qgamma(0.999, shape = posterior_shape, rate = posterior_rate),
      na.rm = TRUE
    ) 

    x_grid <- seq(0, x_max, length.out = grid_length) # Create an evenly spaced grid of parameter values for plotting densities.

    # Add the prior density curve evaluated on the common x-grid.
    rows[[length(rows) + 1]] <- data.frame(
      Case = case_i,
      parameter = parameter_name,
      curve = "Prior",
      value = x_grid,
      density = dgamma(x_grid, shape = prior_shape, rate = prior_rate),
      stringsAsFactors = FALSE
    )

    # Add the likelihood shape normalized as a Gamma density for visual comparison.
    rows[[length(rows) + 1]] <- data.frame(
      Case = case_i,
      parameter = parameter_name,
      curve = "Normalized likelihood",
      value = x_grid,
      density = dgamma(x_grid, shape = likelihood_shape, rate = likelihood_rate),
      stringsAsFactors = FALSE
    )

    # Add the posterior density curve implied by the conjugate update.
    rows[[length(rows) + 1]] <- data.frame(
      Case = case_i,
      parameter = parameter_name,
      curve = "Posterior",
      value = x_grid,
      density = dgamma(x_grid, shape = posterior_shape, rate = posterior_rate),
      stringsAsFactors = FALSE
    )
  }

  # Combine all curve data frames into one plotting table.
  # 2 cases × 3 curves × 800 grid points = around 4800 rows per parameter.
  out <- do.call(rbind, rows)

  # Fix the curve order used in legends and manual scales.
  out$curve <- factor(
    out$curve,
    levels = c("Prior", "Normalized likelihood", "Posterior")
  )

  out
}

# Shared ordering for curves in legends and manual plot scales.
curve_order <- c("Prior", "Normalized likelihood", "Posterior")

# Shorter display labels for the legend.
curve_labels <- c("Prior" = "Prior",
                  "Normalized likelihood" = "Likelihood",
                  "Posterior" = "Posterior")

curve_colors <- c("Prior" = "#D55E00",
                  "Normalized likelihood" = "#009E73",
                  "Posterior" = "#0072B2")

# Different line types make the curves distinguishable even without color.
curve_linetypes <- c("Prior" = "dotted",
                     "Normalized likelihood" = "dotdash",
                     "Posterior" = "solid")

curve_linewidths <- c("Prior" = 0.9,
                      "Normalized likelihood" = 1.0,
                      "Posterior" = 1.15)

# Prepare curve data by fixing curve order and creating ordered case labels.
prepare_plp_data <- function(plp_data) {
  plp_data %>%
    dplyr::mutate(
      curve = factor(curve, levels = curve_order),
      Case_label = factor(make_case_label(Case), 
                          levels = c("Sparse information", "Rich information")))
}

# Prepare true-value reference lines so they match the same facet labels.
prepare_vline_data <- function(vline_data) {
  vline_data %>%
    dplyr::mutate(
      Case_label = factor(make_case_label(Case), 
                          levels = c("Sparse information", "Rich information")))
  }

# Compute x-axis limits based on the visible parts of the curves while still including the true-value reference lines.
get_visible_xlim <- function(plp_data, vline_data, eps = 0.002, pad = 0.08) {
  visible_data <- plp_data %>%
    dplyr::group_by(Case, curve) %>%
    dplyr::mutate(
      max_density = max(density, na.rm = TRUE),
      relative_density = ifelse(max_density > 0, density / max_density, 0) # Keep only parts of each curve whose density is visibly above zero.
    ) %>%
    dplyr::ungroup() %>%
    dplyr::filter(relative_density >= eps)

  # Include both visible curve values and true values when setting x-limits.
  x_range <- range(c(visible_data$value, vline_data$true_value),na.rm = TRUE)

  # Add a small padding so curves and reference lines are not placed on the plot edge.
  x_padding <- diff(x_range) * pad

  c(max(0, x_range[1] - x_padding), x_range[2] + x_padding) # Lower x and Upper x
}

# Create and save a prior-likelihood-posterior plot with true-value reference lines.
make_plp_plot <- function(plp_data, # curve data
                          vline_data, # true value
                          title, x_label,
                          filename_base,
                          width = 7.6,
                          height = 4.0) {
  
  # Prepare plotting data and compute readable x-axis limits.
  plot_data <- prepare_plp_data(plp_data)
  vline_data <- prepare_vline_data(vline_data)
  x_limits <- get_visible_xlim(plot_data, vline_data)

  # Map each curve type to color, line type, and line width.
  p <- ggplot(plot_data, aes(x = value, y = density, 
                             color = curve, linetype = curve, linewidth = curve)) +
    geom_line(lineend = "round") +
    # Add a vertical reference line at the true simulation value.
    geom_vline(
      data = vline_data,
      aes(xintercept = true_value),
      inherit.aes = FALSE, # Do not inherit curve aesthetics because the reference-line data has different columns.
      linewidth = 0.55,
      linetype = "longdash",
      color = "grey20"
    ) +
    # Use separate y-axis scales because density peaks can differ strongly by case.
    facet_wrap(~ Case_label, nrow = 1, scales = "free_y") +
    coord_cartesian(xlim = x_limits, clip = "off") + # Zoom to readable x-limits without dropping data from the plot.
    
    # Apply consistent colors and legend labels for the curve types.
    scale_color_manual(
      values = curve_colors,
      breaks = curve_order,
      labels = curve_labels,
      name = NULL
    ) +
    # Use line types for accessibility but suppress a separate linetype legend.
    scale_linetype_manual(
      values = curve_linetypes,
      breaks = curve_order,
      labels = curve_labels,
      guide = "none"
    ) +
    scale_linewidth_manual(
      values = curve_linewidths,
      guide = "none"
    ) +
    # Show the correct line types and widths inside the single color legend.
    guides(
      color = guide_legend(nrow = 1,
                           override.aes = list(
                             linetype = unname(curve_linetypes[curve_order]),
                             linewidth = unname(curve_linewidths[curve_order])))) +
    labs(title = title,
         x = x_label,
         y = "Density") +
    
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.title.position = "plot",
      legend.position = "bottom",
      legend.text = element_text(size = 10),
      legend.key.width = grid::unit(1.6, "cm"),
      legend.margin = margin(t = 2, b = 2),
      strip.text = element_text(face = "bold", size = 11),
      panel.spacing = grid::unit(1.2, "lines"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.25),
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10),
      plot.margin = margin(6, 10, 6, 6)
    )

  save_plot_both(p, filename_base, width = width, height = height)
  p
}

# Create prior, likelihood, and posterior curve data for lambda.
lambda_plp_data <- make_prior_likelihood_posterior_data(
  parameter_name = "lambda",
  prior_table = prior_table,
  stats_all = stats_all,
  posterior_params_all = posterior_params_all
)

# Create prior, likelihood, and posterior curve data for beta.
beta_plp_data <- make_prior_likelihood_posterior_data(
  parameter_name = "beta",
  prior_table = prior_table,
  stats_all = stats_all,
  posterior_params_all = posterior_params_all
)

# Extract true lambda values for vertical reference lines.
lambda_true_vline <- true_values_long %>%
  dplyr::filter(variable == "lambda")

# Extract true beta values for vertical reference lines.
beta_true_vline <- true_values_long %>%
  dplyr::filter(variable == "beta")

# Plot and save the prior-likelihood-posterior comparison for lambda.
p_lambda_plp <- make_plp_plot(
  plp_data = lambda_plp_data,
  vline_data = lambda_true_vline,
  title = "Lambda: prior, likelihood, and posterior",
  x_label = expression(lambda),
  filename_base = "prior_likelihood_posterior_lambda_baseline"
)

# Plot and save the prior-likelihood-posterior comparison for beta.
p_beta_plp <- make_plp_plot(
  plp_data = beta_plp_data,
  vline_data = beta_true_vline,
  title = "Beta: prior, likelihood, and posterior",
  x_label = expression(beta),
  filename_base = "prior_likelihood_posterior_beta_baseline"
)


# ============================================================
# 11.3 Posterior distribution of ultimate ruin probability
# Visualize the posterior distribution of the ultimate ruin probability psi(u).
# Both the conditional continuous part and the full posterior draws are shown.
# ============================================================

# Keep only posterior draws where the net profit condition holds. These draws form the continuous part of the posterior distribution of psi(u).
posterior_draws_psi_given_net_profit_holds <- posterior_draws_all %>%
  dplyr::filter(net_profit_holds) %>%
  dplyr::mutate(
    Case_label = factor(make_case_label(Case), 
                        levels = c("Sparse information", "Rich information")))

# Extract the true simulated psi(u) values for vertical reference lines.
true_psi_vline <- true_values_long %>%
  dplyr::filter(variable == "psi") %>%
  dplyr::mutate(
    Case_label = factor(make_case_label(Case), 
                        levels = c("Sparse information", "Rich information")))

# Plot the conditional posterior draws of psi(u), excluding net-profit-failure draws.
p_psi_continuous <- ggplot(
  posterior_draws_psi_given_net_profit_holds,
  aes(x = psi_given_net_profit_holds, linetype = Case_label)) +
  geom_density(linewidth = 0.9, na.rm = TRUE) + # Draw a smoothed density estimate for the conditional posterior distribution.
  geom_vline( # Add true psi(u) as vertical reference lines without inheriting the density aesthetics.
    data = true_psi_vline,
    aes(xintercept = true_value, linetype = Case_label),
    linewidth = 0.55,
    show.legend = FALSE,
    inherit.aes = FALSE
  ) +
  labs(
    title = "Posterior distribution of ultimate ruin probability when net profit holds",
    subtitle = "Only posterior draws satisfying the net profit condition are shown",
    x = expression(psi(u)),
    y = "Density",
    linetype = "Case"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom",
        panel.grid.minor = element_blank())

# Save the conditional posterior density plot for psi(u).
save_plot_both(p_psi_continuous, 
               "posterior_psi_less_than_one_baseline",
               width = 8, height = 5)

######## Histogram for all posterior draws ######## 
bin_width_psi <- 0.025

# Compute the posterior point mass at psi(u) = 1 for each case.
# This corresponds to posterior draws where the net profit condition fails.
psi_equal_one_table <- posterior_draws_all %>%
  dplyr::group_by(Case) %>%
  dplyr::summarise(
    pr_psi_equal_1 = mean(psi == 1),
    label = paste0("psi = 1: ", round(100 * pr_psi_equal_1, 1), "%"),
    .groups = "drop" # Drop grouping after summarising to return a regular table.
  ) %>%
  dplyr::mutate(
    Case_label = factor(
      make_case_label(Case),
      levels = c("Sparse information", "Rich information")
    )
  )

# Save the posterior mass at psi(u) = 1 for reporting.
write.csv(psi_equal_one_table,
          file.path(tables_dir, "posterior_mass_at_psi_one_table.csv"),
          row.names = FALSE)

# Keep all posterior draws, including psi(u) = 1, for the full histogram.
posterior_draws_for_histogram <- posterior_draws_all %>%
  dplyr::mutate(Case_label = factor(make_case_label(Case),
                        levels = c("Sparse information", "Rich information")))


p_psi_histogram <- ggplot(posterior_draws_for_histogram,
                          aes(x = psi) # Plot the unconditional posterior draws of psi(u).
                          ) +
  geom_histogram(binwidth = bin_width_psi,
                 boundary = 0, # bin starts at 0
                 closed = "right", # Interval (0,1]. Use right-closed bins so exact psi(u) = 1 values are included in the last bin.
                 na.rm = TRUE) +
  # Comment each panel with the posterior mass at exact psi(u) = 1.
  geom_text(data = psi_equal_one_table,
            aes(x = 0.72, y = Inf,
                label = label),
            inherit.aes = FALSE,
            hjust = 0, vjust = 1.5, size = 3.4
            ) +
  facet_wrap(~ Case_label, ncol = 1) +
  labs(title = "Histogram of posterior draws of ultimate ruin probability",
       subtitle = "The rightmost bin contains values close to 1, including exact psi = 1",
       x = expression(psi(u)), y = "Count"
       ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"),
        panel.grid.minor = element_blank()
        )

save_plot_both(
  p_psi_histogram,
  "posterior_psi_histogram_all_draws_baseline",
  width = 8,
  height = 6
)


# ============================================================
# 12. FINAL MESSAGE
# ============================================================

cat("\nBaseline analysis completed.\n")
cat("Posterior draws per case:", S, "\n")
cat("Sparse observed gains:", result_sparse$stats$n, "\n")
cat("Rich observed gains:", result_rich$stats$n, "\n")
cat("Tables saved in:", normalizePath(tables_dir, winslash = "/"), "\n")
cat("Figures saved in:", normalizePath(figures_dir, winslash = "/"), "\n")

# ============================================================
# End of 03_run_results_cleaned.R
# ============================================================
