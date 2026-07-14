# ============================================================
# 04_prior_sensitivity.R
# Prior sensitivity analysis for Bayesian single-path calibration
# in the classical continuous-time dual risk model
# ============================================================
#
# Research question:
#   How sensitive are the estimated ruin probabilities to different
#   prior assumptions?
#
# Model:
#   U(t) = u - c*t + sum_{i=1}^{N(t)} X_i
#
# Main idea:
#   We keep the simulated data, likelihood, model specification, u, c,
#   and posterior ruin calculation fixed. Only the prior assumptions are
#   changed.
#
# This file deliberately reuses functions from 02_functions_conditional_psi.R:
#   - make_prior_specs()
#   - prior_specs_to_table()
#   - run_case_analysis()
#   - and the lower-level functions called by run_case_analysis()
#
# Therefore, this file should be read as an analysis script, not as a new
# toolbox of model functions.
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================
library(ggplot2)
library(dplyr)

# ============================================================
# 1. SOURCE EXISTING MODEL FUNCTIONS
# ============================================================
## Important:
# This file contains the model-specific functions for the classical continuous-time dual risk model with exponential gain sizes.
# We reuse those functions instead of redefining them here.
source("02_functions_conditional_psi.R")

# ============================================================
# 2. USER SETTINGS
# ============================================================
seed <- 999
set.seed(seed)

# Number of posterior draws for each Case x prior combination.
S <- 10000

# Same fixed model constants as in the baseline analysis.
# u is the initial surplus and c_rate is the deterministic monthly expense rate.
# These are kept fixed so that the sensitivity analysis only changes the priors.
u <- 207.28
c_rate <- 91.65 / 12

case_sparse <- "sparse_information"
case_rich <- "rich_information"
case_names <- c(case_sparse, case_rich)

### Folder and label settings ###
# The simulated data are read from simulation_dir_name.
# All outputs from this analysis are saved separately in results_dir_name.
# analysis_label is stored in result tables to identify this analysis version.
simulation_dir_name <- "simulated_dual_risk_data"
results_dir_name <- "result_prior_sensitivity"
analysis_label <- "prior_sensitivity_correct_specification"

# This prior is used as the reference point when measuring sensitivity.
# It matches the baseline prior from 03_run_results_conditional_psi.R.
baseline_prior_name <- "truth_centered_baseline"

# ============================================================
# 3. INPUT AND OUTPUT FOLDERS
# ============================================================
# This script reads the simulated data from simulated_dual_risk_data
# and saves all prior sensitivity outputs in a separate folder:
#
#   result_prior_sensitivity/
#     tables/
#     draws/
#     figures/
#
# This avoids mixing baseline results from 03_run_results_conditional_psi.R with prior sensitivity results from this script.

# getwd() = get working directory
# normalizePath() = standardize the path
# basename() = take the last name of the map
current_folder <- basename(normalizePath(getwd(), winslash = "/", mustWork = TRUE))


# Try to find the simulated data folder:
# 1. If I am already inside it, use "."
# 2. Else, if it exists in the current folder, use "simulated_dual_risk_data"
# 3. Else, if it exists one level above, use "../simulated_dual_risk_data"
# 4. Else, stop because the data folder cannot be found
if (identical(current_folder, simulation_dir_name)) {
  input_dir <- "." # "." = current working folder
} else if (dir.exists(simulation_dir_name)) {
  input_dir <- simulation_dir_name
} else if (dir.exists(file.path("..", simulation_dir_name))) {
  input_dir <- file.path("..", simulation_dir_name) # ".." = parent folder, one level above "."
} else {
  stop("Could not find the folder '", simulation_dir_name, "'. ",
       "Run the script from the main project folder or check the folder name."
  )
}

# Define the output folder for the prior sensitivity results.
# If the script is already run from inside the result folder, use the current folder.
if (identical(current_folder, results_dir_name)) {
  results_dir <- "."
} else {
  results_dir <- results_dir_name
}

# Separate output paths for tables, posterior draws and figures.
tables_dir <- file.path(results_dir, "tables")
draws_dir <- file.path(results_dir, "draws")
figures_dir <- file.path(results_dir, "figures")

# Create the output folders if they do not already exist.
# Existing folders are allowed and do not produce warnings.
dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(draws_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

cat("Reading simulation files from:", normalizePath(input_dir, winslash = "/"), "\n")
cat("Saving prior sensitivity results in:", normalizePath(results_dir, winslash = "/"), "\n")
cat("Tables will be saved in:", normalizePath(tables_dir, winslash = "/"), "\n")
cat("Draws will be saved in:", normalizePath(draws_dir, winslash = "/"), "\n")
cat("Figures will be saved in:", normalizePath(figures_dir, winslash = "/"), "\n")


# ============================================================
# 4. READ SIMULATION DATA
# We read the same simulated data used in the baseline analysis.
# No new data are simulated here.
# ============================================================
# Initialise a list that will store the event data and summary data for each case.
case_data <- list()

# Read the simulated event log and summary file for each information scenario.
# Each case is stored as a list with two components:
#   events  = observed gain events,
#   summary = case-level information such as the observation horizon.s
for (case_name in case_names) {
  case_data[[case_name]] <- list(
    events = read.csv(file.path(input_dir, paste0("events_", case_name, ".csv")),
                      stringsAsFactors = FALSE),
    summary = read.csv(file.path(input_dir, paste0("summary_", case_name, ".csv")),
                       stringsAsFactors = FALSE))
  }

true_parameters <- read.csv(file.path(input_dir, "true_parameters_two_cases.csv"),
                            stringsAsFactors = FALSE)

# ============================================================
# 5. PRIOR SENSITIVITY DESIGN
# ============================================================
# We use two types of prior sensitivity checks.
# 1. Prior strength sensitivity:
#    The prior is centered at the true simulation parameters, but the
#    concentration is varied. This asks whether conclusions change when
#    the analyst is more or less confident about a correctly centered prior.
#
# 2. Prior location sensitivity:
#    The concentration is fixed or strengthened, but the prior mean is moved
#    in optimistic or pessimistic directions. This asks whether conclusions
#    change when the analyst's prior assumptions are systematically wrong.
#
# Interpretation of multipliers:
#   lambda_multiplier > 1 -> prior expects more frequent gains.
#   lambda_multiplier < 1 -> prior expects less frequent gains.
#
#   beta is the exponential rate parameter, so E[X] = 1 / beta.
#   beta_multiplier > 1 -> prior expects smaller gain sizes.
#   beta_multiplier < 1 -> prior expects larger gain sizes.
#
# Therefore:
#   optimistic prior = larger lambda and smaller beta.
#   pessimistic prior = smaller lambda and larger beta.

# Each row in prior_grid defines one prior scenario.
# The truth-centered priors keep the prior mean fixed at the true simulation value and only vary the concentration parameter s.
# The optimistic and pessimistic priors shift the prior means away from the truth.
prior_grid <- data.frame(
  prior_name = c(
    "truth_very_diffuse",
    "truth_diffuse",
    "truth_weak",
    "truth_centered_baseline",
    "truth_strong",
    "truth_very_strong",
    "pessimistic_moderate",
    "optimistic_moderate",
    "pessimistic_strong",
    "optimistic_strong"
  ),
  prior_display_name = c(
    "Truth-centered, s = 0.5",
    "Truth-centered, s = 1",
    "Truth-centered, s = 2",
    "Truth-centered, s = 4",
    "Truth-centered, s = 10",
    "Truth-centered, s = 25",
    "Pessimistic, s = 4",
    "Optimistic, s = 4",
    "Pessimistic, s = 10",
    "Optimistic, s = 10"
  ),
  prior_family = c(rep("Prior strength", 6),
                   rep("Prior location", 4)),
  prior_interpretation = c(
    "Truth-centered prior with very low prior concentration.",
    "Truth-centered prior with low prior concentration.",
    "Truth-centered prior with weak prior concentration.",
    "Truth-centered baseline prior used in the first analysis.",
    "Truth-centered prior with strong prior concentration.",
    "Truth-centered prior with very strong prior concentration.",
    "Prior expects fewer arrivals and smaller gain sizes.",
    "Prior expects more arrivals and larger gain sizes.",
    "Strong prior expects fewer arrivals and smaller gain sizes.",
    "Strong prior expects more arrivals and larger gain sizes."
  ),
  lambda_multiplier = c(1, 1, 1, 1, 1, 1, 0.67, 1.50, 0.67, 1.50),
  beta_multiplier = c(1, 1, 1, 1, 1, 1, 1.50, 0.67, 1.50, 0.67),
  lambda_concentration = c(0.5, 1, 2, 4, 10, 25, 4, 4, 10, 10),
  beta_concentration = c(0.5, 1, 2, 4, 10, 25, 4, 4, 10, 10),
  stringsAsFactors = FALSE)

write.csv(prior_grid, file.path(tables_dir, "prior_sensitivity_design_grid.csv"),
          row.names = FALSE)


# ============================================================
# 6. RUN PRIOR SENSITIVITY ANALYSIS
# ============================================================
# The loop below runs every Case x prior combination.
# The actual Bayesian update and ruin probability calculation are handled by
# run_case_analysis(), which is imported from 02_functions_conditional_psi.R.

# Initialize containers for all Case x prior results.
# results_list stores the full output from each Bayesian calibration.
# prior_table_list stores the prior specification used in each run.
# result_id is a running index used to store each result in a simple list.
results_list <- list()
prior_table_list <- list()
result_id <- 0

# Loop over the two information scenarios: sparse information and rich information.
for (case_name in case_names) {
  true_i <- true_parameters[true_parameters$Case == case_name, , drop = FALSE] # Extract the true simulation parameters for the current case. drop = FALSE keeps the result as a data frame, even when only one row is selected.
  
  if (nrow(true_i) != 1) {
    stop("Expected exactly one row in true_parameters for case: ", case_name)
  }

  # Loop over all prior specifications in the prior design grid.
  for (j in seq_len(nrow(prior_grid))) {
    # Construct the actual Gamma prior specification for the current Case x prior row.
    prior_specs_j <- make_prior_specs(
      prior_name = prior_grid$prior_name[j],
      lambda_mean = true_i$lambda_true[1] * prior_grid$lambda_multiplier[j],
      lambda_concentration = prior_grid$lambda_concentration[j],
      beta_mean = true_i$beta_true[1] * prior_grid$beta_multiplier[j],
      beta_concentration = prior_grid$beta_concentration[j]
    )

    result_id <- result_id + 1

    cat("Running", case_name, "with prior", prior_specs_j$prior_name, "(", result_id, "of", length(case_names) * nrow(prior_grid), ")\n")

    # Run the Bayesian calibration for this Case x prior combination.
    # run_case_analysis() performs the posterior update, posterior simulation, net profit check, Lundberg root calculation, and ruin probability calculation.
    results_list[[result_id]] <- run_case_analysis(
      events = case_data[[case_name]]$events,
      summary = case_data[[case_name]]$summary,
      prior_specs = prior_specs_j,
      u = u,
      c_rate = c_rate,
      S = S,
      case_name = case_name
    )

    # Convert the prior specification to a table row.
    # This is saved separately so the exact prior used in each run is documented.
    prior_table_list[[result_id]] <- prior_specs_to_table(
      prior_specs = prior_specs_j,
      case_name = case_name
    )
  }
}


# ============================================================
# 7. COLLECT POSTERIOR DRAWS, PRIORS, POSTERIOR PARAMETERS
# ============================================================

# Combine posterior draws from all Case x prior runs into one data frame.
# Then attach prior metadata from prior_grid so each draw can be linked
# to its prior assumption and analysis label.
posterior_draws_all <- dplyr::bind_rows(
  lapply(results_list, function(x) x$draws)
) %>%
  dplyr::left_join(prior_grid, by = "prior_name") %>%
  dplyr::mutate(
    analysis_label = analysis_label
  )

# Combine all prior specifications into one table.
# This table documents the exact prior parameters used for each Case x prior run.
prior_table_all <- dplyr::bind_rows(prior_table_list) %>%
  dplyr::left_join(prior_grid, by = "prior_name") %>%
  dplyr::mutate(
    analysis_label = analysis_label
  )

# Combine posterior Gamma parameters from all runs.
# These parameters show how each prior was updated by the observed data.
posterior_params_all <- dplyr::bind_rows(
  lapply(results_list, function(x) x$posterior_params)
) %>%
  dplyr::left_join(prior_grid, by = "prior_name") %>%
  dplyr::mutate(
    analysis_label = analysis_label
  )

# Extract data-level sufficient statistics from all runs.
# Since the same data are reused across priors, many rows are duplicates.
# distinct() keeps only the unique case-level statistics.
stats_per_case <- dplyr::bind_rows(
  lapply(results_list, function(x) x$stats)
) %>%
  dplyr::distinct()

# Combine diagnostic quantities from all runs.
# These diagnostics are important because net-profit-failure draws are assigned psi(u) = 1.
# The diagnostics therefore help explain why some priors imply higher ruin probabilities.
diagnostics_all <- dplyr::bind_rows(
  lapply(results_list, function(x) x$diagnostics)
) %>%
  dplyr::left_join(prior_grid, by = "prior_name") %>%
  dplyr::mutate(
    analysis_label = analysis_label
  )

saveRDS(posterior_draws_all, 
        file.path(draws_dir, "posterior_draws_all_cases_prior_sensitivity.rds"))

write.csv(prior_table_all,
          file.path(tables_dir, "prior_table_prior_sensitivity.csv"),
          row.names = FALSE)

write.csv(posterior_params_all,
          file.path(tables_dir, "posterior_gamma_parameters_prior_sensitivity.csv"),
          row.names = FALSE)

write.csv(diagnostics_all,
          file.path(tables_dir, "net_profit_posterior_diagnostics_prior_sensitivity.csv"),
          row.names = FALSE)

# ============================================================
# 8. POSTERIOR SUMMARY TABLES
# ============================================================
# We summarize both the model parameters and the ruin quantities.
#
# psi: Unconditional posterior ruin probability.
#      This includes psi = 1 when the posterior draw violates the net profit condition.
#
# psi_given_net_profit_holds: Conditional summary of the continuous part of the posterior distribution,
#                             using only draws where the net profit condition holds.

summary_vars <- c("lambda", "beta", "R_raw", "psi", "psi_given_net_profit_holds")

# Create a long posterior summary table.
# For each variable, we summarise posterior draws separately by Case and prior.
# .data[[variable_i]] is used because the variable name is stored as a string.
posterior_summary_long <- dplyr::bind_rows(
  lapply(summary_vars, function(variable_i) {
    posterior_draws_all %>%
      dplyr::group_by(
        Case,
        prior_name,
        prior_display_name,
        prior_family,
        prior_interpretation,
        lambda_multiplier,
        beta_multiplier,
        lambda_concentration,
        beta_concentration
      ) %>%
      dplyr::summarise(
        variable = variable_i,
        n_draws = dplyr::n(),
        n_non_missing = sum(!is.na(.data[[variable_i]])), # n_non_missing is important for psi_given_net_profit_holds because draws where the net profit condition fails are stored as NA.
        mean = if (sum(!is.na(.data[[variable_i]])) == 0) NA_real_ else mean(.data[[variable_i]], na.rm = TRUE),
        median = if (sum(!is.na(.data[[variable_i]])) == 0) NA_real_ else stats::median(.data[[variable_i]], na.rm = TRUE),
        sd = if (sum(!is.na(.data[[variable_i]])) == 0) NA_real_ else stats::sd(.data[[variable_i]], na.rm = TRUE),
        q025 = if (sum(!is.na(.data[[variable_i]])) == 0) NA_real_ else as.numeric(stats::quantile(.data[[variable_i]], 0.025, na.rm = TRUE)),
        q975 = if (sum(!is.na(.data[[variable_i]])) == 0) NA_real_ else as.numeric(stats::quantile(.data[[variable_i]], 0.975, na.rm = TRUE)),
        ci_width = q975 - q025,
        .groups = "drop"
      )
  })
)


true_values_long <- dplyr::bind_rows(
  data.frame(Case = true_parameters$Case, variable = "lambda", true_value = true_parameters$lambda_true),
  data.frame(Case = true_parameters$Case, variable = "beta", true_value = true_parameters$beta_true),
  data.frame(Case = true_parameters$Case, variable = "R_raw", true_value = true_parameters$R_raw_true),
  data.frame(Case = true_parameters$Case, variable = "psi", true_value = true_parameters$psi_true_ultimate),
  data.frame(Case = true_parameters$Case, variable = "psi_given_net_profit_holds", true_value = true_parameters$psi_true_ultimate)
)

# Add the true simulation values so posterior summaries can be compared against the known data-generating parameters.
# Compute posterior errors relative to the true simulation values.
posterior_summary_long <- posterior_summary_long %>%
  dplyr::left_join(true_values_long, by = c("Case", "variable")) %>%
  dplyr::mutate(
    error_mean_minus_true = mean - true_value,
    error_median_minus_true = median - true_value,
    abs_error_median = abs(error_median_minus_true),
    analysis_label = analysis_label
  )

write.csv(posterior_summary_long,
          file.path(tables_dir, "posterior_summary_long_prior_sensitivity.csv"),
          row.names = FALSE)


# ============================================================
# 9. RUIN PROBABILITY SUMMARY TABLE
# ============================================================

# Extract the unconditional posterior summary for psi(u).
# This is the main posterior ruin probability, including psi = 1 for draws where the net profit condition fails.
psi_unconditional_summary <- posterior_summary_long %>%
  dplyr::filter(variable == "psi") %>%
  dplyr::select(Case,
                prior_name,
                prior_display_name,
                prior_family,
                prior_interpretation,
                lambda_multiplier,
                beta_multiplier,
                lambda_concentration,
                beta_concentration,
                true_psi = true_value,
                posterior_mean_psi_unconditional = mean,
                posterior_median_psi_unconditional = median,
                psi_unconditional_ci_lower = q025,
                psi_unconditional_ci_upper = q975,
                psi_unconditional_ci_width = ci_width,
                psi_unconditional_abs_error_median = abs_error_median)

# Extract the conditional summary for psi(u) given that the net profit condition holds.
# This separates the continuous part of the posterior from the point mass at psi = 1.
psi_conditional_summary <- posterior_summary_long %>%
  dplyr::filter(variable == "psi_given_net_profit_holds") %>%
  dplyr::select(Case,
                prior_name,
                posterior_mean_psi_given_net_profit_holds = mean,
                posterior_median_psi_given_net_profit_holds = median,
                psi_given_net_profit_holds_ci_lower = q025,
                psi_given_net_profit_holds_ci_upper = q975,
                psi_given_net_profit_holds_ci_width = ci_width,
                psi_given_net_profit_holds_abs_error_median = abs_error_median)

# Compute practical posterior risk measures (mainly for reporting in our thesis)
practical_risk_table <- posterior_draws_all %>%
  dplyr::group_by(Case, prior_name) %>%
  dplyr::summarise(posterior_pr_psi_gt_095_unconditional = mean(psi > 0.95),
                   posterior_pr_psi_gt_099_unconditional = mean(psi > 0.99),
                   .groups = "drop")

# Combine unconditional psi summaries, conditional psi summaries, net profit diagnostics, and practical high-risk probabilities
# into one ruin-probability-focused table.
ruin_probability_summary_table <- psi_unconditional_summary %>%
  dplyr::left_join(psi_conditional_summary, by = c("Case", "prior_name")) %>%
  dplyr::left_join(diagnostics_all %>%
                     dplyr::select(Case,
                                   prior_name,
                                   posterior_pr_net_profit_holds,
                                   posterior_pr_net_profit_fails,
                                   posterior_pr_psi_equals_one),
                   by = c("Case", "prior_name")) %>%
  dplyr::left_join(practical_risk_table, by = c("Case", "prior_name")) %>%
  dplyr::mutate(analysis_label = analysis_label)

write.csv(ruin_probability_summary_table,
          file.path(tables_dir, "ruin_probability_summary_prior_sensitivity.csv"),
          row.names = FALSE)

# ============================================================
# 10. SENSITIVITY AGAINST BASELINE PRIOR
# ============================================================
# Here we measure sensitivity relative to the truth-centered baseline prior.
# This gives a simple answer to RQ2: how much do posterior ruin summaries move when prior assumptions are changed?

# Extract the baseline prior results separately for each case.
# The baseline is the truth-centered prior with concentration s = 4.
baseline_by_case <- ruin_probability_summary_table %>%
  dplyr::filter(prior_name == baseline_prior_name) %>%
  dplyr::select(Case,
                baseline_mean_psi_unconditional = posterior_mean_psi_unconditional,
                baseline_median_psi_unconditional = posterior_median_psi_unconditional,
                baseline_psi_unconditional_ci_width = psi_unconditional_ci_width,
                baseline_pr_net_profit_fails = posterior_pr_net_profit_fails,
                baseline_pr_psi_equals_one = posterior_pr_psi_equals_one)

# Attach the baseline values to every prior result within the same case.
# This allows each prior to be compared against the correct case-specific baseline.
prior_sensitivity_vs_baseline <- ruin_probability_summary_table %>%
  dplyr::left_join(baseline_by_case, by = "Case") %>%
  dplyr::mutate(diff_mean_psi_vs_baseline = posterior_mean_psi_unconditional - baseline_mean_psi_unconditional,
                diff_median_psi_vs_baseline = posterior_median_psi_unconditional - baseline_median_psi_unconditional,
                abs_diff_median_psi_vs_baseline = abs(diff_median_psi_vs_baseline),
                diff_ci_width_vs_baseline = psi_unconditional_ci_width - baseline_psi_unconditional_ci_width,
                diff_pr_net_profit_fails_vs_baseline = posterior_pr_net_profit_fails - baseline_pr_net_profit_fails,
                diff_pr_psi_equals_one_vs_baseline = posterior_pr_psi_equals_one - baseline_pr_psi_equals_one)

write.csv(prior_sensitivity_vs_baseline,
          file.path(tables_dir, "prior_sensitivity_vs_baseline.csv"),
          row.names = FALSE)

# ============================================================
# 11. THESIS-FRIENDLY COMPACT TABLE
# ============================================================

# This is the table most likely to be used directly in our thesis.
# It focuses on the ruin probability and the most important diagnostics.

thesis_prior_sensitivity_table <- prior_sensitivity_vs_baseline %>%
  dplyr::select(Case,
                prior_display_name,
                prior_family,
                lambda_multiplier,
                beta_multiplier,
                lambda_concentration,
                beta_concentration,
                true_psi,
                posterior_median_psi_unconditional,
                psi_unconditional_ci_lower,
                psi_unconditional_ci_upper,
                posterior_pr_net_profit_fails,
                posterior_pr_psi_equals_one,
                posterior_pr_psi_gt_095_unconditional,
                abs_diff_median_psi_vs_baseline,
                diff_pr_net_profit_fails_vs_baseline)

write.csv(thesis_prior_sensitivity_table,
          file.path(tables_dir, "thesis_prior_sensitivity_table.csv"),
          row.names = FALSE)

# ============================================================
# 12. FIGURES
# ============================================================
# Display labels used only for plotting.
case_label_map <- c( "sparse_information" = "Sparse information",
                     "rich_information" = "Rich information")

# Convert prior labels to a factor to control their order on the y-axis.
# rev() is used so the displayed order matches the intended top-to-bottom order.
plot_ruin_summary <- prior_sensitivity_vs_baseline %>%
  dplyr::mutate(Case_label = factor( unname(case_label_map[Case]),
                                     levels = c("Sparse information", "Rich information")),
                prior_display_name = factor(prior_display_name,
                                            levels = rev(prior_grid$prior_display_name))) 

# Prepare true psi values with the same case labels as the plotting data.
# This is used to draw the dashed reference line in each facet.    
true_psi_plot <- true_parameters %>%
  dplyr::mutate(Case_label = factor(unname(case_label_map[Case]),
                                    levels = c("Sparse information", "Rich information")))

# ------------------------------------------------------------
# Figure 1: Posterior credible intervals for psi(u)
# ------------------------------------------------------------
# Base plot for posterior ruin probability summaries.
# Each prior is placed on the y-axis, while the x-axis shows posterior psi(u).
# The point is the posterior median and xmin/xmax define the 95% credible interval.
p_prior_interval <- ggplot(
  plot_ruin_summary,
  aes(y = prior_display_name,
      x = posterior_median_psi_unconditional,
      xmin = psi_unconditional_ci_lower,
      xmax = psi_unconditional_ci_upper
  )) +
  geom_vline(
    data = true_psi_plot,
    aes(xintercept = psi_true_ultimate),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 0.55
  ) +
  geom_pointrange(linewidth = 0.45) + # Draw the posterior median and its 95% credible interval for each prior.
  facet_wrap(~ Case_label, ncol = 1) + # Show sparse and rich information scenarios in separate vertical panels.
  coord_cartesian(xlim = c(0, 1)) +
  labs(title = "Prior sensitivity of posterior ruin probability",
       subtitle = "Points: medians. Intervals: 95% CrI. Dashed lines: true values.",
       x = expression(psi(u)),
       y = "Prior assumption"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        strip.text = element_text(face = "bold"),
        panel.grid.minor = element_blank())
    

ggsave(file.path(figures_dir, "prior_sensitivity_posterior_psi_intervals.png"),
       plot = p_prior_interval,
       width = 9,
       height = 7,
       dpi = 300)

ggsave(file.path(figures_dir, "prior_sensitivity_posterior_psi_intervals.pdf"),
       plot = p_prior_interval,
       width = 9,
       height = 7)

# ------------------------------------------------------------
# Figure 2: Posterior probability that net profit condition fails
# ------------------------------------------------------------

# This figure shows how much posterior probability each prior assigns to the event that the net profit condition fails.
#
## Important:
#   These are unconditional posterior diagnostics.
#   If the net profit condition fails for a posterior draw, the corresponding ultimate ruin probability is set to psi(u) = 1.
#
# We colour the bars by prior direction:
#   - Truth-centered: prior centered at the true simulation parameters.
#   - Pessimistic: prior expects fewer arrivals and smaller gain sizes.
#   - Optimistic: prior expects more arrivals and larger gain sizes.
#
# This makes it easier to see that pessimistic priors tend to increase the
# posterior probability of net profit failure, while optimistic priors reduce it.

# Classify each prior into a broader prior direction.
# This is used only for colouring the bars in the figure.
# grepl("^Pessimistic", ...) checks whether the prior label starts with "Pessimistic".
plot_net_profit_fails <- plot_ruin_summary %>%
  dplyr::mutate(prior_direction = dplyr::case_when(grepl("^Pessimistic", as.character(prior_display_name)) ~ "Pessimistic",
                                                   grepl("^Optimistic", as.character(prior_display_name)) ~ "Optimistic",
                                                   TRUE ~ "Truth-centered"),
                prior_direction = factor(prior_direction,
                                         levels = c("Truth-centered", "Pessimistic", "Optimistic")))

prior_direction_colours <- c("Truth-centered" = "#0072B2",
                             "Pessimistic"    = "#D55E00",
                             "Optimistic"     = "#009E73")
  

p_net_profit_fails <- ggplot(plot_net_profit_fails,
                             aes(y = prior_display_name,
                                 x = posterior_pr_net_profit_fails,
                                 fill = prior_direction)
                             ) +
  geom_col(width = 0.65, colour = "grey30", linewidth = 0.15) + # Draw bars using the already-computed posterior probabilities. geom_col() is appropriate because the bar heights are stored in the data.
  facet_wrap(~ Case_label, ncol = 1) +
  coord_cartesian(xlim = c(0, 1)) +
  scale_fill_manual(values = prior_direction_colours) +
  labs(title = "Posterior probability that the net profit condition fails",
       subtitle = "Net-profit-failure draws are assigned psi(u) = 1.",
       x = "Posterior probability",
       y = "Prior assumption",
       fill = "Prior type"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(figures_dir, "prior_sensitivity_net_profit_failure_probability.png"),
       plot = p_net_profit_fails,
       width = 9,
       height = 7,
       dpi = 300)
  
ggsave(file.path(figures_dir, "prior_sensitivity_net_profit_failure_probability.pdf"),
       plot = p_net_profit_fails,
       width = 9,
       height = 7)

# ------------------------------------------------------------
# Figure 3: Sensitivity measured as distance from baseline prior
# ------------------------------------------------------------
# This figure measures how much the posterior median ruin probability moves relative to the baseline prior.
#
# Important:
#   The baseline prior is not treated as the truth. It is only a reference prior.
#   The plotted quantity is:
#     | posterior median psi(u) under prior j - posterior median psi(u) under the baseline prior |
# 
#   The posterior median is based on the unconditional posterior psi(u).
#   This means that all posterior draws are included, including draws where the net profit condition fails and psi(u) is set to 1.

plot_sensitivity_baseline <- plot_ruin_summary %>%
  dplyr::mutate(prior_direction = dplyr::case_when(grepl("^Pessimistic", as.character(prior_display_name)) ~ "Pessimistic",
                                                   grepl("^Optimistic", as.character(prior_display_name)) ~ "Optimistic",
                                                   TRUE ~ "Truth-centered"),
                prior_direction = factor(prior_direction,
                                         levels = c("Truth-centered", "Pessimistic", "Optimistic")))

# Use the same colour logic as in the net-profit-failure figure.
prior_direction_colours <- c("Truth-centered" = "#0072B2",
                             "Pessimistic"    = "#D55E00",
                             "Optimistic"     = "#009E73")

# Keep the x-axis range interpretable: at least 0.10 to avoid over-zooming small changes, and at most 1 because psi is a probability.
x_upper_sensitivity <- max(plot_sensitivity_baseline$abs_diff_median_psi_vs_baseline, na.rm = TRUE) * 1.08
x_upper_sensitivity <- min(1, max(0.10, x_upper_sensitivity))

p_sensitivity_baseline <- ggplot(
  plot_sensitivity_baseline,
  aes(y = prior_display_name,
      x = abs_diff_median_psi_vs_baseline,
      fill = prior_direction
  )) +
  geom_col(width = 0.65, colour = "grey30", linewidth = 0.15) +
  facet_wrap(~ Case_label, ncol = 1) +
  coord_cartesian(xlim = c(0, x_upper_sensitivity)) +
  scale_fill_manual(values = prior_direction_colours) +
  labs(title = "Median change vs baseline prior",
       subtitle = "Baseline: truth-centered Gamma prior, s = 4.",
       x = expression(abs(Delta~median~psi(u))),
       y = "Prior assumption",
       fill = "Prior type"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(figures_dir, "prior_sensitivity_absolute_median_change_vs_baseline.png"),
       plot = p_sensitivity_baseline,
       width = 9,
       height = 7,
       dpi = 300)

ggsave(file.path(figures_dir, "prior_sensitivity_absolute_median_change_vs_baseline.pdf"),
       plot = p_sensitivity_baseline,
       width = 9,
       height = 7)

# ------------------------------------------------------------
# Figure 4: Conditional posterior density of psi(u) for selected priors
# ------------------------------------------------------------
# This plot excludes draws where the net profit condition fails. It therefore
# visualizes the continuous part of the posterior distribution of psi(u).
# The point mass at psi(u) = 1 is summarized separately in the tables and in
# the net-profit-failure figure above.

selected_density_priors <- c("truth_diffuse",
                             "truth_centered_baseline",
                             "pessimistic_strong",
                             "optimistic_strong")

selected_density_prior_labels <- prior_grid$prior_display_name[match(selected_density_priors, prior_grid$prior_name)]

selected_prior_colours <- c("Truth-centered, s = 1" = "#1b9e77",
                            "Truth-centered, s = 4" = "#377eb8",
                            "Pessimistic, s = 10" = "#d95f02",
                            "Optimistic, s = 10" = "#7570b3")

selected_prior_linetypes <- c("Truth-centered, s = 1" = "solid",
                              "Truth-centered, s = 4" = "dashed",
                              "Pessimistic, s = 10" = "dotdash",
                              "Optimistic, s = 10" = "twodash")

# Keep only the selected priors and only posterior draws where the net profit condition holds.
# In dplyr::filter(), net_profit_holds is equivalent to net_profit_holds == TRUE.
plot_density_data <- posterior_draws_all %>%
  dplyr::filter(prior_name %in% selected_density_priors,
                net_profit_holds
                ) %>%
  dplyr::mutate(
    Case_label = factor(unname(case_label_map[Case]),
                        levels = c("Sparse information", "Rich information")),
    prior_display_name = factor(prior_display_name,
                                levels = selected_density_prior_labels))

true_line_data <- true_parameters %>%
  dplyr::mutate(
    Case_label = factor(unname(case_label_map[Case]),
                        levels = c("Sparse information", "Rich information"))
  ) %>%
  dplyr::select(Case, Case_label, psi_true_ultimate)

p_conditional_density <- ggplot(
  plot_density_data,
  aes(x = psi_given_net_profit_holds,
      colour = prior_display_name,
      linetype = prior_display_name)
  ) +
  geom_vline(data = true_line_data,
             aes(xintercept = psi_true_ultimate),
             inherit.aes = FALSE,
             linetype = "longdash",
             linewidth = 0.8,
             colour = "black",
             alpha = 0.9
  ) +
  geom_density(linewidth = 1.05, na.rm = TRUE) +
  facet_wrap(~ Case_label, ncol = 1, scales = "free_y") +
  coord_cartesian(xlim = c(0, 1)) +
  scale_colour_manual(values = selected_prior_colours) +
  scale_linetype_manual(values = selected_prior_linetypes) +
  labs(title = "Conditional posterior density of ultimate ruin probability for selected priors",
       subtitle = "Conditional on net profit holding. Dashed line shows true psi.",
       x = expression(psi(u)),
       y = "Density",
       colour = "Prior",
       linetype = "Prior"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

ggsave(file.path(figures_dir, "prior_sensitivity_conditional_psi_density_selected_priors.png"),
       plot = p_conditional_density,
       width = 9,
       height = 6,
       dpi = 300)

ggsave(file.path(figures_dir, "prior_sensitivity_conditional_psi_density_selected_priors.pdf"),
       plot = p_conditional_density,
       width = 9,
       height = 6)

# ============================================================
# 13. FINAL MESSAGE
# ============================================================

cat("\nPrior sensitivity analysis completed.\n")
cat("Posterior draws per Case x prior:", S, "\n")
cat("Number of cases:", length(case_names), "\n")
cat("Number of priors:", nrow(prior_grid), "\n")
cat("Total posterior draws:", nrow(posterior_draws_all), "\n")
cat("Tables saved in:", normalizePath(tables_dir, winslash = "/"), "\n")
cat("Draws saved in:", normalizePath(draws_dir, winslash = "/"), "\n")
cat("Figures saved in:", normalizePath(figures_dir, winslash = "/"), "\n")

# ============================================================
# End of 04_prior_sensitivity.R
# ============================================================
