# ============================================================
# 05_model_misspecification.R
# Gain-size misspecification analysis for Bayesian calibration in the classical continuous-time dual risk model
# ============================================================
#
# Research question 3:
#   How sensitive is the posterior distribution of ultimate ruin probability
#   to gain-size model misspecification?
#
# Main idea:
#   - Use the same simulated data as the baseline analysis.
#   - Keep the same arrival process, u, and c.
#   - Replace the exponential gain-size model with a log-normal gain-size model.
#   - Compare the misspecified log-normal results against the exponential baseline.
#
# Model:
#   U(t) = u - c*t + sum_{i=1}^{N(t)} X_i
#
# Misspecified gain-size model:
#   X_i ~ Lognormal(mu, sigma^2)
#   Y_i = log(X_i) ~ Normal(mu, sigma^2)
# ============================================================

# ============================================================
# 0. PACKAGES
# ============================================================
library(ggplot2)
library(dplyr)
library(posterior)
library(tidyr)

# ============================================================
# 1. SOURCE FUNCTIONS
# ============================================================
source("02_functions_conditional_psi.R")
source("02_functions_lognormal_misspecification.R")

# ============================================================
# 2. USER SETTINGS
# ============================================================
seed <- 999
set.seed(seed)
S <- 10000

# Same constants as in the baseline analysis.
u <- 207.28
c_rate <- 91.65 / 12

case_sparse <- "sparse_information"
case_rich <- "rich_information"
case_names <- c(case_sparse, case_rich)

simulation_dir_name <- "simulated_dual_risk_data"
baseline_results_dir_name <- "results"
misspec_results_dir_name <- "result_model_misspecification"

analysis_label <- "gain_size_lognormal_misspecification"

lambda_concentration <- 4 # Prior concentration for the gain-arrival intensity lambda.
kappa0 <- 4 # Prior strength for the lognormal mean parameter mu in the normal-inverse-chi-square prior.
nu0 <- 4 # Prior degrees-of-freedom-like parameter controlling prior information about sigma^2 in the normal-inverse-chi-square prior.

# ============================================================
# 3. INPUT AND OUTPUT FOLDERS
# ============================================================
# Finds the name of the current working directory.
current_folder <- basename(normalizePath(getwd(), winslash = "/", mustWork = TRUE))

# If the script is run from inside the simulation-data folder, use ".".
# Otherwise, look for the simulation-data folder from the current directory.
input_dir <- if (identical(current_folder, simulation_dir_name)) "." else simulation_dir_name

# If the script is run from inside the misspecification result folder, save outputs in the current folder.
# Otherwise, create/use the result_model_misspecification folder.
misspec_results_dir <- if (identical(current_folder, misspec_results_dir_name)) {
  "."
} else {
  misspec_results_dir_name
}

tables_dir <- file.path(misspec_results_dir, "tables") # Subfolder for CSV tables.
draws_dir <- file.path(misspec_results_dir, "draws") # Subfolder for posterior draw objects saved as RDS files.
figures_dir <- file.path(misspec_results_dir, "figures") # Subfolder for generated figures.

# Creates the output folders if they do not already exist.
# recursive = TRUE allows nested folders to be created.
# showWarnings = FALSE avoids warnings when folders already exist.
dir.create(misspec_results_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(draws_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

cat("Reading simulation files from:", normalizePath(input_dir, winslash = "/"), "\n")
cat("Saving model misspecification results in:", normalizePath(misspec_results_dir, winslash = "/"), "\n")

###### Some defensive coding tips from AI ######
# Possible locations of the saved exponential baseline posterior draws.
# Multiple candidate paths are used so the script can work from different working directories.
baseline_draws_candidates <- c(
  file.path(baseline_results_dir_name, "draws", "posterior_draws_all_cases_baseline.rds"),
  file.path("..", baseline_results_dir_name, "draws", "posterior_draws_all_cases_baseline.rds"),
  file.path("draws", "posterior_draws_all_cases_baseline.rds")
)
baseline_draws_path <- baseline_draws_candidates[file.exists(baseline_draws_candidates)][1] # Checks which candidate paths actually exist and selects the first valid one.

if (is.na(baseline_draws_path)) {
  stop("Could not find posterior_draws_all_cases_baseline.rds. ",
       "Run 03_run_results_conditional_psi.R first and check the results/draws folder.")
  }

# ============================================================
# 4. READ SIMULATION DATA AND BASELINE DRAWS
# ============================================================
read_case_files <- function(case_name) {
  # Reads the event file and summary file for one simulation case.
  # The case_name is used to construct the correct file names.
  
  list(events = read.csv(file.path(input_dir, paste0("events_", case_name, ".csv")), # Event-level data: one row per observed gain event.
                         stringsAsFactors = FALSE),
       summary = read.csv(file.path(input_dir, paste0("summary_", case_name, ".csv")), # Case-level summary data, including the observation horizon.
                          stringsAsFactors = FALSE))
  }

# Empty list used to store the data for all cases.
case_data <- list()

# Loops over sparse and rich cases and stores each case as a named list element.
for (case_name in case_names) {
  case_data[[case_name]] <- read_case_files(case_name)
}

# Reads the true simulation parameters.
true_parameters <- read.csv(file.path(input_dir, "true_parameters_two_cases.csv"),
                            stringsAsFactors = FALSE)

# Reads the saved posterior draws from the correctly specified exponential model and labels them as the exponential baseline.
baseline_draws_all <- readRDS(baseline_draws_path) %>% dplyr::mutate(model = "Exponential baseline")

# ============================================================
# 5. RUN LOGNORMAL MISSPECIFICATION ANALYSIS
# ============================================================
# Empty list used to store the full lognormal misspecification result for each case.
results_list <- list()

for (case_name in case_names) { # Runs the lognormal misspecification analysis separately for each case.
  
  true_i <- true_parameters[true_parameters$Case == case_name, , drop = FALSE] # Extracts the row of true simulation parameters for the current case.

  if (nrow(true_i) != 1) {stop("Expected exactly one row in true_parameters for case: ", case_name)}

  cat("Running lognormal misspecification case:", case_name, "\n")

  # Runs the full Bayesian lognormal misspecification analysis for this case.
  # The function extracts log-gain statistics, constructs the prior,
  # computes posterior parameters, simulates posterior draws, and computes
  # ruin-related quantities for the lognormal gain-size model.
  # Details in 02_functions_lognormal_misspecification.R
  results_list[[case_name]] <- run_lognormal_misspecification_case(
    events = case_data[[case_name]]$events,
    summary = case_data[[case_name]]$summary,
    true_row = true_i,
    u = u,
    c_rate = c_rate,
    S = S,
    case_name = case_name,
    lambda_concentration = lambda_concentration,
    kappa0 = kappa0,
    nu0 = nu0)
  }

# Extracts posterior draws from each case result and combines them into one data frame.
# The model label and analysis label are added so the draws can later be compared with the exponential baseline and traced back to this analysis.
posterior_draws_lognormal_all <- dplyr::bind_rows(lapply(results_list, function(x) x$draws)
                                                  ) %>%
  dplyr::mutate(model = "Lognormal misspecified",
                analysis_label = analysis_label)

saveRDS(posterior_draws_lognormal_all,
        file.path(draws_dir, "posterior_draws_all_cases_lognormal_misspecification.rds"))

for (case_name in case_names) {
  saveRDS(results_list[[case_name]]$draws,
          file.path(draws_dir, paste0("posterior_draws_", case_name, "_lognormal_misspecification.rds")))
  }

# ============================================================
# 6. TABLE HELPERS
# ============================================================
safe_mean <- function(x) {
  # Computes the mean while safely handling all-NA vectors.
  # If all values are NA, the function returns NA instead of producing
  # a warning or an invalid summary.
  if (all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE) # Otherwise, compute the mean after removing missing values.
}

safe_median <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::median(x, na.rm = TRUE) # Compute the median after removing missing values.
}

safe_sd <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  stats::sd(x, na.rm = TRUE) # Compute the standard deviation after removing missing values.

}

# Computes the 2.5% posterior quantile while safely handling all-NA vectors.
# This is used as the lower endpoint of a 95% credible interval.
q025 <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  unname(stats::quantile(x, 0.025, na.rm = TRUE))
}

# Computes the 97.5% posterior quantile while safely handling all-NA vectors.
# This is used as the upper endpoint of a 95% credible interval.
q975 <- function(x) {
  if (all(is.na(x))) return(NA_real_)
  unname(stats::quantile(x, 0.975, na.rm = TRUE))
}

#### Summarise posterior draws for selected variables ####
#
## Purpose:
#   This function creates a posterior summary table for selected variables.
#   It is written in long format, meaning that each row corresponds to one
#   Case x variable combination.
#
## Arguments:
#   - draws: A data frame containing posterior draws. It must include at least: case, prior_name, the variables listed in the argument 'variables'
#
#   - variables: A character vector with the names of the posterior variables to summarise, for example c("lambda", "mu", "sigma", "psi").
#
#   - model_name: A text label describing which model the draws come from, for example "Lognormal misspecified".
#
## Output:
#   A long-format data frame containing posterior summaries:
#   mean, median, standard deviation, 95% credible interval endpoints,
#   and credible interval width.
summarise_draws_long <- function(draws,
                                 variables,
                                 model_name = NA_character_) {
  # First split the posterior draws by Case.
  # This creates a list where each element contains the draws for one case.
  out <- dplyr::bind_rows(
    lapply(split(draws, draws$Case), function(draws_i) {
      
      # For the current case, loop over all variables that should be summarised.
      # Each variable produces one row in the output table.
      dplyr::bind_rows(
        lapply(variables, function(v) {
          
          # Extract the posterior draws for the current variable. For example, if v = "psi", then draws_i[[v]] means draws_i[["psi"]].
          x <- draws_i[[v]]
          
          # Create one summary row for the current Case x variable combination.
          data.frame(
            Case = unique(draws_i$Case),
            prior_name = unique(draws_i$prior_name),
            model = model_name,
            variable = v,
            mean = safe_mean(x),
            median = safe_median(x),
            sd = safe_sd(x),
            q025 = q025(x),
            q975 = q975(x),
            ci_width = q975(x) - q025(x),
            stringsAsFactors = FALSE
          )
        })
      )
    })
  )
  out
}

get_summary_value <- function(summary_table,
                              case_name,
                              variable_name,
                              statistic_name) {
  # Extracts one specific statistic from a posterior summary table.
  #
  # Example: get_summary_value(table, "sparse_information", "psi", "median")
  # returns the posterior median of psi for the sparse case.
  
  # Identifies the row matching the requested case and variable.
  row_id <- summary_table$Case == case_name & summary_table$variable == variable_name

  # Extracts the requested statistic from that row.
  as.numeric(summary_table[[statistic_name]][row_id][1])
}

case_label_map <- c("sparse_information" = "Sparse information",
                    "rich_information" = "Rich information")

## Converts internal case names into readable labels for figures.
make_case_label <- function(x) {
  out <- unname(case_label_map[x]) # Looks up each input value in the named vector case_label_map.
  
  # If no label exists, return the original input. Otherwise, return the readable label.
  ifelse(is.na(out), x, out)
}

# Defines the desired order of case labels in plots.
# This ensures that sparse information appears before rich information.
case_levels <- c("Sparse information", "Rich information")

# Creates a consistent minimal theme for thesis figures (Using one shared theme).
theme_thesis_figure <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(
        face = "bold",
        size = 12,
        margin = margin(b = 4)
      ),
      plot.subtitle = element_text(
        size = 9,
        margin = margin(b = 8)
      ),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      strip.text = element_text(face = "bold", size = 10),
      legend.position = "bottom",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 9),
      panel.grid.minor = element_blank()
    )
}

save_plot_both <- function(plot, filename_base, width = 8, height = 5) {
  ggplot2::ggsave(file.path(figures_dir, paste0(filename_base, ".png")),
                  plot = plot,
                  width = width,
                  height = height,
                  dpi = 300)

  ggplot2::ggsave(file.path(figures_dir, paste0(filename_base, ".pdf")),
                  plot = plot,
                  width = width,
                  height = height)}

# ============================================================
# 7. OUTPUT TABLES
# ============================================================
prior_lognormal_table <- dplyr::bind_rows(
  lapply(names(results_list), function(case_name) { # Loop over the names of all cases stored in results_list.
    lognormal_misspec_prior_to_table(
      # Convert the prior specification for the current case into a table row.
      # The prior itself is stored inside results_list[[case_name]]$prior_specs.
      prior_specs = results_list[[case_name]]$prior_specs,
      case_name = case_name
    )
  })
)

write.csv(prior_lognormal_table,
          file.path(tables_dir, "prior_lognormal_misspecification.csv"),
          row.names = FALSE)

# Extract posterior parameter summaries from each case result.
# Each element in results_list contains a component called posterior_params.
posterior_lognormal_parameters <- dplyr::bind_rows(
  lapply(results_list, function(x) x$posterior_params)
)

write.csv(posterior_lognormal_parameters,
          file.path(tables_dir, "posterior_lognormal_parameters.csv"),
          row.names = FALSE)

posterior_summary_lognormal <- summarise_draws_long(
  draws = posterior_draws_lognormal_all,   # Use all posterior draws from the lognormal misspecification model.
  variables = c("lambda",
                "mu",
                "sigma",
                "sigma2",
                "implied_mean_gain",
                "expected_gain_rate",
                "R_raw",
                "psi",
                "psi_given_net_profit_holds"),
  model_name = "Lognormal misspecified"
)

write.csv(posterior_summary_lognormal,
          file.path(tables_dir, "posterior_summary_lognormal_misspecification.csv"),
          row.names = FALSE)

# Extract diagnostic information from each case result.
net_profit_diagnostics_lognormal <- dplyr::bind_rows(
  lapply(results_list, function(x) x$diagnostics)
)

write.csv(
  net_profit_diagnostics_lognormal,
  file.path(tables_dir, "net_profit_diagnostics_lognormal_misspecification.csv"),
  row.names = FALSE
)

# transmate() creates a table with only thoses columns that we specified
npc_failure_lognormal_table <- net_profit_diagnostics_lognormal %>%
  dplyr::transmute(
    Case,
    posterior_pr_net_profit_holds,
    posterior_pr_net_profit_fails,
    posterior_pr_psi_equals_one,
    root_failure_rate_given_net_profit_holds
  )

write.csv(npc_failure_lognormal_table,
          file.path(tables_dir, "npc_failure_lognormal_misspecification.csv"),
          row.names = FALSE)

# Extract observed-data diagnostics from each case result.
# These statistics describe the observed gains and log-gains used by the lognormal misspecification model.
stats_lognormal <- dplyr::bind_rows(
  lapply(results_list, function(x) x$stats)
)

write.csv(stats_lognormal,
          file.path(tables_dir, "data_diagnostics_lognormal_misspecification.csv"),
          row.names = FALSE)

# ============================================================
# Figure: true exponential DGP vs fitted exponential and lognormal MLE
# ============================================================
#
## Purpose: 
#   This figure is a diagnostic for the gain-size model.
#   It compares three gain-size density curves:
#     1. the true exponential distribution used to simulate the data (DGP - Data generating process),
#     2. the exponential MLE fitted to the observed gains,
#     3. the lognormal MLE fitted to the observed gains.
#
# Important:
#   This figure is not used to compute posterior draws or ruin probabilities.
#   It is only used to visually assess how the fitted gain-size distributions
#   compare with the true data-generating distribution and the observed gains that we report in our thesis.
# ============================================================

# Add the true simulation parameters to the observed-data diagnostics.
# stats_lognormal contains fitted MLE values from the observed gains, while true_parameters contains the true exponential DGP parameters.
gain_fit_grid_wide <- stats_lognormal %>%
  dplyr::left_join(
    true_parameters %>%
      dplyr::select(Case, beta_true, mean_gain_true),
    by = "Case"
  ) %>%
  dplyr::rowwise() %>%  # Work one case at a time. Each case has its own true beta, fitted exponential beta and fitted lognormal parameters.
  dplyr::do({
    row_i <- . # Store the current one-row data frame as row_i.
    
    x_max_i <- stats::qexp(0.995, rate = row_i$beta_true)
    # Use the 99.5% quantile of the true exponential distribution as the maximum x-value.
    # This keeps the plot focused on the range where almost all relevant gain-size probability mass lies, instead of drawing a very
    # long and mostly empty right tail. Note that 0.995 is chosen so that the figure looks good in the thesis, so there are no deep reasons at all.
    
    x_grid <- seq(0.001, x_max_i, length.out = 1000)
    # Create a fine grid of gain-size values where the density curves are evaluated.
    # We start slightly above zero because the lognormal density is defined for x > 0.
    
    data.frame(Case = row_i$Case,
               x = x_grid,
               # Density of the true exponential distribution used in the simulation.
               # This is the data-generating process, not an estimated model.
               true_exponential_density = stats::dexp(x_grid,
                                                      rate = row_i$beta_true),
               
               # Density of the exponential distribution fitted to the observed gains using the MLE beta_hat = n / sum(x_i).
               fitted_exponential_mle_density = stats::dexp(x_grid, 
                                                            rate = row_i$beta_mle),
               
               # Density of the lognormal distribution fitted to the observed gains. 
               # The fitted parameters are based on log(gain_size): mu_mle = mean(log(x_i)) and sigma2_mle = MLE variance of log(x_i).
               fitted_lognormal_mle_density = stats::dlnorm(x_grid,
                                                            meanlog = row_i$mu_mle,
                                                            sdlog = sqrt(row_i$sigma2_mle))
    )
  }) %>%
  dplyr::ungroup() # Remove the rowwise() grouping after the case-specific density grids have been constructed.
         
# Stack the three density curves into one long-format data frame for ggplot.
gain_fit_grid <- dplyr::bind_rows(
  # First curve: the true exponential density used in the data-generating process.
  gain_fit_grid_wide %>%
    dplyr::transmute(
      Case,
      x,
      distribution = "True exponential DGP",
      density = true_exponential_density
    ),
  
  # Second curve: the exponential MLE fitted to the observed gains.
  gain_fit_grid_wide %>%
    dplyr::transmute(
      Case,
      x,
      distribution = "Fitted exponential MLE",
      density = fitted_exponential_mle_density
    ),
  
  # Third curve: the lognormal MLE fitted to the observed gains.
  gain_fit_grid_wide %>%
    dplyr::transmute(
      Case,
      x,
      distribution = "Fitted lognormal MLE",
      density = fitted_lognormal_mle_density
    )
) %>%
  dplyr::mutate(
    Case_label = factor(make_case_label(Case), levels = case_levels),
    distribution = factor(distribution,
                          levels = c("True exponential DGP",
                                     "Fitted exponential MLE",
                                     "Fitted lognormal MLE"))
    )
      
#############
observed_gains_plot <- dplyr::bind_rows(
  lapply(names(case_data), function(case_name) {
    case_data[[case_name]]$events %>%
      dplyr::transmute(Case = case_name,
                       gain_size = gain_size) # Observed gain sizes used for the rug marks in the plot.
    })) %>%
  dplyr::mutate(Case_label = factor(make_case_label(Case), levels = case_levels))
  
# Plot density curves over the gain-size grid.
# Both color and linetype identify which distribution each curve represents.
p_gain_distribution_fit <- ggplot(
  gain_fit_grid,
  aes(x = x,
      y = density,
      color = distribution,
      linetype = distribution)
  ) +
  geom_line(linewidth = 0.9, na.rm = TRUE) +
  geom_rug(data = observed_gains_plot, # Add rug marks for the observed gains at the bottom of each panel. This shows where the actual observed gain sizes are located.
           aes(x = gain_size),
           inherit.aes = FALSE,
           sides = "b",
           alpha = 0.35,
           color = "grey35"
           ) +
  facet_wrap(~ Case_label, ncol = 1, scales = "free_x") +
  scale_color_manual(
    values = c("True exponential DGP" = "black",
               "Fitted exponential MLE" = "#0072B2",
               "Fitted lognormal MLE" = "#D55E00")
    ) +
  scale_linetype_manual(
    values = c("True exponential DGP" = "solid",
               "Fitted exponential MLE" = "dashed",
               "Fitted lognormal MLE" = "dotdash")
    ) +
  labs(title = "Gain-size distribution fit",
       subtitle = "True exponential DGP compared with exponential and lognormal MLE fits",
       x = "Gain size",
       y = "Density",
       color = "Distribution",
       linetype = "Distribution"
  ) +
  theme_thesis_figure(base_size = 10)
      

save_plot_both(p_gain_distribution_fit,
               "gain_size_true_exponential_vs_mle_fits",
               width = 8,
               height = 5.4)

# ============================================================
# 8. COMPARISON AGAINST EXPONENTIAL BASELINE
# ============================================================
#
## Purpose:
# This section compares the misspecified lognormal gain-size model against the correctly specified exponential baseline model.
#
## Main comparison:
#   We compare posterior summaries of the ultimate ruin probability psi(u),
#   including posterior mean, median, 95% credible interval, and the posterior
#   probability that psi(u) equals 1.
#
## Important:
#   The full posterior distribution of psi is used. This means that draws where the net profit condition fails are included
#   with psi = 1.
#
## Interpretation:
#   Differences are computed as: lognormal misspecified result - exponential baseline result.
#
## Therefore:
#   A negative delta means that the lognormal misspecified model gives a lower posterior estimate than the exponential baseline.
# ============================================================
baseline_summary <- summarise_draws_long(draws = baseline_draws_all, # Posterior draws from the correctly specified exponential baseline model.
                                         variables = c("lambda", "beta", "R_raw", "psi", "psi_given_net_profit_holds"), # Variables to summarise from the exponential baseline.
                                         model_name = "Exponential baseline") # Label used in the output summary table.

baseline_psi_summary <- baseline_summary %>%
  dplyr::filter(variable == "psi") %>% # Keep only the posterior summary of the full ruin probability psi(u).
  dplyr::select(Case,
                baseline_mean_psi = mean,
                baseline_median_psi = median,
                baseline_psi_q025 = q025,
                baseline_psi_q975 = q975,
                baseline_psi_ci_width = ci_width)

lognormal_psi_summary <- posterior_summary_lognormal %>%
  dplyr::filter(variable == "psi") %>% # Keep only the posterior summary of psi(u) under the lognormal misspecified model.
  dplyr::select(Case,
                lognormal_mean_psi = mean,
                lognormal_median_psi = median,
                lognormal_psi_q025 = q025,
                lognormal_psi_q975 = q975,
                lognormal_psi_ci_width = ci_width)

baseline_diag <- baseline_draws_all %>%
  dplyr::group_by(Case) %>% # Compute diagnostics separately for each case.
  dplyr::summarise(baseline_pr_net_profit_holds = mean(net_profit_holds), # Posterior probability that the net profit condition holds under the exponential baseline model.
                   baseline_pr_net_profit_fails = mean(!net_profit_holds), # Posterior probability that the net profit condition fails.
                   baseline_pr_psi_equals_one = mean(psi == 1), # Posterior mass at psi(u) = 1. # In this thesis, psi = 1 mainly occurs when the net profit condition fails.
                   .groups = "drop") # Drop the grouping after summarising so the output is an ordinary data frame.

lognormal_diag <- net_profit_diagnostics_lognormal %>%
  dplyr::select(Case,
                lognormal_pr_net_profit_holds = posterior_pr_net_profit_holds,
                lognormal_pr_net_profit_fails = posterior_pr_net_profit_fails,
                lognormal_pr_psi_equals_one = posterior_pr_psi_equals_one,
                
                # Among draws where the net profit condition holds, this records the proportion where the numerical Lundberg root calculation failed.
                lognormal_root_failure_rate_given_net_profit_holds = root_failure_rate_given_net_profit_holds) 

model_misspecification_vs_baseline <- baseline_psi_summary %>%
  dplyr::left_join(lognormal_psi_summary, by = "Case") %>% # Add lognormal posterior summaries of psi to the baseline summary table.
  dplyr::left_join(baseline_diag, by = "Case") %>% # Add baseline diagnostics for the net profit condition and psi = 1.
  dplyr::left_join(lognormal_diag, by = "Case") %>% # Add lognormal diagnostics for the net profit condition, psi = 1 and numerical root-finding failures.
  dplyr::left_join(
    true_parameters %>% # Add the true simulation values.
      dplyr::select(Case, psi_true_ultimate, mean_gain_true),
    by = "Case"
  ) %>%
  dplyr::mutate(
    delta_mean_psi = lognormal_mean_psi - baseline_mean_psi,
    delta_median_psi = lognormal_median_psi - baseline_median_psi,
    absolute_delta_median_psi = abs(delta_median_psi),
    delta_ci_width_psi = lognormal_psi_ci_width - baseline_psi_ci_width,
    analysis_label = analysis_label,
    posterior_draws_S = S,
    seed = seed)

write.csv(model_misspecification_vs_baseline,
  file.path(tables_dir, "model_misspecification_vs_baseline.csv"),
  row.names = FALSE)

# Draw-level paired comparison by draw_id.
# This is mainly a Monte Carlo diagnostic for visualising the distribution of differences. 
# The main table above reports differences in posterior summaries.
delta_psi_draws <- posterior_draws_lognormal_all %>%
  dplyr::select(Case, draw_id, psi_lognormal = psi) %>%
  
  # Match each lognormal draw with the baseline draw that has the same Case and draw_id.
  # inner_join keeps only draw IDs that exist in both models.
  dplyr::inner_join(
    baseline_draws_all %>%
      dplyr::select(Case, draw_id, psi_exponential = psi),
    by = c("Case", "draw_id")
  ) %>%
  dplyr::mutate(delta_psi = psi_lognormal - psi_exponential)

saveRDS(delta_psi_draws,
        file.path(draws_dir, "delta_psi_draws_lognormal_minus_baseline.rds"))

# ============================================================
# 8b. IMPLIED MEAN GAIN COMPARISON
# ============================================================
#
## Purpose:
# The exponential and lognormal gain-size models use different parameters, so their posterior parameters cannot be compared directly.
# Instead, both models are transformed to the same interpretable quantity: the implied expected gain size E[X].
#
# Exponential baseline: X ~ Exponential(beta), so E[X] = 1 / beta.
# Lognormal misspecified model: X ~ Lognormal(mu, sigma^2), so E[X] = exp(mu + sigma^2 / 2).
#
# The resulting posterior draws can therefore be compared on the same scale.
# ============================================================

# Stack the posterior implied-mean-gain draws from the two models into one long-format comparison table
mean_gain_compare_draws <- dplyr::bind_rows(
  baseline_draws_all %>%
    dplyr::transmute(
      Case,
      draw_id,
      model = "Exponential baseline",
      implied_mean_gain = 1 / beta # Convert each posterior beta draw into an expected gain size. For an exponential distribution, E[X] = 1 / beta.
    ),
  posterior_draws_lognormal_all %>%
    dplyr::transmute(
      Case,
      draw_id,
      model = "Lognormal misspecified",
      implied_mean_gain = implied_mean_gain # E[X] = exp(mu + sigma^2 / 2)
    )
) %>%
  dplyr::left_join(
    true_parameters %>%
      dplyr::select(Case, mean_gain_true), # Add the true expected gain size for each simulation case.
    by = "Case"
  )

saveRDS(mean_gain_compare_draws,
        file.path(draws_dir, "implied_mean_gain_baseline_vs_lognormal_draws.rds"))

# Safely calculate a requested quantile of x. The argument p determines which quantile is returned.
q_safe <- function(x, p) {
  if (all(is.na(x))) return(NA_real_) # Return a numeric NA if the vector contains no valid observations.
  unname(stats::quantile(x, p, na.rm = TRUE)) # Calculate the requested quantile after removing missing values. unname() removes the automatic quantile label from the result.
}

mean_gain_compare_summary <- mean_gain_compare_draws %>%
  dplyr::group_by(Case, model) %>%
  
  # Create separate posterior summaries for each Case x Model combination.
  dplyr::summarise(mean_gain_true = dplyr::first(mean_gain_true), # Keep the true expected gain size for the current case. The value is constant within each Case x Model group, so first() is sufficient.
                   mean = mean(implied_mean_gain, na.rm = TRUE), # Posterior mean of the implied expected gain size.
                   median = stats::median(implied_mean_gain, na.rm = TRUE), # Posterior median of the implied expected gain size.
                   sd = stats::sd(implied_mean_gain, na.rm = TRUE), # Posterior standard deviation, measuring posterior spread.
                   q025 = q_safe(implied_mean_gain, 0.025), # Lower endpoint of the central 95% credible interval.
                   q25 = q_safe(implied_mean_gain, 0.25), # First posterior quartile.
                   q75 = q_safe(implied_mean_gain, 0.75),# Third posterior quartile.
                   q975 = q_safe(implied_mean_gain, 0.975), # Upper endpoint of the central 95% credible interval.
                   q99 = q_safe(implied_mean_gain, 0.99), # The 99th posterior percentile. # Only 1% of posterior draws lie above this value.
                   max = max(implied_mean_gain, na.rm = TRUE), # Largest simulated posterior value.
                   ci_width = q_safe(implied_mean_gain, 0.975) - q_safe(implied_mean_gain, 0.025), # Width of the central 95% credible interval. A larger width indicates greater posterior uncertainty in E[X].
                   pr_above_true_mean = mean(implied_mean_gain > mean_gain_true, na.rm = TRUE), # Posterior probability that the implied mean gain exceeds the true mean gain.
                   pr_above_2x_true_mean = mean(implied_mean_gain > 2 * mean_gain_true, na.rm = TRUE), # Posterior probability that the implied mean gain is more than twice the true expected gain size.
                   pr_above_5x_true_mean = mean(implied_mean_gain > 5 * mean_gain_true, na.rm = TRUE), # This helps identify extreme upper-tail behaviour under misspecification.
                   .groups = "drop") # Remove the Case x Model grouping after the summary has been created.

write.csv(mean_gain_compare_summary,
          file.path(tables_dir, "implied_mean_gain_baseline_vs_lognormal.csv"),
          row.names = FALSE)

# ============================================================
# 8c. HARD SANITY CHECKS FOR MODEL MISSPECIFICATION
# ============================================================
#
## Purpose:
# These checks verify that the stored posterior draws and derived quantities 
# are internally consistent with the mathematical definitions used in the code.
#
## The script stops immediately if any required condition is violated.
#
## Important:
#   Passing these checks does not prove that the statistical model is empirically
#   correct or realistic. The checks only verify that the implemented calculations
#   follow the intended model logic.
# ============================================================

# Stop execution and print a clear audit-specific error message.
assert_true <- function(condition, message) {
  if (!isTRUE(condition)) {stop(paste0("AUDIT FAILED: ", message), call. = FALSE)} # call. = FALSE suppresses the function-call information in the error output.
  }

max_abs_diff <- function(x, y) {
  # Identify positions where both vectors contain finite numeric values.
  # Missing and infinite values are excluded from the comparison.
  idx <- is.finite(x) & is.finite(y)
  if (!any(idx)) return(NA_real_) # Return numeric NA if there are no valid pairs to compare.
  max(abs(x[idx] - y[idx])) # Return the largest absolute difference between corresponding valid values.
}

max_rel_diff <- function(x, y) {
  idx <- is.finite(x) & is.finite(y)
  if (!any(idx)) return(NA_real_)
  max(abs(x[idx] - y[idx]) / pmax(1, abs(y[idx]))) # Return the largest scaled difference. The denominator is bounded below by 1 to avoid unstable relative errors when the reference value y is close to zero.
}

audit_rows <- list() # Empty list used to collect one summary row for each completed audit check.

add_audit_row <- function(check, value, tolerance = NA) {
  audit_rows[[length(audit_rows) + 1]] <<- data.frame( # <<- "superassignment" updates audit_rows outside the function.
    check = as.character(check), # Description of the audit check.
    value = as.character(value), # Observed result, for example "passed" or a numerical difference.
    tolerance = as.character(tolerance), # Numerical tolerance used by the check, when applicable.
    stringsAsFactors = FALSE # Keep text columns as character variables.
  )
}

# ------------------------------------------------------------
# 1. Basic draw structure
# ------------------------------------------------------------

assert_true(all(case_names %in% unique(posterior_draws_lognormal_all$Case)),
            "Not all cases are present in posterior_draws_lognormal_all.")

assert_true(all(case_names %in% unique(baseline_draws_all$Case)),
            "Not all cases are present in baseline_draws_all.")

lognormal_counts <- posterior_draws_lognormal_all %>% dplyr::count(Case)

baseline_counts <- baseline_draws_all %>% dplyr::count(Case)

assert_true(all(lognormal_counts$n == S),
            "Lognormal posterior draw count is not equal to S for all cases.")

assert_true(all(baseline_counts$n == S),
            "Baseline posterior draw count is not equal to S for all cases.")

add_audit_row("Lognormal draws per case", paste(lognormal_counts$n, collapse = ", "))
add_audit_row("Baseline draws per case", paste(baseline_counts$n, collapse = ", "))

# ------------------------------------------------------------
# 2. True-parameter consistency
# ------------------------------------------------------------

true_parameter_check <- true_parameters %>%
  dplyr::mutate(mean_gain_from_beta = 1 / beta_true,
                expected_gain_rate_check = lambda_true * mean_gain_true,
                R_true_check = lambda_true / c_rate - beta_true,
                psi_true_check = ifelse(lambda_true * mean_gain_true > c_rate & R_true_check > 0, exp(-R_true_check * u), 1))

assert_true(max_abs_diff(true_parameter_check$mean_gain_true, true_parameter_check$mean_gain_from_beta) < 1e-10,
            "mean_gain_true is not equal to 1 / beta_true.")

assert_true(max_abs_diff(true_parameter_check$expected_gain_rate, true_parameter_check$expected_gain_rate_check) < 1e-10,
            "expected_gain_rate is not equal to lambda_true * mean_gain_true.")

assert_true(max_abs_diff(true_parameter_check$psi_true_ultimate, true_parameter_check$psi_true_check) < 1e-10,
            "psi_true_ultimate is not consistent with true R and u.")

add_audit_row("True parameter consistency", "passed")

# ------------------------------------------------------------
# 3. Baseline exponential calculation checks
# ------------------------------------------------------------

baseline_required_cols <- c("Case", "draw_id", "lambda", "beta",
                            "R_raw", "net_profit_holds",
                            "psi", "psi_given_net_profit_holds")

assert_true(all(baseline_required_cols %in% names(baseline_draws_all)),
            "baseline_draws_all is missing required columns.")

# Recalculate whether the net profit condition holds for every baseline draw: lambda * E[X] = lambda / beta > c.
baseline_np_calc <- baseline_draws_all$lambda / baseline_draws_all$beta > c_rate
baseline_R_calc <- baseline_draws_all$lambda / c_rate - baseline_draws_all$beta # Recalculate the analytical exponential Lundberg root: R = lambda / c - beta.

# Recalculate the full posterior ruin probability.
baseline_psi_calc <- ifelse(baseline_np_calc & baseline_R_calc > 0,
                            exp(-baseline_R_calc * u), 1) # Valid positive-root draws use exp(-R*u); all other draws are assigned psi = 1.

# Recalculate psi conditional on a valid positive-root draw.
baseline_psi_given_calc <- ifelse(baseline_np_calc & baseline_R_calc > 0,
                                  exp(-baseline_R_calc * u), NA_real_) # The quantity is undefined and stored as NA when the condition does not hold.

assert_true(all(baseline_draws_all$net_profit_holds == baseline_np_calc),
            "Baseline net_profit_holds is not equal to lambda / beta > c.")

assert_true(max_abs_diff(baseline_draws_all$R_raw, baseline_R_calc) < 1e-12,
            "Baseline R_raw is not equal to lambda / c - beta.")

assert_true(max_abs_diff(baseline_draws_all$psi, baseline_psi_calc) < 1e-12,
            "Baseline psi is not consistent with exp(-R*u) and psi = 1 when NPC fails.")

# Identify draws where the stored conditional ruin probability is finite.
baseline_cond_idx <- is.finite(baseline_draws_all$psi_given_net_profit_holds)

# Verify that all finite stored conditional ruin probabilities agree with the recalculated exp(-R*u) values.
assert_true(max_abs_diff(baseline_draws_all$psi_given_net_profit_holds[baseline_cond_idx],
                         baseline_psi_given_calc[baseline_cond_idx]) < 1e-12,
            "Baseline psi_given_net_profit_holds is inconsistent.")

add_audit_row("Baseline exponential ruin calculation", "passed")

# ------------------------------------------------------------
# 4. Lognormal implied mean gain and NPC checks
# ------------------------------------------------------------

lognormal_required_cols <- c("Case", "draw_id", "lambda", "mu", "sigma2", "sigma",
                             "implied_mean_gain", "expected_gain_rate",
                             "net_profit_holds", "R_raw", "psi",
                             "psi_given_net_profit_holds", "root_status", "root_failed")

assert_true(all(lognormal_required_cols %in% names(posterior_draws_lognormal_all)),
            "posterior_draws_lognormal_all is missing required columns.")

# Require every lognormal lambda draw to be finite and strictly positive.
assert_true(all(is.finite(posterior_draws_lognormal_all$lambda) & posterior_draws_lognormal_all$lambda > 0),
            "Some lognormal lambda draws are non-positive or non-finite.")

# Require every lognormal variance draw to be finite and strictly positive.
assert_true(all(is.finite(posterior_draws_lognormal_all$sigma2) & posterior_draws_lognormal_all$sigma2 > 0),
            "Some lognormal sigma2 draws are non-positive or non-finite.")

# Recalculate the implied expected gain size for every posterior draw.
lognormal_mean_calc <- with(posterior_draws_lognormal_all, # with(data, expression) means no need to explicitly write posterior_draws_lognormal_all$mu or ...$sigma
                            exp(mu + sigma2 / 2))

lognormal_gain_rate_calc <- posterior_draws_lognormal_all$lambda * lognormal_mean_calc # lambda * E[X]

lognormal_np_calc <- lognormal_gain_rate_calc > c_rate # lambda * E[X] > c

assert_true(max_rel_diff(posterior_draws_lognormal_all$implied_mean_gain,
                         lognormal_mean_calc) < 1e-12,
            "Lognormal implied_mean_gain is not equal to exp(mu + sigma2 / 2).")

assert_true(max_rel_diff(posterior_draws_lognormal_all$expected_gain_rate,
                         lognormal_gain_rate_calc) < 1e-12,
            "Lognormal expected_gain_rate is not equal to lambda * implied_mean_gain.")

assert_true(all(posterior_draws_lognormal_all$net_profit_holds == lognormal_np_calc),
            "Lognormal net_profit_holds is not equal to lambda * E[X] > c.")

add_audit_row("Lognormal implied mean gain and NPC", "passed")

# ------------------------------------------------------------
# 5. Lognormal root-status and psi logic
# ------------------------------------------------------------

allowed_root_status <- c("found", # A valid positive Lundberg root was found.
                         "net_profit_fails", # The net profit condition failed, so no positive root was searched for.
                         "no_sign_change", # The numerical search interval did not contain a sign change.
                         "integration_failed_at_lower", # Numerical integration failed near the lower search boundary.
                         "uniroot_failed") # The numerical root-finding procedure failed.

assert_true(all(posterior_draws_lognormal_all$root_status %in% allowed_root_status),
            "Unexpected root_status value found.")

# Identify posterior draws where the net profit condition fails.
idx_np_fails <- !posterior_draws_lognormal_all$net_profit_holds

# Identify posterior draws where a valid positive Lundberg root was found.
idx_found <- posterior_draws_lognormal_all$root_status == "found"

# Identify draws where the net profit condition holds but the numerical procedure did not return a valid root.
idx_np_holds_not_found <- posterior_draws_lognormal_all$net_profit_holds & !idx_found

assert_true(all(posterior_draws_lognormal_all$root_status[idx_np_fails] == "net_profit_fails"),
            "Draws where NPC fails do not all have root_status = net_profit_fails.")

assert_true(all(posterior_draws_lognormal_all$psi[idx_np_fails] == 1),
            "Draws where NPC fails do not all have psi = 1.")

assert_true(all(is.na(posterior_draws_lognormal_all$R_raw[idx_np_fails])),
            "Draws where NPC fails should have R_raw = NA.")

assert_true(all(is.na(posterior_draws_lognormal_all$psi_given_net_profit_holds[idx_np_fails])),
            "Draws where NPC fails should have psi_given_net_profit_holds = NA.")

assert_true(all(is.finite(posterior_draws_lognormal_all$R_raw[idx_found]) & posterior_draws_lognormal_all$R_raw[idx_found] > 0),
            "Draws with root_status = found do not all have positive finite R_raw.")

# Recalculate psi(u) = exp(-R*u) for all draws with a successfully found root.
psi_found_calc <- exp(-posterior_draws_lognormal_all$R_raw[idx_found] * u)

assert_true(max_abs_diff(posterior_draws_lognormal_all$psi[idx_found], psi_found_calc) < 1e-10,
            "For found roots, psi is not equal to exp(-R*u).")

assert_true(max_abs_diff(posterior_draws_lognormal_all$psi_given_net_profit_holds[idx_found], psi_found_calc) < 1e-10,
            "For found roots, psi_given_net_profit_holds is not equal to exp(-R*u).")

assert_true(all(posterior_draws_lognormal_all$psi[idx_np_holds_not_found] == 1),
            "Draws where NPC holds but root is not found should have psi = 1.")

assert_true(all(is.na(posterior_draws_lognormal_all$psi_given_net_profit_holds[idx_np_holds_not_found])),
            "Draws where NPC holds but root is not found should have psi_given_net_profit_holds = NA.")

assert_true(all(posterior_draws_lognormal_all$root_failed 
                == (posterior_draws_lognormal_all$net_profit_holds & posterior_draws_lognormal_all$root_status != "found")),
            "root_failed is not equal to net_profit_holds & root_status != 'found'.")

add_audit_row("Lognormal root-status and psi logic", "passed")

# ------------------------------------------------------------
# 6. Lundberg equation residual check for found lognormal roots
# ------------------------------------------------------------
# Checking all roots can be slow because each residual uses numerical integration. We check a random subset of found roots.

set.seed(12345)

found_indices <- which(idx_found) # Return the row indices of all posterior draws where a root was found.
n_root_checks <- min(250, length(found_indices)) # Check at most 250 found roots.

if (n_root_checks > 0) {
  check_indices <- sample(found_indices, n_root_checks) # Randomly select the found-root rows to audit (without replacement)
  
  root_residuals <- mapply(
    FUN = function(R, lambda, mu, sigma2) {
      # Evaluate the lognormal Lundberg equation at one stored root using the corresponding posterior parameter draw.
      lundberg_function_lognormal(R = R,
        lambda = lambda,
        mu = mu,
        sigma2 = sigma2,
        c_rate = c_rate
      )
    },
    R = posterior_draws_lognormal_all$R_raw[check_indices],
    lambda = posterior_draws_lognormal_all$lambda[check_indices],
    mu = posterior_draws_lognormal_all$mu[check_indices],
    sigma2 = posterior_draws_lognormal_all$sigma2[check_indices]
  )
  # mapply() evaluates the function draw by draw using matching values of R, lambda, mu, and sigma2. Ex: R[1], lambda[1], mu[1], sigma2[1] and then [2], then [3]
  # A valid numerical root should produce a residual close to zero.
  
  max_root_residual <- max(abs(root_residuals), na.rm = TRUE) # Calculate the largest absolute Lundberg-equation residual among the audited roots.
  
  # Require the largest checked residual to be finite and below 1e-5.
  # A looser tolerance than in the analytical exponential case is used because the lognormal calculation involves numerical integration and root-finding.
  assert_true(is.finite(max_root_residual) && max_root_residual < 1e-5,
              paste0("Some checked lognormal roots do not satisfy the Lundberg equation closely enough. ",
                     "Max residual = ", signif(max_root_residual, 4)))
  
  add_audit_row("Max absolute Lundberg residual, checked roots", 
                signif(max_root_residual, 6),
                tolerance = 1e-5) 
  } else {
  add_audit_row("Lundberg residual check", "skipped: no found roots")
    }

# For a non-negative gain size X, its Laplace transform satisfies 0 <= L_X(R) <= 1.
#
# Since the Lundberg equation is cR = lambda * (1 - L_X(R)), the right-hand side cannot exceed lambda.
#
# Therefore, every valid positive root must satisfy: R <= lambda / c.
assert_true(all(posterior_draws_lognormal_all$R_raw[idx_found] <= posterior_draws_lognormal_all$lambda[idx_found] / c_rate + 1e-8), # The small 1e-8 addition allows for numerical floating-point error (Tips from AI).
            "Some found lognormal roots violate the bound R <= lambda / c.")

add_audit_row("Lognormal root bound R <= lambda / c", "passed")

# ------------------------------------------------------------
# 7. Mean-gain comparison table consistency
# ------------------------------------------------------------
# Count the number of posterior draws in each Case x Model combination.
mean_gain_counts <- mean_gain_compare_draws %>% dplyr::count(Case, model)

# Verify that each Case x Model combination contains the expected number of posterior draws.
assert_true(all(mean_gain_counts$n == S),
            "mean_gain_compare_draws does not have S rows per Case x Model.")

# Select the exponential baseline draws and reattach the corresponding posterior draw of beta using Case and draw_id as matching keys.
baseline_mean_gain_check <- mean_gain_compare_draws %>%
  dplyr::filter(model == "Exponential baseline") %>%
  dplyr::left_join(
    baseline_draws_all %>%
      dplyr::select(Case, draw_id, beta),
    by = c("Case", "draw_id")
  ) %>%
  dplyr::mutate(implied_mean_gain_check = 1 / beta) # Independently recompute the mean gain under the exponential model, where E[X] = 1 / beta.

# Verify that the mean gain stored in the comparison table agrees with the independently recomputed value for every posterior draw.
assert_true(max_rel_diff(baseline_mean_gain_check$implied_mean_gain,
              baseline_mean_gain_check$implied_mean_gain_check) < 1e-12, # The tolerance 1e-12 allows only negligible floating-point differences.
            "Exponential implied mean gain in comparison table is not equal to 1 / beta.")

# Select the lognormal misspecification draws and reattach the corresponding posterior draws of mu and sigma^2 using Case and draw_id as matching keys.
lognormal_mean_gain_check <- mean_gain_compare_draws %>%
  dplyr::filter(model == "Lognormal misspecified") %>%
  dplyr::left_join(
    posterior_draws_lognormal_all %>%
      dplyr::select(Case, draw_id, mu, sigma2),
    by = c("Case", "draw_id")
  ) %>%
  dplyr::mutate(implied_mean_gain_check = exp(mu + sigma2 / 2)) # Independently recompute the mean gain under the lognormal model, where E[X] = exp(mu + sigma^2 / 2).

# Verify that the mean gain stored in the comparison table agrees with the independently recomputed lognormal mean for every posterior draw.
assert_true(max_rel_diff(lognormal_mean_gain_check$implied_mean_gain,
                         lognormal_mean_gain_check$implied_mean_gain_check) < 1e-12,
            "Lognormal implied mean gain in comparison table is not equal to exp(mu + sigma2 / 2).")

add_audit_row("Mean-gain comparison table", "passed")

# ------------------------------------------------------------
# 8. Save audit summary
# ------------------------------------------------------------

audit_table <- dplyr::bind_rows(audit_rows)

write.csv(audit_table,
          file.path(tables_dir, "model_misspecification_audit_checks.csv"),
          row.names = FALSE)

cat("\nHard audit checks passed.\n")
cat("Audit table saved in:", file.path(tables_dir, "model_misspecification_audit_checks.csv"), "\n")

# ============================================================
# 9. FIGURES
# ============================================================
# Add readable case labels to the lognormal posterior draws.
# Converting the labels to a factor ensures that the cases appear in the intended order in all subsequent figures.
posterior_draws_lognormal_plot <- posterior_draws_lognormal_all %>%
  dplyr::mutate(Case_label = factor(make_case_label(Case), levels = case_levels))

# Add the same ordered case labels to the table containing the true simulation parameters so that reference lines are matched to the correct panels in the figures.
true_parameters_plot <- true_parameters %>%
  dplyr::mutate(
    Case_label = factor(make_case_label(Case), levels = case_levels)
  )

# ============================================================
# Figure: implied mean gain comparison
# ============================================================

# Calculate a case-specific upper plotting limit using the 99th percentile of the combined implied-mean-gain draws.
# This prevents a small number of extreme upper-tail draws from compressing the main posterior densities in the figure.
mean_gain_zoom_limits <- mean_gain_compare_draws %>%
  dplyr::group_by(Case) %>%
  dplyr::summarise(x_max = q_safe(implied_mean_gain, 0.99),
                   .groups = "drop")

# Attach the case-specific plotting limit to every posterior draw and remove draws above the corresponding 99th percentile.
mean_gain_compare_plot <- mean_gain_compare_draws %>%
  dplyr::left_join(mean_gain_zoom_limits, by = "Case") %>%
  dplyr::filter(implied_mean_gain <= x_max) %>%
  dplyr::mutate(Case_label = factor(make_case_label(Case), levels = case_levels),
                model = factor(model,levels = c("Exponential baseline", "Lognormal misspecified")))

# Compare the posterior densities of the implied expected gain size under the exponential baseline and lognormal misspecified models.
p_mean_gain <- ggplot(
  mean_gain_compare_plot,
  aes(x = implied_mean_gain, linetype = model)
) +
  geom_density(linewidth = 0.8, na.rm = TRUE) + # Estimate and draw one posterior density curve for each model.
  geom_vline( # Add a dashed vertical reference line at the true expected gain size used in the exponential data-generating process.
    data = true_parameters_plot,
    aes(xintercept = mean_gain_true),
    linewidth = 0.5,
    linetype = "dashed",
    inherit.aes = FALSE
  ) +
  facet_wrap(~ Case_label, ncol = 1, scales = "free_x") +
  labs(title = "Posterior implied mean gain",
       subtitle = "Central 99% of posterior draws shown; dashed line = true exponential mean",
       x = expression("Implied mean gain " ~ E[X]),
       y = "Density",
       linetype = "Model") +
  theme_thesis_figure(base_size = 10)

save_plot_both(p_mean_gain,
               "posterior_implied_mean_gain_baseline_vs_lognormal",
               width = 8,
               height = 5.4)

# Create panel-specific labels reporting the posterior probabilities of net profit condition failure and certain ruin.
npc_fail_labels_lognormal <- net_profit_diagnostics_lognormal %>%
  dplyr::mutate(
    Case_label = factor(make_case_label(Case), levels = case_levels), # Convert the internal case names into readable and ordered labels.
    
    # Construct a two-line annotation for each case.
    # sprintf("%.1f%%", ...) formats the probability as a percentage with one decimal place.
    label = paste0(
      "NPC fails: ", sprintf("%.1f%%", 100 * posterior_pr_net_profit_fails),
      "\n",
      "psi = 1: ", sprintf("%.1f%%", 100 * posterior_pr_psi_equals_one)),
    x = 0.56, # Set a fixed horizontal position for the annotation.
    y = Inf # Place the annotation at the upper boundary of each panel.
  )

# Plot the full posterior distribution of the ultimate ruin probability under the lognormal misspecified model.
p_psi_lognormal <- ggplot(
  posterior_draws_lognormal_plot,
  aes(x = psi)
) +
  # Group the posterior draws into histogram bins of width 0.025.
  # boundary = 0 aligns the bins with zero, and closed = "right" makes the right endpoint of each bin inclusive.
  geom_histogram(
    binwidth = 0.025,
    boundary = 0,
    closed = "right"
  ) +
  # Add the NPC-failure and psi = 1 annotations to each case panel.
  geom_text( 
    data = npc_fail_labels_lognormal,
    aes(x = x, y = y, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1.25,
    size = 3.1
  ) +
  facet_wrap(~ Case_label, ncol = 1) +
  labs(title = "Posterior ruin probability: lognormal model",
       subtitle = "Full posterior; mass at psi = 1 is due to net profit condition failures",
       x = expression(psi(u)),
       y = "Posterior draw count") +
  theme_thesis_figure(base_size = 10)

save_plot_both(p_psi_lognormal,
               "posterior_psi_lognormal_misspecification",
               width = 8,
               height = 5.4)

# Combine the exponential baseline and lognormal misspecification posterior draws into one long-format data frame.
psi_compare_plot_data <- dplyr::bind_rows(
  baseline_draws_all %>% dplyr::select(Case, draw_id, model, psi, psi_given_net_profit_holds, net_profit_holds),
  posterior_draws_lognormal_all %>% dplyr::select(Case, draw_id, model, psi, psi_given_net_profit_holds, net_profit_holds)
  ) %>%
  dplyr::mutate(
    Case_label = factor(make_case_label(Case), levels = case_levels),
    model = factor(model, levels = c("Exponential baseline", "Lognormal misspecified"))
  )

true_psi_plot <- true_parameters %>%
  dplyr::transmute(Case_label = factor(make_case_label(Case), levels = case_levels),
                   psi_true_ultimate = psi_true_ultimate)

# Keep only valid conditional ruin-probability draws.
psi_compare_conditional_plot <- psi_compare_plot_data %>%
  dplyr::filter(net_profit_holds, # Retain draws where the net profit condition holds.
                is.finite(psi_given_net_profit_holds), # Remove NA, NaN, Inf, and -Inf values. For the lognormal model, this also removes draws where the net profit condition holds but no valid root was found.
                
                # Confirm that the retained values are valid probabilities.
                psi_given_net_profit_holds >= 0,
                psi_given_net_profit_holds <= 1)

# Compare the conditional posterior densities of ruin probability under the two gain-size models.
p_psi_compare <- ggplot(
  psi_compare_conditional_plot,
  aes(x = psi_given_net_profit_holds, linetype = model)
) +
  geom_density(linewidth = 0.8, na.rm = TRUE) +
  geom_vline(
    data = true_psi_plot,
    aes(xintercept = psi_true_ultimate),
    inherit.aes = FALSE,
    linetype = "dotdash",
    linewidth = 0.6
  ) +
  facet_wrap(~ Case_label, ncol = 1) +
  labs(
    title = "Posterior ruin probability",
    subtitle = "Conditional draws shown; dot-dashed line = true ultimate ruin probability",
    x = expression(psi(u)),
    y = "Density",
    linetype = "Model"
  ) +
  theme_thesis_figure(base_size = 10)

save_plot_both(p_psi_compare,
               "posterior_psi_baseline_vs_lognormal",
               width = 8,
               height = 5.4)

# Add readable case labels to the summary table used in the bar chart.
misspec_change_plot_data <- model_misspecification_vs_baseline %>%
  dplyr::mutate(Case_label = factor(make_case_label(Case), levels = case_levels))

# Plot the absolute change in posterior median ruin probability between the lognormal misspecified model and exponential baseline.
p_abs_change <- ggplot(
  misspec_change_plot_data,
  aes(x = Case_label, y = absolute_delta_median_psi)
) +
  geom_col(width = 0.55) + # Draw one bar for each information case.
  labs(title = "Absolute median change in ruin probability from gain-size misspecification",
       subtitle = "Change is computed as |median psi_lognormal - median psi_exponential|",
       x = "Case",
       y = expression(abs(Delta~median~psi))
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

save_plot_both(p_abs_change,
               "model_misspecification_absolute_median_change_vs_baseline",
               width = 9,
               height = 4.8)

# ============================================================
# Appendix figure: full posterior implied mean gain, four panels
# ============================================================
#
# This figure is intended for the appendix.
#
# It shows the full posterior distribution of the implied mean gain
# separately for:
#   1. Exponential baseline, sparse information
#   2. Exponential baseline, rich information
#   3. Lognormal misspecified, sparse information
#   4. Lognormal misspecified, rich information
#
# We use base R plotting instead of ggplot facets because each panel
# should have its own x-axis. This is important because the lognormal
# misspecified model can produce extreme upper-tail draws, especially
# in the sparse information case.
# ============================================================

# Draw the full posterior density of implied mean gain for one specific Case x Model combination
plot_implied_mean_gain_density <- function(draws, case_name, model_name, main_title, show_legend = TRUE) {
  
  # Extract finite and strictly positive implied-mean-gain draws for the requested case and model.
  x <- draws$implied_mean_gain[
    draws$Case == case_name &
      draws$model == model_name &
      is.finite(draws$implied_mean_gain) &
      draws$implied_mean_gain > 0]
  
  # Extract the true mean gain for the requested case.
  # The value is constant within the case, so the first unique finite value is sufficient.
  true_mean <- unique(draws$mean_gain_true[draws$Case == case_name & is.finite(draws$mean_gain_true)])[1]
  
  # Display an informative empty panel if there are too few valid posterior draws to estimate a density. (Defensive)
  if (length(x) < 2) {
    plot.new()
    title(main = main_title)
    text(0.5, 0.5, "Too few posterior draws")
    return(invisible(NULL))
  }
  
  # Estimate the kernel density of the complete posterior sample.
  dens <- stats::density(x, na.rm = TRUE)
  
  # Plot the posterior density.
  plot(dens, main = main_title, xlab = "Implied mean gain E[X]", ylab = "Density", lwd = 2)
  
  # Add a dashed vertical reference line at the true mean gain.
  abline(v = true_mean, lty = 2, lwd = 2)
  
  # Optionally add a legend explaining the density and reference line.
  if (show_legend) {
    legend("topright",
           legend = c("Posterior density", "True mean gain"),
           lty = c(1, 2),
           lwd = c(2, 2),
           bty = "n",
           cex = 0.8)}
  
  # Return invisibly because the function is used for its plotting side effect rather than for a returned object.
  invisible(NULL)
}

# Draw a four-panel appendix figure containing every combination of information case and gain-size model.
draw_appendix_mean_gain_four_panel <- function(draws) {
  
  # Store the current base R graphics settings.
  old_par <- par(no.readonly = TRUE)
  
  # Restore the original graphics settings when the function ends, including when an error interrupts execution.
  on.exit(par(old_par), add = TRUE)
  
  # Configure a 2 x 2 plotting layout and set panel margins.
  par(mfrow = c(2, 2),
      mar = c(4.2, 4.2, 3.2, 1.2),
      oma = c(0, 0, 2.2, 0))
  
  # Panel 1: exponential baseline with sparse information.
  plot_implied_mean_gain_density(draws = draws,
                                 case_name = "sparse_information",
                                 model_name = "Exponential baseline",
                                 main_title = "Exponential baseline\nSparse information")
  
  # Panel 2: exponential baseline with rich information.
  plot_implied_mean_gain_density(draws = draws,
                                 case_name = "rich_information",
                                 model_name = "Exponential baseline",
                                 main_title = "Exponential baseline\nRich information")
  
  # Panel 3: lognormal misspecification with sparse information.
  plot_implied_mean_gain_density(draws = draws,
                                 case_name = "sparse_information",
                                 model_name = "Lognormal misspecified",
                                 main_title = "Lognormal misspecified\nSparse information")
  
  # Panel 4: lognormal misspecification with rich information.
  plot_implied_mean_gain_density(draws = draws,
                                 case_name = "rich_information",
                                 model_name = "Lognormal misspecified",
                                 main_title = "Lognormal misspecified\nRich information")
  
  # Add one shared title above the complete four-panel figure.
  mtext("Full posterior distribution of implied mean gain",
        outer = TRUE,
        cex = 1.25,
        font = 2)
}

# Save appendix figure as PNG
png(filename = file.path(figures_dir, "appendix_full_posterior_implied_mean_gain_four_panels.png"),
    width = 1800, height = 1200, res = 180)

# Draw the four panels on the currently active PNG device.
draw_appendix_mean_gain_four_panel(mean_gain_compare_draws)

# Close the PNG graphics device and write the completed file.
dev.off()

# Save appendix figure as PDF
pdf(file = file.path(figures_dir, "appendix_full_posterior_implied_mean_gain_four_panels.pdf"),
    width = 9, height = 6.5)

draw_appendix_mean_gain_four_panel(mean_gain_compare_draws)

dev.off()


# ============================================================
# 10. FINISH
# ============================================================

cat("\nModel misspecification analysis completed.\n")
cat("Tables saved in:", normalizePath(tables_dir, winslash = "/"), "\n")
cat("Draws saved in:", normalizePath(draws_dir, winslash = "/"), "\n")
cat("Figures saved in:", normalizePath(figures_dir, winslash = "/"), "\n")


# ============================================================
# End of 05_model_misspecification.R
# ============================================================
