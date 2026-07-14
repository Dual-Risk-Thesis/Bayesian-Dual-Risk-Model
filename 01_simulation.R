# ============================================================
# 01_simulation.R
# Simulated data for the dual risk model
# ============================================================
#
# Purpose:
# This script creates the simulated datasets used in the thesis.
# The data are generated from the classical continuous-time dual risk model
#
#   U(t) = u - c*t + sum_{i=1}^{N(t)} X_i.
#
# In the simulation:
#   - gains arrive according to a Poisson process,
#   - gain sizes are exponentially distributed,
#   - the cost rate c is constant,
#   - the same true model parameters are used in both cases.
#
# We simulate two cases:
#   1. sparse_information: A short observation period with few expected gain events. (This is meant to mimic the limited information we may have when trying to build an event log from real company data.)
#   2. rich_information: A longer observation period with more expected gain events. (An idealized benchmark where the same true process is observed for much longer.)
#
# The purpose of having two cases is to study how the amount of observed information affects Bayesian estimation of the ruin probability.
# 
#
# Main outputs:
#   simulated_dual_risk_data/events_sparse_information.csv
#   simulated_dual_risk_data/summary_sparse_information.csv
#   simulated_dual_risk_data/events_rich_information.csv
#   simulated_dual_risk_data/summary_rich_information.csv
#   simulated_dual_risk_data/true_parameters_two_cases.csv
# ============================================================

# ============================================================
# 1. SETTINGS FOR THE SIMULATION
# ============================================================

seed <- 999
set.seed(seed)

# Initial surplus and deterministic monthly expense rate.
u <- 207.28
c_rate <- 91.65 / 12

# The true process is chosen so that the net profit condition holds,
# but the ultimate ruin probability is still large enough to be interesting.
#
#   lambda_true * E[X] = net_profit_factor * c_rate
#
# With the settings below, the true ultimate ruin probability is around 0.22.
# A net_profit_factor of 1.5 means that the expected gain rate is 50% higher than the monthly expense rate.
net_profit_factor <- 1.5

# Same true lambda in both cases.
# Sparse: close to a realistic limited event log, around 6 expected gains.
# Rich: idealized benchmark, around 30 expected gains.
# E[N(t)] = lambda*T
lambda_common <- 2/12 # Around 2 gains per year

cases <- data.frame(
  Case = c("sparse_information", "rich_information"),
  T_obs = c(36, 180),
  lambda_true = c(lambda_common, lambda_common),
  design_role = c("short observation period with limited event information",
                  "longer observation period used as a rich benchmark"),
  stringsAsFactors = FALSE
)

# Folder where the simulated data will be saved.
output_folder_name <- "simulated_dual_risk_data"


# ============================================================
# 2. OUTPUT FOLDER
# ============================================================
# Get the name of the folder from which the script is currently run.
# getwd() gives the current working directory, normalizePath() standardizes the path format, and basename() keeps only the final folder name.
current_folder <- basename(normalizePath(getwd(), winslash = "/", mustWork = TRUE))

# "." means the current working directory and is used if we are already running the script from the output folder.
if (identical(current_folder, output_folder_name)) {
  output_dir <- "."
} else {
  output_dir <- output_folder_name
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}

cat("Simulation outputs will be saved in:", normalizePath(output_dir, winslash = "/"))


# ============================================================
# 3. TRUE PARAMETERS
# ============================================================

make_true_parameters <- function(case_name, u, c_rate, lambda_true, T_obs, net_profit_factor, design_role) {
  
  mean_gain_true <- (net_profit_factor * c_rate) / lambda_true # E[X]
  beta_true <- 1 / mean_gain_true # E[X] = 1/beta <=> beta = 1/E[X]
  
  expected_number_of_gains <- lambda_true * T_obs # E[N(t)] 
  expected_gain_rate <- lambda_true * mean_gain_true # lambda * E[X]
  net_profit_margin <- expected_gain_rate - c_rate # lambda * E[X] - c (If positive, net profit condition holds)
  expected_gain_rate_over_c <- expected_gain_rate / c_rate # Shoulde be 1.5
  
  R_raw_true <- lambda_true / c_rate - beta_true # Closed-form Lundberg root for the continuous-time dual risk model with exponential gain sizes
  net_profit_condition_holds <- lambda_true / beta_true > c_rate # lambda_true * E[X] > c_rate, where E[X] = 1 / beta_true.

  # Use the Lundberg root only if the net profit condition holds and the root is positive.
  # If no valid root exists, the net profit condition fails and ultimate ruin is set to 1.
  R_true <- if (net_profit_condition_holds && R_raw_true > 0) R_raw_true else NA_real_
  psi_true_ultimate <- if (is.finite(R_true)) exp(-R_true * u) else 1
  
  data.frame(
    Case = case_name,
    design_role = design_role,
    u = u,
    c_rate = c_rate,
    T_obs = T_obs,
    lambda_true = lambda_true,
    beta_true = beta_true,
    mean_gain_true = mean_gain_true,
    expected_number_of_gains = expected_number_of_gains,
    expected_gain_rate = expected_gain_rate,
    net_profit_factor = net_profit_factor,
    net_profit_margin = net_profit_margin,
    expected_gain_rate_over_c = expected_gain_rate_over_c,
    net_profit_condition_holds = net_profit_condition_holds,
    R_raw_true = R_raw_true,
    R_true = R_true,
    psi_true_ultimate = psi_true_ultimate,
    stringsAsFactors = FALSE
  )
}


# ============================================================
# 4. PATH SIMULATION
# ============================================================
# Create an empty event table with the correct column names.
# If gains occur, the event information will later be stored in these columns.
# If no gains occur, the output will still have the same structure.

empty_events_df <- function() {
  data.frame(
    Case = character(),
    event_id = integer(),
    arrival_time = numeric(),
    interarrival_time = numeric(),
    gain_size = numeric(),
    surplus_before_gain = numeric(),
    surplus_after_gain = numeric(),
    stringsAsFactors = FALSE
  )
}

simulate_dual_risk_path <- function(case_name, u, c_rate, lambda_true, beta_true, T_obs) {
  
  # Simulate the number of gains over the full observation horizon.
  # Conditional on N(T), arrival times are ordered Uniform(0, T).
  n_events <- rpois(1, lambda_true * T_obs)
  
  # If no gains occur, there are no arrival times, waiting times, or gain sizes.
  if (n_events == 0) {
    arrival_times <- numeric(0)
    interarrival_times <- numeric(0)
    gain_sizes <- numeric(0)
  } else { # Given the number of gains, their arrival times are simulated as sorted Uniform(0, T_obs) values.
    arrival_times <- sort(runif(n_events, min = 0, max = T_obs))
    interarrival_times <- diff(c(0, arrival_times)) # Time between consecutive gains, including the time from 0 to the first gain.
    gain_sizes <- rexp(n_events, rate = beta_true)
  }
  
  surplus_current <- u
  time_previous <- 0
  ruined <- FALSE
  ruin_time <- NA_real_
  event_rows <- vector("list", n_events) # Create an empty list with n_events places
  
  for (j in seq_len(n_events)) {
    surplus_before_gain <- surplus_current - c_rate * interarrival_times[j] # surplus before gain = current surplus - c * time since previous gain
    
    if (!ruined && surplus_before_gain <= 0) { # If ruin has not already been recorded and the surplus reaches zero or becomes negative before the next gain, register ruin
      ruined <- TRUE
      ruin_time <- time_previous + surplus_current / c_rate # U(t) = u - c*t = 0
    }
    
    surplus_after_gain <- surplus_before_gain + gain_sizes[j]
    
    event_rows[[j]] <- data.frame(
      Case = case_name,
      event_id = j,
      arrival_time = arrival_times[j],
      interarrival_time = interarrival_times[j],
      gain_size = gain_sizes[j],
      surplus_before_gain = surplus_before_gain,
      surplus_after_gain = surplus_after_gain,
      stringsAsFactors = FALSE
    )
    
    surplus_current <- surplus_after_gain
    time_previous <- arrival_times[j]
  }
  
  events <- if (n_events == 0) empty_events_df() else do.call(rbind, event_rows) # do.call(rbind, event_rows) = rbind(event_rows[[1]], event_rows[[2]], ...)
  
  # Surplus at the end of the observation horizon, after the final waiting time.
  final_surplus <- surplus_current - c_rate * (T_obs - time_previous)
  
  if (!ruined && final_surplus <= 0) {
    ruined <- TRUE
    ruin_time <- time_previous + surplus_current / c_rate
  }
  
  last_arrival_time <- if (n_events == 0) 0 else max(arrival_times)
  
  summary <- data.frame(
    Case = case_name,
    ruined = as.integer(ruined),
    ruin_time = ruin_time,
    number_of_gains = n_events,
    final_surplus = final_surplus,
    observation_horizon = T_obs,
    T_for_lambda_likelihood = T_obs,
    exposure_until_ruin_or_T = if (ruined) ruin_time else T_obs, # Here, the survival time within the observation period is stored.
    last_arrival_time = last_arrival_time,
    censored_time_after_last_gain = T_obs - last_arrival_time, # Time from the last gain to the end of the observation period
    stringsAsFactors = FALSE
  )
  
  list(events = events, summary = summary)
}


# ============================================================
# 5. RUN SIMULATION
# ============================================================
true_parameters_list <- vector("list", nrow(cases)) # This creates an empty list with one slot for each row in cases.
events_list <- vector("list", nrow(cases))
summary_list <- vector("list", nrow(cases))

names(true_parameters_list) <- cases$Case
names(events_list) <- cases$Case
names(summary_list) <- cases$Case

for (i in seq_len(nrow(cases))) {
  case_name <- cases$Case[i]
  
  true_i <- make_true_parameters(
    case_name = case_name,
    u = u,
    c_rate = c_rate,
    lambda_true = cases$lambda_true[i],
    T_obs = cases$T_obs[i],
    net_profit_factor = net_profit_factor, 
    design_role = cases$design_role[i]
  )
  
  sim_i <- simulate_dual_risk_path(
    case_name = case_name,
    u = u,
    c_rate = c_rate,
    lambda_true = true_i$lambda_true,
    beta_true = true_i$beta_true,
    T_obs = true_i$T_obs
  )

  true_parameters_list[[case_name]] <- true_i
  events_list[[case_name]] <- sim_i$events
  summary_list[[case_name]] <- sim_i$summary
}

true_parameters <- do.call(rbind, true_parameters_list)
summaries_all_cases <- do.call(rbind, summary_list)

rownames(true_parameters) <- NULL
rownames(summaries_all_cases) <- NULL

# Take each simulated case summary and attach the true parameters belonging to the same case.
simulation_diagnostics <- merge(
  summaries_all_cases,
  true_parameters,
  by = "Case",
  all.x = TRUE, # This keeps all rows from the first table (summaries_all_cases) because it is our main table
  sort = FALSE # Keep the original order
)


# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

for (case_name in cases$Case) {
  write.csv(
    events_list[[case_name]],
    file = file.path(output_dir, paste0("events_", case_name, ".csv")),
    row.names = FALSE
  )
  
  write.csv(
    summary_list[[case_name]],
    file = file.path(output_dir, paste0("summary_", case_name, ".csv")),
    row.names = FALSE
  )
}

write.csv(
  true_parameters,
  file = file.path(output_dir, "true_parameters_two_cases.csv"),
  row.names = FALSE
)


# ============================================================
# 7. PRINT DIAGNOSTICS
# ============================================================

cat("\n================ USER SETTINGS ================\n")
cat("Seed:", seed, "\n")
cat("Initial surplus u:", u, "\n")
cat("Monthly expense rate c_rate:", c_rate, "\n")
cat("Common lambda_true:", lambda_common, "\n")
cat("Net profit factor:", net_profit_factor, "\n")

cat("\n================ CASE DESIGN ================\n")
print(cases)

cat("\n================ TRUE PARAMETERS ================\n")
print(
  true_parameters[, c(
    "Case",
    "design_role",
    "T_obs",
    "lambda_true",
    "beta_true",
    "mean_gain_true",
    "expected_number_of_gains",
    "net_profit_condition_holds",
    "R_true",
    "psi_true_ultimate"
  )]
)

cat("\n================ SIMULATION DIAGNOSTICS ================\n")
print(
  simulation_diagnostics[, c(
    "Case",
    "number_of_gains",
    "expected_number_of_gains",
    "observation_horizon",
    "T_for_lambda_likelihood",
    "ruined",
    "ruin_time",
    "final_surplus",
    "last_arrival_time",
    "censored_time_after_last_gain"
  )]
)

cat("\nSimulation completed. Files saved in:", normalizePath(output_dir, winslash = "/"), "\n")
