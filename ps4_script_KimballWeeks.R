library(pacman)

p_load(tidyverse, fpp3, BVAR)


# Part 2: Liquor store sales

# 2.1 Data -----------------------------

liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)

# Time plot of sales
autoplot(liquor, sales) +
  labs(title = "Sales at Beer, Wine, and Liquor Stores",
       x = "Month",
       y = "Sales")

# Season plot of log(sales)
liquor |>
  gg_season(log(sales)) +
  labs(title = "Seasonal Plot of log(Sales) at Beer, Wine, and Liquor Stores",
       x = "Month",
       y = "log(Sales)")


# 2.2-------------------------

liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

# Residual checks
liquor_fit |> select(arima) |> gg_tsresiduals()

# Ljung-Box
liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)


# 2.3 -----------------------

liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))


liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
        arima  = ARIMA(log(sales)))

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")


liquor_cv_fit <- readRDS("results/liquor_cv_fit.rds")

liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))


# 2.4 ------------------------------------

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

# ARIMA vs. ETS
forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)

# ARIMA vs. seasonal random walk
forecast::dm.test(liquor_errs$arima, liquor_errs$snaive, h = 1, power = 2)


# Part 3: Forecast combination --------------------------------------------

# 3a. Equal-weight combination of ARIMA and ETS
liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

# 3b. Correlation between ARIMA and ETS forecast errors
cor(liquor_errs$arima, liquor_errs$ets)


# Part 4: A medium-scale VAR vs. a BVAR ------------------

# 4.1 Data ------------------------

qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

# Growth rate in percent
growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),         # real GDP
            cons   = growth(PCECC96),       # real consumption
            inv    = growth(GPDIC1),        # real investment
            emp    = growth(PAYEMS),        # payroll employment
            infl   = growth(GDPCTPI),       # GDP price index
            unrate = difference(UNRATE),    # unemployment rate
            ffr    = difference(FEDFUNDS),  # federal funds rate
            gs10   = difference(GS10)) |>   # 10-year Treasury yield
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()

nrow(y)   # should be 240
head(y)

# 4.1c. TS-CV setup
N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]


# 4.2 OLS VARs ------------------------------------------------------------

# Estimates a VAR(p) by OLS and forecasts one period ahead
var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)            # columns: y_t, then its lags 1 to p
  M <- ncol(z)                    # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))

rmse_by_var(fc_var1)
rmse_by_var(fc_var4)


# 4.3 A BVAR --------------------------------------------------------------

# 4.3a. BVAR(4) with a Minnesota prior, lambda held fixed
bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))         # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")

# 4.3b
set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)
saveRDS(fc_bvar_005, "results/fc_bvar_005.rds")

fc_bvar_03  <- readRDS("results/fc_bvar_03.rds")
fc_bvar_005 <- readRDS("results/fc_bvar_005.rds")

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)

saveRDS(var_results, "results/var_results.rds")

var_results |> knitr::kable(digits = 3)

# 4.3c
e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)

# Inflation
e_var4_infl <- actual[, "infl"] - fc_var4[, 5]
e_bvar_infl <- actual[, "infl"] - fc_bvar_03[, 5]

forecast::dm.test(e_var4_infl, e_bvar_infl, h = 1, power = 2)



# Part 5: A factor model with FRED-MD --------------

# 5.1 Data ---------------------------

md_raw <- read_csv("data/fredmd_2026-09.csv", show_col_types = FALSE)
tcode <- unlist(md_raw[1, -1])
md <- md_raw[-1, ] |> mutate(date = as.Date(sasdate, "%m/%d/%Y"))

# Applies a McCracken-Ng
transform_series <- function(x, code) {
  d <- function(v) c(NA, diff(v))
  switch(as.character(code),
         "1" = x,                      # level
         "2" = d(x),                   # first difference
         "3" = d(d(x)),                # second difference
         "4" = log(x),                 # log
         "5" = 100 * d(log(x)),        # growth rate
         "6" = 100 * d(d(log(x))),     # change in the growth rate
         "7" = d(x / dplyr::lag(x) - 1))
}

# Transform every series
fm <- map2_dfc(md[names(tcode)], tcode, transform_series) |>
  mutate(date = md$date, .before = 1) |>
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

# Drop any series with missing values in the sample
fm <- fm[, colSums(is.na(fm)) == 0]

dim(fm)   

tcode[c("INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS")]


# 5.2 Estimate the factors ------------------------------------------------

Z <- fm |> select(-date) |> as.matrix()

pc <- prcomp(scale(Z))

summary(pc)$importance[, 1:5]

# First factor, sign-flipped so it rises with industrial production
F1 <- pc$x[, 1] * sign(cor(pc$x[, 1], fm$INDPRO))

tibble(date = fm$date, F1 = F1) |>
  ggplot(aes(date, F1)) +
  geom_line() +
  labs(title = "First Factor from FRED-MD", x = NULL, y = NULL) +
  theme_minimal()

# Eight series most correlated with each of the first two factors
sort(abs(cor(pc$x[, 1], Z))[1, ], decreasing = TRUE)[1:8]
sort(abs(cor(pc$x[, 2], Z))[1, ], decreasing = TRUE)[1:8]


# 5.3 Replicate the horse race from class ---------------------------------

ip <- fm$INDPRO
n <- nrow(Z)
init <- which(fm$date == as.Date("1989-12-01"))

e_ar <- NULL
e_fa <- NULL
for (T in init:(n - 1)) {
  # Factors re-estimated using only data through T
  F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:3]
  d <- data.frame(y_next = ip[2:T], y = ip[1:(T - 1)], F_T[1:(T - 1), ])
  new <- data.frame(y = ip[T], t(F_T[T, ]))
  e_ar <- c(e_ar, ip[T + 1] - predict(lm(y_next ~ y, d), new))
  e_fa <- c(e_fa, ip[T + 1] - predict(lm(y_next ~ ., d), new))
}

c(RMSE_AR = sqrt(mean(e_ar^2)), RMSE_FA = sqrt(mean(e_fa^2)))   # class: 0.617, 0.589


# 5.4 forecast of inflation --------------------------------------

direct_race <- function(series, h = 12, r = 3) {
  lv <- md |>
    mutate(log_x = log(.data[[series]]),
           g = 100 * (log_x - lag(log_x))) |>
    filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))
  yh <- 100 * (lead(lv$log_x, h) - lv$log_x)   # growth over the next h months
  g  <- lv$g                                   # growth over the latest month

  e_ar <- NULL
  e_fa <- NULL
  for (T in init:(n - h)) {
    F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:r, drop = FALSE]
    i <- 1:(T - h)   # only targets already observed at origin T
    d <- data.frame(yh = yh[i], g = g[i], F_T[i, , drop = FALSE])
    new <- data.frame(g = g[T], F_T[T, , drop = FALSE])
    e_ar <- c(e_ar, yh[T] - predict(lm(yh ~ g, d), new))
    e_fa <- c(e_fa, yh[T] - predict(lm(yh ~ ., d), new))
  }

  tibble(series, forecasts = length(e_ar),
         RMSE_AR = sqrt(mean(e_ar^2)), RMSE_FA = sqrt(mean(e_fa^2)),
         MAE_AR = mean(abs(e_ar)), MAE_FA = mean(abs(e_fa)),
         DM_pvalue = forecast::dm.test(e_ar, e_fa, h = h, power = 2,
                                       varestimator = "bartlett")$p.value)
}

direct_race("CPIAUCSL")
