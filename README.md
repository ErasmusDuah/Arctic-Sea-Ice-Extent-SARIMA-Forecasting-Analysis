# Arctic Sea Ice Extent: SARIMA Forecasting Analysis

A time series analysis of monthly Arctic sea ice extent (1979–2005), fitting and validating a SARIMA model in R.

## Overview

This project explores whether Arctic sea ice extent, a strongly seasonal climate variable, can be forecast using classical time series methods. This series has a real, strong seasonal structure, making it a useful case for practicing the full SARIMA workflow: stationarity testing, model selection, residual diagnostics, and out-of-sample validation.

## Data

- **Source**: TTU Time Series Datasets, `MonthlyArcticSeaIce1979-2005.tsm`
- **Range**: January 1979 – December 2005
- **Frequency**: Monthly (324 observations)
- **Units**: Million square kilometers

## Methodology

1. **Visual inspection** — plotted the raw series, revealing a strong repeating annual cycle.
2. **Stationarity testing** — an ADF test on the raw series confirmed non-seasonal stationarity (p = 0.01), so no ordinary differencing was needed (d = 0). An OCSB test checked the seasonal component and found a seasonal unit root (test statistic -1.4499 vs 5% critical value -1.8030), indicating one seasonal difference was needed (D = 1).
3. **Model selection** — `auto.arima()`, run with a full search (`stepwise = FALSE`, `approximation = FALSE`), selected SARIMA(1,0,0)(0,1,1)[12] as the best model by AIC (56.90).
4. **Residual diagnostics** — Ljung-Box test on residuals: p = 0.616, no leftover autocorrelation.
5. **Volatility check** — ARCH-LM test: p = 0.160, no significant volatility clustering; GARCH not required.
6. **Forecasting** — generated a 24-month-ahead forecast, correctly reproducing the seasonal wave.
7. **Validation** — trained on the first 300 months, held out the final 24 months, and scored the forecast against the real, hidden values.

## Results

| Metric | Value |
|---|---|
| RMSE | 0.5246 |
| MAE | 0.4663 |

Train size: 300 months. Test size: 24 months (held out, never seen during fitting).

Sample of predicted vs actual (first 6 months of the test period):

| Month | Predicted | Actual |
|---|---|---|
| 1 | 14.37 | 14.05 |
| 2 | 15.27 | 14.97 |
| 3 | 15.43 | 15.07 |
| 4 | 14.65 | 14.16 |
| 5 | 13.32 | 12.64 |
| 6 | 11.81 | 11.62 |

## Tools

R, `forecast`, `tseries`, `FinTS`

## How to reproduce

<pre> ```r #
# Load all libraries needed for the full workflow
library(tseries)     # adf.test(), kpss.test()
library(FinTS)        # ArchTest()
library(forecast)     # Arima(), auto.arima(), checkresiduals()
library(rugarch)      # GARCH modeling
library(uroot)


# Load the file from my pc, check its structure before assuming a format
ice_raw <- read.table("C:/Users/DevOps/Downloads/ArcticSeaIce.tsm", header = FALSE)

str(ice_raw)
head(ice_raw)


# Convert to ts object: monthly data starting January 1979
ice <- ts(ice_raw$V1, start = c(1979, 1), frequency = 12)

# Plot it
plot(ice, main = "Monthly Arctic Sea Ice Extent", ylab = "Million sq km", xlab = "Year")

# Decompose into trend, seasonal, and random components
ice_decomp <- decompose(ice)
plot(ice_decomp)

# Check stationarity: ADF and KPSS together
adf.test(ice)


# ============================================
# Step: Test for SEASONAL unit roots using HEGY
# ADF already told us: no regular unit root detected (p = 0.01, d = 0 likely)
# Now we check if seasonal differencing (D) is needed
# ============================================

# Run OCSB test explicitly to check seasonal differencing (D)
nsdiffs(ice, test = "ocsb")

# For comparison, check what it says for regular differencing (d) too
ndiffs(ice, test = "kpss")

ocsb.test(ice)

# ============================================
# Step: Plot ACF and PACF on the raw series
# ============================================

# Plot both together for easy comparison, lag.max = 48 shows 4 years of lags
# so you can see seasonal spikes at lag 12, 24, 36
par(mfrow = c(2, 1))  # stack ACF and PACF one above the other

acf(ice, lag.max = 48, main = "ACF of Arctic Sea Ice Extent")
pacf(ice, lag.max = 48, main = "PACF of Arctic Sea Ice Extent")

par(mfrow = c(1, 1))  # reset plotting layout back to normal after


#using auto arima to select the best model. ie the sarima model with least AIC value
library(forecast)

fit <- auto.arima(ice, 
                  seasonal = TRUE,
                  stepwise = FALSE,      
                  approximation = FALSE, 
                  trace = TRUE)          

summary(fit)


# ============================================
# Step: Fit final model and check residuals
# Going with auto.arima's pick, confirmed by OCSB test: D=1
# ARIMA(1,0,0)(0,1,1)[12]
# p=1 (non-seasonal AR), d=0, q=0
# P=0, D=1 (seasonal differencing), Q=1 (seasonal MA)
# ============================================

fit <- arima(ice, order = c(1,0,0),                    # non-seasonal part: p=1, d=0, q=0
             seasonal = list(order = c(0,1,1), period = 12))  # seasonal part: P=0, D=1, Q=1, s=12

# checkresiduals() runs several diagnostics at once:
# - plots the residuals over time (should look like random noise, no pattern)
# - plots the ACF of residuals (bars should mostly stay inside the blue band)
# - runs a Ljung-Box test (checks if residuals are significantly different from white noise)
checkresiduals(fit)

# ============================================
# Step: Plot ACF and PACF of the residuals from the fitted model
# This is a visual check, same idea as Ljung-Box, but lets you SEE
# whether any lag pokes out past the blue band rather than just
# getting a single p-value summary
# ============================================

par(mfrow = c(2, 1))  # stack ACF and PACF one above the other

acf(residuals(fit), lag.max = 48, main = "ACF of Residuals")
pacf(residuals(fit), lag.max = 48, main = "PACF of Residuals")

par(mfrow = c(1, 1))  # reset plotting layout back to normal

# ARCH LM test on residuals from your fitted SARIMA model
ArchTest(residuals(fit), lags = 12)

#no GARCH needed since no volatility clustering

# ============================================
# Step: Hold out the last 24 months as a test set
# Fit the model on data BEFORE that period only
# Then forecast forward and compare against the real values we hid
# ============================================

n <- length(ice)
h <- 24  # number of months held out for testing

train <- window(ice, end = time(ice)[n - h])
test  <- window(ice, start = time(ice)[n - h + 1])

# Fit the same model structure on the training data only
fit_train <- arima(train, order = c(1,0,0), 
                   seasonal = list(order = c(0,1,1), period = 12))

# Forecast forward exactly as many periods as held out
fc <- forecast(fit_train, h = h)

# ============================================
# Step: Build a table comparing predicted vs actual side by side
# ============================================
comparison <- data.frame(
  Date     = time(test),
  Actual   = as.numeric(test),
  Forecast = as.numeric(fc$mean),
  Lo_95    = as.numeric(fc$lower[,2]),
  Hi_95    = as.numeric(fc$upper[,2])
)
print(comparison)

# ============================================
# Step: Plot forecast against actual visually
# ============================================
plot(fc, main = "Predicted vs Actual (Hold-out Test)")
lines(test, col = "red", lwd = 2)
legend("topleft", legend = c("Forecast", "Actual"), 
       col = c("blue", "red"), lty = 1, lwd = 2)

# ============================================
# Step: Accuracy metrics comparing forecast vs actual numerically
# ============================================
accuracy(fc, test)

# ============================================
# Step: Forecast future values using the fitted model
# fit = ARIMA(1,0,0)(0,1,1)[12], already confirmed good via residual checks
# h = how many future periods to forecast (here, 24 months = 2 years ahead)
# ============================================

forecast_result <- forecast(fit, h = 24)

# ============================================
# Step: Plot the forecast
# Shows the historical series plus the forecasted values,
# with shaded confidence intervals (default 80% and 95%)
# ============================================
plot(forecast_result, main = "Arctic Sea Ice Extent Forecast")

# ============================================
# Step: View the actual forecasted numbers
# Includes point forecast plus lower/upper bounds for both CI levels
# ============================================
print(forecast_result)
``` </pre>
