# Arctic Sea Ice Extent: SARIMA Forecasting Analysis

A time series analysis of monthly Arctic sea ice extent (1979–2005), fitting and validating a SARIMA model in R.

## Overview

This project explores whether Arctic sea ice extent, a strongly seasonal climate variable, can be forecast using classical time series methods. Unlike financial return data, this series has real, identifiable trend and seasonal structure, making it a useful contrast case for practicing the full ARIMA/SARIMA workflow: decomposition, stationarity testing, model selection, residual diagnostics, and out-of-sample validation.

## Data

- **Source:** [TTU Time Series Datasets](https://www.math.ttu.edu/~atrindad/tsdata/index.html), `MonthlyArcticSeaIce1979-2005.tsm`
- **Range:** January 1979 – December 2005
- **Frequency:** Monthly (324 observations)
- **Units:** Million square kilometers

## Methodology

1. **Visual inspection** — plotted the raw series, revealing a strong repeating annual cycle plus a gradual downward trend.
2. **Decomposition** — used `decompose()` to separate trend, seasonal, and remainder components, confirming a decline from roughly 12.7 to 11.2 million sq km over the period.
3. **Stationarity testing** — ADF and KPSS tests on the raw series were misleading due to large seasonal swings masking the trend. `nsdiffs()` and `ndiffs()` gave the correct read: one seasonal difference needed, no regular differencing needed. Applied one seasonal difference and re-confirmed stationarity.
4. **Model selection** — `auto.arima()` selected **SARIMA(1,0,0)(0,1,1)[12]**.
5. **Residual diagnostics** — Ljung-Box test on residuals: p = 0.616, no leftover autocorrelation.
6. **Volatility check** — ARCH-LM test: p = 0.160, no significant volatility clustering; GARCH not required.
7. **Forecasting** — generated a 24-month-ahead forecast, correctly reproducing the seasonal wave and ongoing decline.
8. **Validation** — trained on the first 300 months, held out the final 24 months, and scored the forecast against the real, hidden values.

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

See `analysis.R` for the full code, or the block below.

```r
# Load all libraries needed for the full workflow
library(tseries)     # adf.test(), kpss.test()
library(FinTS)        # ArchTest()
library(forecast)     # Arima(), auto.arima(), checkresiduals()
library(rugarch)      # GARCH modeling

# Load the file, check its structure before assuming a format
ice_raw <- read.table("ArcticSeaIce.tsm", header = FALSE)
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
kpss.test(ice)

# Check how many seasonal differences are recommended
nsdiffs(ice)

# Check how many regular (non-seasonal) differences are recommended
ndiffs(ice)

# Apply one seasonal difference, as recommended by nsdiffs
ice_diff <- diff(ice, lag = 12)
plot(ice_diff, main = "Seasonally Differenced Arctic Sea Ice", ylab = "Difference", xlab = "Year")

# Re-test stationarity on the seasonally differenced series
adf.test(ice_diff)
kpss.test(ice_diff)

# Automatic search, letting R find the best SARIMA order
auto_ice <- auto.arima(ice, stepwise = FALSE, approximation = FALSE)
summary(auto_ice)

# Fit the SARIMA model selected by auto.arima
ice_model <- Arima(ice, order = c(1,0,0), seasonal = c(0,1,1))

# Check residuals: is there any leftover pattern?
checkresiduals(ice_model)

# Check whether the variance shows clustering
ArchTest(residuals(ice_model), lags = 12)

# Forecast forward using the SARIMA model, since no GARCH is needed
ice_forecast <- forecast(ice_model, h = 24)
plot(ice_forecast)

# Hide the last 24 months, train on everything before that
n_ice <- length(ice)
train_ice <- window(ice, end = c(1979 + floor((n_ice-24-1)/12), ((n_ice-24-1) %% 12) + 1))
test_ice <- window(ice, start = c(1979 + floor((n_ice-24)/12), ((n_ice-24) %% 12) + 1))
length(train_ice)
length(test_ice)

# Fit the same SARIMA order, but only on the training portion
ice_model_train <- Arima(train_ice, order = c(1,0,0), seasonal = c(0,1,1))

# Forecast forward exactly 24 months, into the hidden test period
ice_guess <- forecast(ice_model_train, h = 24)

# Pull out just the predicted values
predicted_ice <- as.numeric(ice_guess$mean)
head(predicted_ice)

# The real, hidden answers
actual_ice <- as.numeric(test_ice)
head(actual_ice)

# Score the forecast
rmse_ice <- sqrt(mean((predicted_ice - actual_ice)^2))
mae_ice  <- mean(abs(predicted_ice - actual_ice))
rmse_ice
mae_ice
```
