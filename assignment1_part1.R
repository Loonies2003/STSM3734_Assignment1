############################################################
# 1. Setup
############################################################

set.seed(2026)   # Ensures reproducibility
n <- 500         # Number of observations


############################################################
# 2. Simulate Predictor Variables
############################################################

# ---- Size (continuous) ----
# Realistic SA house sizes (square meters)
size <- runif(n, min = 50, max = 300)


# ---- Bedrooms (count, correlated with size) ----
# Larger houses tend to have more bedrooms
beds <- round(size / 60 + rnorm(n, 0, 1))

# Clamp values between 1 and 6
beds <- pmin(6, pmax(1, beds))


# ---- Renovation status (categorical) ----
# About 30% renovated
renovated_bin <- rbinom(n, 1, 0.3)

renovated <- factor(renovated_bin,
                    levels = c(0,1),
                    labels = c("No", "Yes"))


# ---- Location (categorical) ----
# About 60% Urban
location_bin <- rbinom(n, 1, 0.6)

location <- factor(location_bin,
                   levels = c(0,1),
                   labels = c("Rural", "Urban"))


# ---- Distance from city (continuous) ----
# Urban houses closer to city
# Rural houses farther away
dist_km <- ifelse(location == "Urban",
                  runif(n, 1, 30),
                  runif(n, 30, 100))


############################################################
# 3. Define the TRUE Regression Model
############################################################

# Intercept reflects baseline SA property value
beta0 <- 650000

# Effects chosen to mimic realistic pricing behaviour
beta_size <- 5000        # Price per m²
beta_beds <- 30000       # Price per bedroom
beta_reno <- 150000      # Renovation premium
beta_urban <- 200000     # Urban premium
beta_dist <- -2000       # Distance penalty
beta_interaction <- 2000 # Extra size value in urban areas


############################################################
# 4. Convert Factors → Numeric Effects (for simulation)
############################################################

reno_effect  <- ifelse(renovated == "Yes", 1, 0)
urban_effect <- ifelse(location == "Urban", 1, 0)


############################################################
# 5. Add Random Noise (Normal Error)
############################################################

# Noise represents unobserved influences
epsilon <- rnorm(n, mean = 0, sd = 150000)


############################################################
# 6. Generate House Prices
############################################################

price <- beta0 +
  beta_size * size +
  beta_beds * beds +
  beta_reno * reno_effect +
  beta_urban * urban_effect +
  beta_dist * dist_km +
  beta_interaction * size * urban_effect +  # Interaction term
  epsilon


############################################################
# 7. Create Final Dataset
############################################################

house_data <- data.frame(price, size, beds, renovated, location, dist_km)

head(house_data)
############################################################
# 8. Exploratory Plots
############################################################

# ---- Scatterplot Matrix (ALL predictors) ----
pairs(~ size + beds + renovated + location + dist_km,
      data = house_data,
      main = "Scatterplot Matrix of Predictors")


# ---- Histograms / Barplots ----

# Continuous variables → histograms
hist(size, main = "Histogram of House Size", xlab = "Size (m²)")
hist(beds, main = "Histogram of Bedrooms", xlab = "Number of Bedrooms")
hist(dist_km, main = "Histogram of Distance", xlab = "Distance from City (km)")

# Categorical variables → barplots
barplot(table(renovated), main = "Renovation Status")
barplot(table(location), main = "Location Type")


############################################################
# 9. Fit Linear Regression Model
############################################################

model <- lm(price ~ size + beds + renovated + location + dist_km +
              size:location,
            data = house_data)

summary(model)

plot(model)


############################################################
# 1. Fit Linear Regression Model
############################################################
model <- lm(price ~ size + beds + renovated + location + dist_km + size:location,
            data = house_data)

############################################################
# 2. Standard Diagnostic Plots
############################################################

# Residuals vs Fitted
plot(model, which = 1, main = "Residuals vs Fitted")
# Interpretation:
# - Checks linearity and homoscedasticity
# - Look for random scatter around zero

# Normal Q-Q Plot
plot(model, which = 2, main = "Normal Q-Q")
# Interpretation:
# - Checks normality of residuals
# - Deviations at tails expected due to F-distributed errors

# Scale-Location Plot
plot(model, which = 3, main = "Scale-Location")
# Interpretation:
# - Checks homoscedasticity
# - Funnel shape indicates variance increases with fitted price

# Residuals vs Leverage (with Cook's Distance)
# Compute leverage and Cook's distance
lev <- hatvalues(model)
cook <- cooks.distance(model)

# Plot standardized residuals vs leverage
plot(lev, rstandard(model),
     xlab = "Leverage", ylab = "Standardized Residuals",
     main = "Residuals vs Leverage")
abline(h = 0, lty = 2, col = "gray")  # horizontal line at 0

# Cook's distance cutoff
n <- nrow(house_data)
k <- length(coef(model)) - 1
cutoff <- 4/(n - k - 1)

# Add vertical dashed lines for 2x and 3x Cook's cutoff
abline(v = 2*cutoff, lty = 2, col = "red")
abline(v = 3*cutoff, lty = 2, col = "red")

# Optional: Highlight points with high Cook's distance
points(lev[cook > 3*cutoff], rstandard(model)[cook > 3*cutoff],
       col = "red", pch = 19)

############################################################
# 1. Interaction Plot: Size × Location
############################################################

# Aggregate predicted prices by location and size (optional: smooth line)
library(ggplot2)

# Add predicted values from the model
house_data$predicted <- predict(model)

# Plot interaction
ggplot(house_data, aes(x = size, y = predicted, color = location)) +
  geom_point(alpha = 0.5) +                     # individual predicted points
  geom_smooth(method = "lm", se = FALSE) +      # regression lines for each location
  labs(
    title = "Interaction: Size × Location on House Price",
    x = "House Size (m²)",
    y = "Predicted House Price (ZAR)",
    color = "Location"
  ) +
  theme_minimal(base_size = 14) +
  scale_color_manual(values = c("Rural" = "forestgreen", "Urban" = "steelblue")) +
  theme(plot.title = element_text(hjust = 0.5))





