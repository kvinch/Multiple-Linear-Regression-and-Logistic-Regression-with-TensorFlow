import sys

try:
    print("Cargando librerías...")
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson
    
    print("Todas las librerías fueron importadas correctamente.\n")

except ImportError as e:
    print("\nNo se pudo importar una librería necesaria.")
    sys.exit(1)

# ==========================================
# 1. DATA LOADING AND CLEANING
# ==========================================
print("--- Data Loading ---")
data = pd.read_csv("enhanced_student_habits_performance_dataset.csv")

# Remove student_id
if 'student_id' in data.columns:
    data = data.drop(columns=['student_id'])

# Define categorical and numeric columns
# In R you used as.factor(). In Python, we identify them for OneHotEncoding.
categorical_cols = [
    'gender', 'major', 'part_time_job', 'diet_quality', 
    'parental_education_level', 'internet_quality', 
    'extracurricular_participation', 'study_environment', 
    'access_to_tutoring', 'learning_style', 'dropout_risk', 'family_income_range'
]

# Identify numeric columns (all columns that are not categorical and not the target)
target_linear = 'exam_score'
numeric_cols = [col for col in data.columns if col not in categorical_cols and col != target_linear]

# Convert dropout_risk to binary (0 and 1) for Logistic Regression
# Assuming "Yes"/"No" or similar text. If already numeric, this line can be skipped.
if data['dropout_risk'].dtype == 'object':
    data['dropout_risk'] = data['dropout_risk'].astype('category').cat.codes

# ==========================================
# 2. DATA PREPROCESSING (PIPELINE)
# ==========================================
print("Data types:")
print(data.dtypes)
print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)
# TensorFlow requires numerical inputs. We use a ColumnTransformer.
preprocessor = ColumnTransformer(
    transformers=[
        # Standardize numeric variables
        ('num', StandardScaler(), numeric_cols), 
        # Convert categories to dummy variables (drop first to avoid multicollinearity)
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), [c for c in categorical_cols if c != 'dropout_risk']) 
    ])

# ==========================================
# 3. MULTIPLE LINEAR REGRESSION (TensorFlow)
# R Equivalent: ModeloCompleto <- lm(exam_score ~ ..., data = datos)
# ==========================================
print("\n--- Linear Regression with TensorFlow ---")

X = data.drop(columns=[target_linear, 'dropout_risk']) # Independent variables
y = data[target_linear] # Dependent variable

# Process X
X_processed = preprocessor.fit_transform(X)
# Ensure it is a dense array for TensorFlow
X_processed = X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed

# Define the model in TensorFlow (1 dense layer = Linear Regression)
model_linear = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_processed.shape[1],)),
    tf.keras.layers.Dense(1, activation='linear') # Linear activation for regression
])

model_linear.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     loss='mse', # Mean Squared Error
                     metrics=['mae'])

# Train the model
history = model_linear.fit(X_processed, y, epochs=100, verbose=0)
print("Linear Model Trained.")

# --- DIAGNOSTICS (Replicating R plots/tests) ---

# Generate Predictions and Residuals
predictions = model_linear.predict(X_processed).flatten()
residuals = y - predictions

# 1. Normality Test (Anderson-Darling, similar to ad.test in R)
print("\nNormality Test (Anderson-Darling):")
result = stats.anderson(residuals)
print(f'Statistic: {result.statistic}, Critical Values: {result.critical_values}')

# 2. QQ Plot
plt.figure(figsize=(6,4))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("QQ Plot (Residuals)")
plt.show()

# 3. VIF (Variance Inflation Factor)
# Note: VIF is calculated on the design matrix (X_processed)
print("\nVIF (First 5 variables as example):")
vif_data = [variance_inflation_factor(X_processed, i) for i in range(min(5, X_processed.shape[1]))]
print(vif_data)

# 4. Breusch-Pagan Test (Homoscedasticity)
# We need to add a constant for statsmodels
X_with_const = sm.add_constant(X_processed)
bp_test = het_breuschpagan(residuals, X_with_const)
print(f"\nBreusch-Pagan p-value: {bp_test[1]}")

# 5. Residuals vs Predictions Plot
plt.figure(figsize=(6,4))
plt.scatter(predictions, residuals, alpha=0.5, color='grey')
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residuals vs Predictions')
plt.show()

# 6. Independence Test (Durbin-Watson)
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson statistic: {dw_stat}")


# ==========================================
# 4. LOGISTIC REGRESSION (TensorFlow)
# R Equivalent: glm(dropout_risk ~ ..., family = binomial)
# ==========================================
print("\n--- Logistic Regression with TensorFlow ---")

# Replicating your 'Final Model' (modelo_final2):
# dropout_risk ~ stress_level + parental_education_level + motivation_level

cols_of_interest = ['stress_level', 'parental_education_level', 'motivation_level']
X_log = data[cols_of_interest]
y_log = data['dropout_risk'] # Target (0 or 1)

# Specific preprocessor for these columns
preprocessor_log = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['stress_level', 'motivation_level']),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ['parental_education_level'])
    ])

X_log_proc = preprocessor_log.fit_transform(X_log)
if hasattr(X_log_proc, "toarray"):
    X_log_proc = X_log_proc.toarray()

# Logistic Model in TensorFlow (Sigmoid Activation)
model_logistic = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_log_proc.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for probability (0-1)
])

model_logistic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

# Train
model_logistic.fit(X_log_proc, y_log, epochs=100, verbose=0)
print("Logistic Model Trained.")

# --- ODDS RATIOS ---
weights, bias = model_logistic.layers[0].get_weights()
odds_ratios = np.exp(weights)
print("\nOdds Ratios (Approximated from TF weights):")
print(odds_ratios.flatten())

# --- PSEUDO R2 (McFadden) ---
# Manual calculation as TF does not provide this natively
def log_likelihood(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y_pred_probs = model_logistic.predict(X_log_proc).flatten()
ll_model = log_likelihood(y_log, y_pred_probs)

# Null model (predicts only the mean)
mean_y = np.mean(y_log)
ll_null = log_likelihood(y_log, np.full_like(y_log, mean_y))

pseudo_r2 = 1 - (ll_model / ll_null)
print(f"Pseudo R2 (McFadden): {pseudo_r2}")


# ==========================================
# 5. PREDICTIONS (New Students)
# ==========================================
print("\n--- Predictions for New Students ---")

# Student 1
student1 = pd.DataFrame({
    'stress_level': [8.5],
    'parental_education_level': ["Master"],
    'motivation_level': [7]
})

# Student 2
student2 = pd.DataFrame({
    'stress_level': [9.7],
    'parental_education_level': ["Some College"],
    'motivation_level': [3]
})

try:
    # Important: Use the FITTED preprocessor (transform, NOT fit_transform)
    X_new1 = preprocessor_log.transform(student1)
    if hasattr(X_new1, "toarray"): X_new1 = X_new1.toarray()
    
    X_new2 = preprocessor_log.transform(student2)
    if hasattr(X_new2, "toarray"): X_new2 = X_new2.toarray()

    pred1 = model_logistic.predict(X_new1)
    pred2 = model_logistic.predict(X_new2)

    print(f"Dropout Probability Student 1: {pred1[0][0]:.4f}")
    print(f"Dropout Probability Student 2: {pred2[0][0]:.4f}")

except Exception as e:
    print(f"Prediction Error (Likely due to unseen categories): {e}")