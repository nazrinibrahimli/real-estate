
# Istanbul Real Estate Intelligence System (2019–2026)
**Developer:** Nazrin Ibrahimli  

## Project Overview
This repository hosts a multi-stage Machine Learning and Time-Series pipeline developed to analyze, segment, and project residential property values within Istanbul's volatile, high-inflation real estate market. The project transitions seamlessly from spatial asset valuation to strategic investment classification, concluding with macro-temporal forecasting.

The pipeline resolves critical statistical boundaries across three consecutive tasks:
1. **Property Valuation (Price Prediction):** A tree-based ensemble regressor resolving point-in-time fair values (R-squared = 0.87$).
2. **Strategic Tiering (Classification):** A gradient-boosted classification layer mapping assets into discrete risk profiles with a 97% accuracy rate.
3. **Future Growth Forecasting (Time-Series):** A log-transformed trend engine isolating market velocity to project asset pricing for the April 2026 window.

---

## Pipeline Architecture & Implementation Details

### Stage 1: Property Valuation Engine (Regression)
* **Dataset Foundation:** ~50,000 listings mapped against critical spatial-structural features including `Building Age`, `Room Sizes`, `District`, `Size`, `Floor Number`, and `Inflation_Rate`.
* **Overfitting & Variance Diagnosis:** Initial baseline modeling with an unconstrained tree architecture (depth of 20) introduced an severe **32% Generalization Gap** (Training R-squared: 0.8991 vs. Testing R-squared: 0.5783). This high-variance state was driven by data sparsity in the 2025 dimension—where empty fields imputed as "0" generated false signals—and hyper-complex paths memorizing individual luxury listings.
* **The Solution:** Implemented a log-transformation (`np.log1p`) on the target pricing column to shift optimization from absolute error to percentage variance, alongside structural constraints on tree depth. 
* **Final Performance:** Achieved a stabilized, verified **Testing R-squared of 0.87**, where training (0.8785) and testing (0.8716) scores converged symmetrically, confirming true generalization of market mechanics.
* **Feature Drivers:** Feature importance vectors proved that `District_Score` (0.329) and `Inflation_Rate` (0.310) dictated the largest relative pricing contributions, closely followed by asset layout dynamics.

### Stage 2: Strategic Tiering Layer (Classification)
* **Objective:** Group properties cleanly into three strategic tranches:
  * **Value Tier:** High-yield, entry-level assets.
  * **Core Tier:** Stable, mid-range family housing.
  * **Premium Tier:** High-capital, luxury assets.
* **Dataset Balancing:** Avoided synthetic up-sampling vulnerabilities (e.g., SMOTE) by processing target splits through **Quantile-based binning**. This forced an exact 33.3% distribution across all three tranches, preventing class bias.
* **Feature Engineering Breakthrough:** Standard feature inputs baseline models stalled at a 67% accuracy ceiling due to an inability to interpret fluid classification borderlines between Value and Core sectors. To inject contextual intelligence, two specialized interaction terms were engineered:
  * `Space_Per_Room` ($$\text{Metrekare} / \text{Total Rooms}$$): Captures spatial asset density to effectively isolate premium open plans from dense, low-tier layouts.
  * `Location_Size_Score` ($$\text{Metrekare} \times \text{District Score}$$): Weights raw footprint scale directly against structural geographical prestige.
* **Algorithm Selection:** Gradient Boosting (XGBoost) was chosen over Random Forest for final production due to its iterative correction of error bounds across fuzzy transition zones. Built-in L1 (Lasso) and L2 (Ridge) penalties acted as a essential mathematical brake against over-indexing.
* **Validation Performance:** Achieved near-flawless diagonal classification with a **97% overall accuracy / F1-Score**. During **5-Fold Cross-Validation**, the pipeline demonstrated supreme stability, tracking a mean accuracy of **95%** with a minimal standard deviation of **0.04**.

### Stage 3: Future Growth Forecasting (Time-Series)
* **The Challenge:** Compounding macro growth across longitudinal market samples (2019–2025) induces heavy exponential residuals when processed through basic arithmetic models.
* **Methodology:** Applied a log-linear approach to transform exponential inflation curves into flat, calculable linear slopes, altering the primary loss objective to minimize relative percentage errors.
* **Model Selection Matrix:** Four competitive architectures were built and examined under strict guardrails for economic plausibility:

| Architecture Model | R-squared Metric | MAPE % | Projected April 2026 Target | Production Selection Status |
| :--- | :--- | :--- | :--- | :--- |
| 1. Simple Linear | 0.9950 | 14.72% | 8,434,927 TL | Rejected (Under-indexed momentum) |
| 2. Polynomial (Deg 2) | 1.0000 | 0.00% | 14,377,640 TL | **Rejected (Extreme Overfitting)** |
| 3. **Weighted Linear** | **0.9973** | **13.10%** | **8,919,825 TL** | **SELECTED FOR PRODUCTION** |
| 4. Weighted Poly | 1.0000 | 0.00% | 14,377,640 TL | **Rejected (Extreme Overfitting)** |

* **The Polynomial Paradox:** While polynomial configurations returned deceptively flawless curves (R-squared = 1.0000$), they were systematically rejected. Due to narrow historical temporal resolution, the curves hallucinated an aggressive, economically unviable exponential spike (14.3M TL) out of phase with realistic macro parameters.
* **Operational Strategy:** The finalized **Weighted Linear Regression** engine prioritizes recent inflation-driven momentum (2024–2025) via a decay weight matrix. It uncovers a concrete **1.47x macro growth coefficient ($G$)**. Rather than assigning a flat projection value universally, this coefficient scales individual property baselines derived from the Stage 1 spatial engine forward through time:
$$\text{Future Price}_{2026} = \text{Predicted Spatial Baseline}_{2025} \times 1.47$$---

