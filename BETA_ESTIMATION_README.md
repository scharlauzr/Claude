# Beta Estimation Using Comparable Firms

A comprehensive Jupyter notebook for estimating beta using two methods:
1. **Bottom-up beta** using comparable firms
2. **Traditional regression beta** using historical returns

**Target Firm:** Lululemon Athletica (LULU)

## 📋 Project Overview

This project compares two approaches to estimating beta for valuation purposes:

- **Bottom-Up Beta (Comparables Method)**: Identifies comparable firms, unleverages their betas, and then releverages using the target firm's capital structure
- **Regression Beta**: Traditional approach using OLS regression of historical stock returns against market returns

The analysis includes:
- Statistical comparison of both methods
- Uncertainty quantification (standard errors)
- Interactive visualizations showing how sample size affects precision
- Discussion of practical implications for valuation

## 🎯 Learning Objectives

- Use AI (OpenAI API) to identify comparable firms
- Extract and process financial data using APIs (yfinance)
- Understand beta unlevering and relevering formulas
- Perform OLS regression analysis
- Compare estimation methods statistically and visually
- Understand the trade-offs between different beta estimation approaches

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- (Optional) OpenAI API key for comparable firm selection

### Installation

1. **Clone or download this repository**

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Google Colab users:**
   ```python
   # Run this in the first cell
   !pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels openai ipywidgets
   ```

### Running the Notebook

#### Option 1: Local Jupyter
```bash
jupyter notebook beta_estimation_comparables.ipynb
```

#### Option 2: Google Colab
1. Upload `beta_estimation_comparables.ipynb` to Google Colab
2. Install packages in the first code cell
3. Run all cells sequentially

#### Option 3: JupyterLab
```bash
jupyter lab beta_estimation_comparables.ipynb
```

## 📊 Notebook Structure

### Part 1: Setup and Configuration
- Import libraries
- Set target firm ticker and parameters
- Configure API keys (optional)

### Part 2: Get Comparable Firms
- (Optional) Use OpenAI API to identify comparables
- Load standardized list of comparable firms
- Clean and filter the list

### Part 3: Data Collection
- Get financial data for target firm (LULU)
- Loop through comparable firms to collect:
  - Beta (5-year monthly)
  - Market Capitalization
  - Total Debt
- Handle errors and missing data gracefully

### Part 4: Bottom-Up Beta Calculation
- Calculate D/E ratios for each firm
- Unlever betas: `Unlevered Beta = Beta / (1 + D/E * (1 - tax_rate))`
- Relever using target firm's D/E ratio
- Calculate average levered beta (bottom-up beta estimate)

### Part 5: Regression Beta Calculation
- Get 5 years of monthly returns data
- Calculate risk-free rate (10-year Treasury)
- Compute equity market risk premium (EMRP)
- Perform OLS regression: `Excess Return = α + β * EMRP + ε`
- Extract beta coefficient

### Part 6: Comparison and Visualization
- Calculate standard errors for both methods
- Create KDE plots showing distributions
- Build interactive plot with slider for number of comparables
- Statistical comparison of estimates

## 🔑 Key Formulas

### Unlevering Beta
```
Unlevered Beta = Levered Beta / (1 + D/E Ratio * (1 - Tax Rate))
```

### Relevering Beta
```
Levered Beta = Unlevered Beta * (1 + Target D/E Ratio * (1 - Tax Rate))
```

### Standard Error of Bottom-Up Beta
```
SE = σ / √n
```
where σ is the standard deviation of levered betas and n is the number of comparables

### Regression Beta (CAPM)
```
Excess Return = α + β * EMRP + ε
```
where EMRP = Market Return - Risk-Free Rate

## 📈 Key Results

The notebook produces:

1. **Bottom-Up Beta**: Average of relevered comparable firm betas
2. **Regression Beta**: Coefficient from OLS regression
3. **Standard Errors**: Measure of precision for each estimate
4. **KDE Plots**: Visual comparison of beta distributions
5. **Interactive Analysis**: Effect of sample size on precision

## 💡 Discussion Questions

The notebook includes discussion questions:

1. **OpenAI Prompt Design**: Why the prompt is structured to give relevant comparables
2. **Data Collection Strategy**: How the loop collects and handles data
3. **Appropriate Sample Size**: How many comparables are needed for precision
4. **Overall Conclusions**: Insights from comparing both methods

## 🎨 Customization

To use this notebook for a different firm:

1. **Change the target ticker:**
   ```python
   TARGET_TICKER = 'AAPL'  # Change to your desired ticker
   ```

2. **Update the OpenAI prompt** (if using):
   - Modify the `lulu_description` to describe your target firm
   - Run the OpenAI API call to get new comparables

3. **Adjust parameters:**
   ```python
   MARGINAL_TAX_RATE = 0.21  # Adjust if needed
   ```

4. **Run all cells** - the notebook will automatically:
   - Fetch data for the new target firm
   - Get data for comparable firms
   - Perform all calculations
   - Generate visualizations

## 📦 Files Included

- `beta_estimation_comparables.ipynb` - Main Jupyter notebook
- `beta_estimation_comparables.py` - Python script version
- `requirements.txt` - Package dependencies
- `BETA_ESTIMATION_README.md` - This file

## 🔧 Troubleshooting

### Common Issues

**Issue: yfinance not returning data**
- Some tickers (especially non-US) may have limited data
- The notebook handles this with try-except blocks
- You may see "Skipped (missing data)" messages - this is normal

**Issue: OpenAI API errors**
- Ensure you have a valid API key
- The OpenAI section is optional - you can skip it and use the provided list

**Issue: Interactive plots not showing**
- Make sure `ipywidgets` is installed
- In Jupyter Lab, you may need: `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
- In Google Colab, widgets should work automatically

**Issue: Plots not displaying**
- Try adding `%matplotlib inline` at the top of the notebook
- Check that matplotlib is properly installed

## 📚 Background Knowledge

### What is Beta?
Beta (β) measures a stock's systematic risk relative to the market:
- β = 1: Moves with the market
- β > 1: More volatile than the market
- β < 1: Less volatile than the market

### Why Two Methods?

**Bottom-Up Beta:**
- Forward-looking
- Works for private companies
- Can adjust for changes in capital structure
- Less affected by market noise

**Regression Beta:**
- Based on actual historical data
- Standard in finance
- Captures firm-specific market relationship
- May be affected by thin trading or structural changes

## 🎓 Educational Value

This project teaches:
- **Financial Analysis**: Understanding beta and its role in valuation
- **Data Science**: API usage, data cleaning, error handling
- **Statistics**: OLS regression, standard errors, confidence intervals
- **Visualization**: Creating informative financial plots
- **Python**: Pandas, NumPy, Matplotlib, Statsmodels
- **Critical Thinking**: Comparing methodologies and understanding limitations

## 📝 Assignment Deliverables

For Google Colab submission:

1. **Run the entire notebook** from start to finish
2. Ensure all outputs are visible and there are no errors
3. Answer all discussion questions completely
4. **Share via Google Colab** with proper permissions:
   - Click "Share" button in Colab
   - Set permissions to "Anyone with the link can view"
   - Submit the link

**Note:** Ensure permissions are granted to avoid point deductions!

## 🤝 Code Quality Standards

This notebook follows best practices:

✓ Clean and neat organization
✓ No repeated code
✓ Clearly labeled sections
✓ Sufficient comments explaining logic
✓ Proper number formatting (commas, decimal places)
✓ Comprehensive error handling
✓ Reusable design (firm-specific info at top)

## 📄 License

This project is provided for educational purposes.

## 🙏 Acknowledgments

- **yfinance**: For providing free access to financial data
- **OpenAI**: For AI-powered comparable firm selection
- **Pandas/NumPy/Scipy**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For visualization

## 📞 Support

For questions or issues:
1. Check the Troubleshooting section above
2. Review the inline code comments
3. Ensure all packages are properly installed
4. Verify you have internet connection (for data downloads)

---

**Happy Analyzing! 📊**
