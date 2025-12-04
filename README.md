# WACC Calculator

A professional calculator for computing the Weighted Average Cost of Capital (WACC), available in both web-based and Streamlit versions.

## What is WACC?

WACC (Weighted Average Cost of Capital) represents the average rate a company expects to pay to finance its assets. It's a crucial metric in corporate finance used for investment decisions and company valuation.

### Formula

```
WACC = (E/V × Re) + (D/V × Rd × (1 - Tc))
```

Where:
- **E** = Market Value of Equity
- **D** = Market Value of Debt
- **V** = Total Value (E + D)
- **Re** = Cost of Equity (%)
- **Rd** = Cost of Debt (%)
- **Tc** = Corporate Tax Rate (%)

## Features

- Clean and intuitive user interface
- Real-time WACC calculation
- Detailed breakdown showing:
  - Total company value
  - Equity and debt weights
  - After-tax cost of debt
  - Formula explanation
- Input validation
- Responsive design for mobile and desktop
- Beautiful gradient design

## 🚀 Streamlit Version (Recommended)

### Local Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to `http://localhost:8501`

### Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set Main file path to: `app.py`
7. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

## 🌐 HTML Version

### Usage

1. Open `index.html` in any modern web browser
2. Enter the required values:
   - Market Value of Equity
   - Market Value of Debt
   - Cost of Equity (as a percentage)
   - Cost of Debt (as a percentage)
   - Corporate Tax Rate (as a percentage)
3. Click "Calculate WACC"
4. View your results with a detailed breakdown

## Example

If a company has:
- Equity: $700,000
- Debt: $300,000
- Cost of Equity: 10%
- Cost of Debt: 5%
- Tax Rate: 25%

The WACC would be calculated as:
- Total Value = $1,000,000
- Equity Weight = 70%
- Debt Weight = 30%
- After-tax Cost of Debt = 3.75%
- **WACC = 8.13%**

## Technologies

### Streamlit Version
- Python 3.7+
- Streamlit
- Modern responsive UI

### HTML Version
- Pure HTML5
- CSS3 with modern gradients and animations
- Vanilla JavaScript (no dependencies)

## License

Open source and free to use.
