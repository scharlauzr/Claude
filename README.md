# Financial Calculators Suite

A collection of professional web-based financial calculators for investment analysis and company valuation.

## Calculators Included

1. **DCF Calculator** - Discounted Cash Flow valuation using Yahoo Finance data
2. **WACC Calculator** - Weighted Average Cost of Capital computation

---

## DCF Calculator

A comprehensive DCF (Discounted Cash Flow) calculator that integrates with Yahoo Finance to fetch real-time financial data and calculate intrinsic stock value.

### What is DCF?

DCF is a valuation method that estimates the intrinsic value of a company by projecting its future free cash flows and discounting them to present value. This calculator helps investors determine whether a stock is overvalued or undervalued.

### Features

- **Yahoo Finance Integration**: Automatically fetches real financial data using stock tickers
- **Real-time Data**: Current price, market cap, free cash flow, and shares outstanding
- **5-Year Projections**: Detailed cash flow projections with customizable growth rates
- **Terminal Value Calculation**: Perpetual growth model for long-term valuation
- **Margin of Safety**: Automatic calculation comparing intrinsic value to market price
- **Investment Recommendations**: Buy/Hold/Sell signals based on margin of safety
- **Professional UI**: Clean, responsive design with detailed breakdowns

### Usage

1. Open `dcf.html` in any modern web browser
2. Enter a stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
3. Click "Fetch Data" to retrieve financial information from Yahoo Finance
4. Review and adjust the pre-filled values:
   - Initial Free Cash Flow (automatically populated)
   - Growth Rate for next 5 years (default: 10%)
   - Terminal Growth Rate (default: 2.5%)
   - Discount Rate/WACC (default: 10%)
   - Shares Outstanding (automatically populated)
5. Click "Calculate DCF" to see:
   - Enterprise value
   - Intrinsic value per share
   - Margin of safety vs. current price
   - Investment recommendation
   - Detailed 5-year cash flow projections

### Example

For Apple (AAPL):
- Ticker: AAPL
- Initial FCF: $100B (fetched automatically)
- Growth Rate: 8%
- Terminal Growth: 2.5%
- Discount Rate: 10%
- Current Price: $180
- **Calculated Intrinsic Value**: $195
- **Margin of Safety**: 8.3% (Buy recommendation)

---

## WACC Calculator

### What is WACC?

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
- Input validation
- Responsive design for mobile and desktop
- Beautiful gradient design

### Usage

1. Open `wacc.html` in any modern web browser
2. Enter the required values:
   - Market Value of Equity
   - Market Value of Debt
   - Cost of Equity (as a percentage)
   - Cost of Debt (as a percentage)
   - Corporate Tax Rate (as a percentage)
3. Click "Calculate WACC"
4. View your results with a detailed breakdown

### Example

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

---

## Technologies

- Pure HTML5
- CSS3 with modern gradients and animations
- Vanilla JavaScript (no dependencies)
- Yahoo Finance API integration

## File Structure

```
├── index.html          # Landing page with links to both calculators
├── dcf.html           # DCF Calculator with Yahoo Finance integration
├── wacc.html          # WACC Calculator
└── README.md          # This file
```

## Getting Started

1. Clone or download this repository
2. Open `index.html` in your web browser to access both calculators
3. No installation or build process required!

## Notes

- The DCF calculator requires an internet connection to fetch data from Yahoo Finance
- The WACC calculator works completely offline
- Both calculators work on desktop and mobile devices
- No personal data is collected or stored

## License

Open source and free to use.
