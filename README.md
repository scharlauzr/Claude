# WACC Calculator

A professional web-based calculator for computing the Weighted Average Cost of Capital (WACC).

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
- Input validation
- Responsive design for mobile and desktop
- Beautiful gradient design

## Usage

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

- Pure HTML5
- CSS3 with modern gradients and animations
- Vanilla JavaScript (no dependencies)

## License

Open source and free to use.
