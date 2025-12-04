import streamlit as st

# Page configuration
st.set_page_config(
    page_title="WACC Calculator",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    div[data-testid="stMarkdownContainer"] h1 {
        color: white;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.2em;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        color: white;
        text-align: center;
        font-weight: 400;
        margin-bottom: 1em;
    }
    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 25px;
    }
    .result-box {
        background: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
    }
    .result-value {
        font-size: 3em;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }
    .breakdown-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #f0f0f0;
    }
    div[data-testid="stForm"] {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("# 💰 WACC Calculator")
st.markdown("### Weighted Average Cost of Capital")

# Info box
st.markdown("""
<div class="info-box">
    <h4 style="color: #667eea; margin-bottom: 8px;">What is WACC?</h4>
    <p style="color: #555; font-size: 0.9em; line-height: 1.5;">
        WACC represents the average rate a company expects to pay to finance its assets.
        It's calculated as: <strong>WACC = (E/V × Re) + (D/V × Rd × (1 - Tc))</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Create form
with st.form("wacc_form"):
    st.markdown("#### Input Parameters")

    # Input fields
    equity = st.number_input(
        "Market Value of Equity (E)",
        min_value=0.0,
        value=700000.0,
        step=1000.0,
        format="%.2f",
        help="Total market value of the company's equity"
    )

    debt = st.number_input(
        "Market Value of Debt (D)",
        min_value=0.0,
        value=300000.0,
        step=1000.0,
        format="%.2f",
        help="Total market value of the company's debt"
    )

    cost_of_equity = st.number_input(
        "Cost of Equity (Re) %",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.1,
        format="%.2f",
        help="Expected rate of return required by equity investors"
    )

    cost_of_debt = st.number_input(
        "Cost of Debt (Rd) %",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1,
        format="%.2f",
        help="Average interest rate on the company's debt"
    )

    tax_rate = st.number_input(
        "Corporate Tax Rate (Tc) %",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
        format="%.2f",
        help="Corporate tax rate (used for tax shield on debt)"
    )

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        calculate_button = st.form_submit_button("Calculate WACC", use_container_width=True, type="primary")
    with col2:
        reset_button = st.form_submit_button("Reset", use_container_width=True)

# Handle reset
if reset_button:
    st.rerun()

# Calculate WACC
if calculate_button:
    # Validation
    if equity <= 0 and debt <= 0:
        st.error("⚠️ Either equity or debt must be greater than zero.")
    else:
        # Calculate total value
        total_value = equity + debt

        # Calculate weights
        equity_weight = equity / total_value
        debt_weight = debt / total_value

        # Calculate after-tax cost of debt
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate / 100)

        # Calculate WACC
        wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)

        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Results")

        # Main WACC result
        st.markdown(f"""
        <div class="result-box">
            <h3 style="color: #333; margin-bottom: 10px;">Your WACC is:</h3>
            <div class="result-value">{wacc:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown
        st.markdown("#### 📊 Detailed Breakdown")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Value (V)", f"${total_value:,.2f}")
            st.metric("Equity Weight (E/V)", f"{equity_weight * 100:.2f}%")
        with col2:
            st.metric("Debt Weight (D/V)", f"{debt_weight * 100:.2f}%")
            st.metric("After-Tax Cost of Debt", f"{after_tax_cost_of_debt:.2f}%")

        # Formula explanation
        with st.expander("📖 How was this calculated?"):
            st.markdown(f"""
            **Formula:** WACC = (E/V × Re) + (D/V × Rd × (1 - Tc))

            **Calculation:**
            - E/V = ${equity:,.2f} / ${total_value:,.2f} = {equity_weight:.4f}
            - D/V = ${debt:,.2f} / ${total_value:,.2f} = {debt_weight:.4f}
            - After-tax Rd = {cost_of_debt:.2f}% × (1 - {tax_rate:.2f}%) = {after_tax_cost_of_debt:.2f}%

            **WACC = ({equity_weight:.4f} × {cost_of_equity:.2f}%) + ({debt_weight:.4f} × {after_tax_cost_of_debt:.2f}%) = {wacc:.2f}%**
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: white; padding: 20px; font-size: 0.9em;">
    <p>💡 WACC is used in DCF valuation models as the discount rate for future cash flows.</p>
    <p style="opacity: 0.8;">Built with Streamlit | Open Source</p>
</div>
""", unsafe_allow_html=True)
