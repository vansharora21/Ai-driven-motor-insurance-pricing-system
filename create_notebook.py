import nbformat as nbf


def create_notebook():
    nb = nbf.v4.new_notebook()

    text_intro = """# freMTPL2 Motor Insurance Pricing Analysis
This notebook walks through the production-grade training flow using the real freMTPL2 frequency and severity datasets."""

    code_imports = """from sklearn.model_selection import train_test_split

from src.data_loader import prepare_model_datasets
from src.frequency_model import train_frequency_model, predict_claim_count, evaluate_frequency_model
from src.pricing_engine import DEFAULT_PRICING_CONFIG, calculate_premium, compute_risk_thresholds
from src.severity_model import train_severity_model, predict_severity, evaluate_severity_model"""

    code_load = """policy_df, claim_df, data_quality = prepare_model_datasets()
policy_df.head()"""

    code_split = """frequency_train, frequency_test = train_test_split(
    policy_df,
    test_size=0.2,
    random_state=42,
    stratify=policy_df['has_claim'],
)
severity_train, severity_test = train_test_split(
    claim_df,
    test_size=0.2,
    random_state=42,
)"""

    code_train = """frequency_model = train_frequency_model(frequency_train)
severity_model = train_severity_model(severity_train)

frequency_metrics = evaluate_frequency_model(frequency_model, frequency_test)
severity_metrics = evaluate_severity_model(severity_model, severity_test)

frequency_metrics, severity_metrics"""

    code_score = """portfolio_frequency = predict_claim_count(frequency_model, policy_df) / policy_df['Exposure']
portfolio_severity = predict_severity(severity_model, policy_df)

preliminary = calculate_premium(
    policy_df,
    portfolio_frequency,
    portfolio_severity,
    pricing_config=DEFAULT_PRICING_CONFIG,
)
risk_thresholds = compute_risk_thresholds(preliminary['pure_premium'])

scored_portfolio = calculate_premium(
    policy_df,
    portfolio_frequency,
    portfolio_severity,
    pricing_config=DEFAULT_PRICING_CONFIG,
    risk_thresholds=risk_thresholds,
)

scored_portfolio[['IDpol', 'predicted_claim_count', 'predicted_claim_severity', 'final_premium', 'risk_category']].head()"""

    nb["cells"] = [
        nbf.v4.new_markdown_cell(text_intro),
        nbf.v4.new_code_cell(code_imports),
        nbf.v4.new_code_cell(code_load),
        nbf.v4.new_code_cell(code_split),
        nbf.v4.new_code_cell(code_train),
        nbf.v4.new_code_cell(code_score),
    ]

    with open("notebooks/analysis.ipynb", "w", encoding="utf-8") as notebook_file:
        nbf.write(nb, notebook_file)
    print("Notebook 'notebooks/analysis.ipynb' created successfully.")


if __name__ == "__main__":
    create_notebook()
