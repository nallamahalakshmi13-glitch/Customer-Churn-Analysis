# Customer-Churn-Analysis
Customer Churn Analysis Analyze customer data to identify patterns behind subscription cancellations. Evaluate user activity, engagement levels, and behavioral trends to determine key factors 
from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# Sample dataset
sample_data = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 105],
    "tenure": [2, 24, 5, 36, 8],
    "monthly_charges": [70, 55, 80, 45, 90],
    "contract": ["Monthly", "Yearly", "Monthly",
                 "Yearly", "Monthly"],
    "churn": ["Yes", "No", "Yes", "No", "Yes"]
})


# Dashboard HTML
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Customer Churn Analysis</title>

<style>
body {
    font-family: Arial;
    background: #f1f5f9;
    margin: 0;
}

header {
    background: #172554;
    color: white;
    padding: 25px;
    text-align: center;
}

.container {
    max-width: 950px;
    margin: 30px auto;
    padding: 15px;
}

.card {
    background: white;
    padding: 25px;
    margin-bottom: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 10px #00000015;
}

input, button {
    padding: 12px;
    margin: 10px 0;
}

button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
}

.stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
}

.stat {
    background: #dbeafe;
    padding: 15px;
    text-align: center;
    border-radius: 8px;
}

.stat h2 {
    color: #1d4ed8;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: center;
}

th {
    background: #dbeafe;
}

.error {
    color: red;
}

</style>
</head>

<body>

<header>
    <h1>Customer Churn Analysis</h1>
    <p>Analyze Customer Behavior and Retention</p>
</header>

<div class="container">

    <div class="card">
        <h2>Upload Customer Dataset</h2>

        <form method="POST" enctype="multipart/form-data">

            <input type="file" name="file"
                   accept=".csv">

            <br>

            <button type="submit">
                Analyze Customers
            </button>

        </form>

        <p>Upload a CSV file or use the sample dataset.</p>
    </div>

    {% if error %}
    <div class="card error">
        {{ error }}
    </div>
    {% endif %}

    {% if result %}

    <div class="card">
        <h2>Analysis Dashboard</h2>

        <div class="stats">

            <div class="stat">
                <p>Total Customers</p>
                <h2>{{ result.total }}</h2>
            </div>

            <div class="stat">
                <p>Churned Customers</p>
                <h2>{{ result.churned }}</h2>
            </div>

            <div class="stat">
                <p>Active Customers</p>
                <h2>{{ result.active }}</h2>
            </div>

            <div class="stat">
                <p>Churn Rate</p>
                <h2>{{ result.rate }}%</h2>
            </div>

        </div>
    </div>

    <div class="card">
        <h2>Churn by Contract</h2>

        <table>
            <tr>
                <th>Contract</th>
                <th>Churned Customers</th>
            </tr>

            {% for row in result.contracts %}
            <tr>
                <td>{{ row.contract }}</td>
                <td>{{ row.count }}</td>
            </tr>
            {% endfor %}

        </table>
    </div>

    <div class="card">
        <h2>Customer Behavior</h2>

        <p>Average tenure: {{ result.tenure }} months</p>

        <p>Average monthly charges:
           {{ result.charges }}</p>

        <p>Average tenure of churned customers:
           {{ result.churn_tenure }} months</p>
    </div>

    <div class="card">
        <h2>Retention Suggestions</h2>

        <ul>
            <li>Offer discounts to customers with monthly contracts.</li>
            <li>Provide loyalty rewards to long-term customers.</li>
            <li>Contact customers with low engagement.</li>
            <li>Improve customer support services.</li>
            <li>Provide special offers to customers at risk of leaving.</li>
        </ul>
    </div>

    {% endif %}

</div>

</body>
</html>
"""


# Customer analysis function
def analyze_data(df):

    # Standardize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    required = [
        "tenure",
        "monthly_charges",
        "contract",
        "churn"
    ]

    # Validate columns
    for col in required:
        if col not in df.columns:
            raise ValueError("Missing column: " + col)

    # Remove duplicate records
    df = df.drop_duplicates().copy()

    # Convert numeric columns
    df["tenure"] = pd.to_numeric(
        df["tenure"], errors="coerce"
    )

    df["monthly_charges"] = pd.to_numeric(
        df["monthly_charges"], errors="coerce"
    )

    # Remove rows with invalid essential values
    df = df.dropna(
        subset=["tenure", "monthly_charges", "churn"]
    ).copy()

    if df.empty:
        raise ValueError("No valid customer records found.")

    if (df["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative.")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("Monthly charges cannot be negative.")

    # Standardize text
    df["contract"] = df["contract"].astype(str).str.strip().str.title()

    df["churn"] = df["churn"].astype(str).str.strip().str.title()

    if not df["churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn values must be Yes or No.")

    # Calculate results
    total = len(df)

    churned = len(df[df["churn"] == "Yes"])

    active = len(df[df["churn"] == "No"])

    rate = round((churned / total) * 100, 2)

    # Contract analysis
    contract_data = df[df["churn"] == "Yes"]

    contracts = (
        contract_data.groupby("contract")
        .size()
        .reset_index(name="count")
        .to_dict(orient="records")
    )

    # Average values
    tenure = round(df["tenure"].mean(), 2)

    charges = round(df["monthly_charges"].mean(), 2)

    churn_df = df[df["churn"] == "Yes"]

    churn_tenure = (
        round(churn_df["tenure"].mean(), 2)
        if not churn_df.empty else 0
    )

    return {
        "total": total,
        "churned": churned,
        "active": active,
        "rate": rate,
        "contracts": contracts,
        "tenure": tenure,
        "charges": charges,
        "churn_tenure": churn_tenure
    }


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    if request.method == "POST":

        try:

            file = request.files.get("file")

            if file and file.filename:

                if not file.filename.lower().endswith(".csv"):
                    raise ValueError("Please upload a CSV file.")

                df = pd.read_csv(file)

            else:

                df = sample_data.copy()

            result = analyze_data(df)

        except Exception as e:

            error = str(e)

    return render_template_string(
        HTML,
        result=result,
        error=error
    )


# Run application
if __name__ == "__main__":
    app.run(debug=True)
that lead to churn and suggest improvements for retention.
