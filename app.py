import os
import pandas as pd
import numpy as np
import joblib
import pickle as pkl
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Load model and categories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
FOREST_PATH = os.path.join(BASE_DIR, "best_random_forest_model.pkl")
CATEGORIES_PATH = os.path.join(BASE_DIR, "categories.joblib")

if not os.path.exists(MODEL_PATH) or not os.path.exists(CATEGORIES_PATH):
    raise FileNotFoundError("Model or categories mapping file is missing. Please run train_model.py first.")

model = joblib.load(MODEL_PATH)
forest_model = pkl.load(open(FOREST_PATH, "rb"))
categories = joblib.load(CATEGORIES_PATH)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/categories', methods=['GET'])
def get_categories():
    return jsonify(categories)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Expected columns order matching the model training features
        columns = [
            "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
            "workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"
        ]

        # Build a dictionary to construct a DataFrame row
        row = {}
        for col in columns:
            val = data.get(col)
            if col in ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
                # Cast to float
                row[col] = float(val) if val is not None and str(val).strip() != "" else 0.0
            else:
                # Cast to str
                row[col] = str(val).strip() if val is not None else ""

        input_df = pd.DataFrame([row])

        # Predict binary target (>50K or <=50K)
        pred_class = int(model.predict(input_df)[0])
        pred_prob = float(model.predict_proba(input_df)[0][1]) # probability of earning >50k
        forest_pred_prob = float(forest_model.predict_proba(input_df)[0][1]) # probability of earning >50k from random forest

        if forest_pred_prob > pred_prob:
            pred_prob = forest_pred_prob

        # Heuristic for continuous annual income estimation
        # We base it on:
        # - XGBoost probability (which is the main driver)
        # - Age, education years, hours worked, occupation and workclass.
        age = row["age"]
        edu_num = row["education_num"]
        hours = row["hours_per_week"]
        cap_gain = row["capital_gain"]
        cap_loss = row["capital_loss"]
        occ = row["occupation"]
        wc = row["workclass"]

        # Profile-based heuristic income
        profile_income = 15000
        profile_income += min(max(0, age - 17) * 350, 18000)  # age factor
        profile_income += edu_num * 1800                      # education factor
        profile_income += (hours - 20) * 500                  # hours worked factor

        # Occupation factor
        high_pay_occ = ["Exec-managerial", "Prof-specialty"]
        med_pay_occ = ["Tech-support", "Sales", "Protective-serv"]
        low_pay_occ = ["Craft-repair", "Adm-clerical", "Machine-op-inspct", "Transport-moving"]
        if occ in high_pay_occ:
            profile_income += 15000
        elif occ in med_pay_occ:
            profile_income += 8000
        elif occ in low_pay_occ:
            profile_income += 4000

        # Workclass factor
        if wc in ["Self-emp-inc", "Federal-gov"]:
            profile_income += 10000
        elif wc in ["Local-gov", "State-gov", "Self-emp-not-inc"]:
            profile_income += 4000

        # Combine XGBoost probability and profile income
        if pred_prob <= 0.5:
            # <=50K range ($12,000 to $50,000)
            estimated_income = 12000 + (38000 * (pred_prob / 0.5)) * 0.75 + (profile_income * 0.25)
            estimated_income = min(estimated_income, 50000)
        else:
            # >50K range ($50,000 to $180,000)
            prob_factor = (pred_prob - 0.5) / 0.5  # ranges from 0 to 1
            estimated_income = 50000 + (130000 * prob_factor) * 0.7 + (profile_income * 0.3)
            estimated_income = max(50000, estimated_income)

        # Adjust for capital gains and capital losses
        estimated_income += cap_gain * 0.8
        estimated_income -= cap_loss * 0.8

        # Enforce logical bounds
        estimated_income = max(10000, estimated_income)
        # Limit max income to $250,000 unless capital gains are massive
        if cap_gain <= 50000:
            estimated_income = min(250000, estimated_income)

        # Round to nearest hundred
        estimated_income = round(estimated_income, -2)

        # Define the income tier
        if estimated_income < 25000:
            tier = "Starter"
            tier_color = "red"
            tier_name = "Horizon Starter Account"
        elif estimated_income < 50000:
            tier = "Standard"
            tier_color = "gray"
            tier_name = "Horizon Standard Account"
        elif estimated_income < 100000:
            tier = "Premium"
            tier_color = "gold"
            tier_name = "Horizon Premium Advantage Account"
        else:
            tier = "Elite"
            tier_color = "black"
            tier_name = "Horizon Elite Private Account"

        # Prepare products list based on the tier
        products = get_products_for_tier(tier, estimated_income)

        return jsonify({
            "prediction_class": pred_class,
            "prediction_probability": pred_prob,
            "estimated_income": estimated_income,
            "tier": tier,
            "tier_name": tier_name,
            "tier_color": tier_color,
            "products": products
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

def get_products_for_tier(tier, income):
    if tier == "Starter":
        return [
            {
                "type": "Credit Card",
                "name": "Apex Horizon Starter Card",
                "limit": "$1,000 Limit",
                "rate": "22.9% Variable APR",
                "features": [
                    "No annual fee",
                    "Build credit score history with automatic monthly bureau reporting",
                    "1% cashback on essential grocery & utility utility payments"
                ],
                "button_text": "Apply to Build Credit"
            },
            {
                "type": "Personal Loan",
                "name": "Apex Starter Loan",
                "limit": "Up to $3,000",
                "rate": "10.9% Fixed APR",
                "features": [
                    "Flexible repayment terms from 12 to 24 months",
                    "No early repayment penalties",
                    "Fast-track decision within 2 hours on mobile app"
                ],
                "button_text": "Check Eligibility"
            },
            {
                "type": "Savings Account",
                "name": "Apex Smart Savings Account",
                "limit": "No minimum deposit",
                "rate": "3.50% Annual APY",
                "features": [
                    "No monthly maintenance or transaction fees",
                    "Automated card purchase round-ups to build savings",
                    "Instant withdrawals via the online portal"
                ],
                "button_text": "Open Savings Account"
            },
            {
                "type": "Financial Coaching",
                "name": "Apex Smart Budget Advisor",
                "limit": "Included for Free",
                "rate": "0% Fees",
                "features": [
                    "Personalized monthly spending analysis & feedback",
                    "Real-time alerts for recurring bills and subscriptions",
                    "Savings goal tracker with smart automated deposits"
                ],
                "button_text": "Activate Smart Advisor"
            }
        ]
    elif tier == "Standard":
        return [
            {
                "type": "Credit Card",
                "name": "Apex Classic Visa Credit Card",
                "limit": "$5,000 Limit",
                "rate": "17.9% Variable APR",
                "features": [
                    "$0 annual fee for the first year (then $29/year)",
                    "1.5% cashback on all dining, entertainment & gas",
                    "Purchase protection insurance up to 90 days from theft or damage"
                ],
                "button_text": "Apply Now"
            },
            {
                "type": "Personal Loan",
                "name": "Apex Standard Personal Loan",
                "limit": "Up to $15,000",
                "rate": "7.9% Fixed APR",
                "features": [
                    "Flexible repayment terms from 12 to 60 months",
                    "Debt consolidation option to merge credit balances",
                    "Funds disbursed to your checking account within 24 hours"
                ],
                "button_text": "Estimate Payments"
            },
            {
                "type": "Savings & Investments",
                "name": "Apex Growth Mutual Funds",
                "limit": "Min investment $100",
                "rate": "Target Return 6-8%",
                "features": [
                    "Diversified fund actively managed by Apex portfolio experts",
                    "Low management fees (0.45% MER)",
                    "Automatic monthly investing plans starting at $25"
                ],
                "button_text": "Explore Portfolios"
            },
            {
                "type": "Auto Loan",
                "name": "Apex FlexCar Loan",
                "limit": "Up to $30,000",
                "rate": "5.49% Fixed APR",
                "features": [
                    "Available for both new and used vehicles",
                    "Up to 84-month terms for affordable monthly payments",
                    "Pre-approval certificate valid for 60 days of shopping"
                ],
                "button_text": "Get Pre-Approved"
            }
        ]
    elif tier == "Premium":
        return [
            {
                "type": "Credit Card",
                "name": "Apex Gold Premium MasterCard",
                "limit": "$25,000 Limit",
                "rate": "14.9% Variable APR",
                "features": [
                    "Annual fee waived with $15,000 annual spend",
                    "2.0% cashback on all travel and retail shopping",
                    "Premium airport lounge access (4 complimentary entries/year)",
                    "Comprehensive international travel medical & delay cover"
                ],
                "button_text": "Apply for Gold Tier"
            },
            {
                "type": "Personal Loan",
                "name": "Apex Premium Flexi-Loan",
                "limit": "Up to $50,000",
                "rate": "5.49% Fixed APR",
                "features": [
                    "Repayment terms up to 84 months",
                    "0.25% interest rate discount for active banking accounts",
                    "Skip up to 2 monthly payments per year with no penalty"
                ],
                "button_text": "Apply Online"
            },
            {
                "type": "Savings & Investments",
                "name": "Apex Balanced Portfolio Plus",
                "limit": "Min investment $5,000",
                "rate": "Target Return 8-10%",
                "features": [
                    "Active asset allocation across global equities and bonds",
                    "Quarterly rebalancing and detailed market insights",
                    "Tax-efficient savings shells (ISA/IRA compatible)"
                ],
                "button_text": "Speak to Advisor"
            },
            {
                "type": "Home Mortgage",
                "name": "Apex Premium Home Buyer Loan",
                "limit": "Up to $650,000",
                "rate": "4.25% Fixed 30yr",
                "features": [
                    "Low 5% down payment options available",
                    "No lender origination fees for Premium customers",
                    "Fast-tracked digital appraisal and closing process"
                ],
                "button_text": "View Mortgage Rates"
            }
        ]
    else:  # Elite
        return [
            {
                "type": "Credit Card",
                "name": "Apex Elite World Elite Visa Infinite",
                "limit": "$75,000+ Limit",
                "rate": "11.9% Variable APR",
                "features": [
                    "Dedicated 24/7 lifestyle & travel concierge service",
                    "Unlimited premium airport lounge access globally for you + 2 guests",
                    "5.0% rewards on airlines, hotels & Michelin-star dining",
                    "Complimentary multi-hazard travel & luxury asset protection"
                ],
                "button_text": "Request Invitation"
            },
            {
                "type": "Personal Loan",
                "name": "Apex Private Elite Line of Credit",
                "limit": "Up to $150,000+",
                "rate": "4.15% Variable (Prime + 0.5%)",
                "features": [
                    "Collateralized against your investment assets to minimize rates",
                    "Flexible interest-only monthly repayment options",
                    "No setup fees, annual fees, or draw restrictions"
                ],
                "button_text": "Request Line of Credit"
            },
            {
                "type": "Savings & Investments",
                "name": "Apex Private Wealth Advisory",
                "limit": "Bespoke Portfolios",
                "rate": "Target Return 10-14%",
                "features": [
                    "Bespoke investment strategy tailored to tax requirements",
                    "Direct access to structured notes, private equity, and venture capital",
                    "Dedicated senior wealth manager with direct phone access"
                ],
                "button_text": "Connect with Private Banker"
            },
            {
                "type": "Asset Backed Finance",
                "name": "Apex Lombard Liquidity Facility",
                "limit": "Up to $5,000,000",
                "rate": "3.85% Variable APR",
                "features": [
                    "Maintain market exposure while securing cash liquidity",
                    "Extremely quick funding with minimal paperwork",
                    "High loan-to-value ratio on quality portfolios"
                ],
                "button_text": "Discuss Liquidity Options"
            }
        ]

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
