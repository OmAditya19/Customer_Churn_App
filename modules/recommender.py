# modules/recommender.py

import pandas as pd
import json

class RecommendationEngine:
    def __init__(self, rule_file: str):
        with open(rule_file, 'r') as file:
            self.rules = json.load(file)

    def recommend(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        recommendations = []
        for _, customer in customer_data.iterrows():
            rec = self._apply_rules(customer)
            recommendations.append(rec)
        return pd.DataFrame(recommendations)

    def _apply_rules(self, customer: pd.Series) -> dict:
        result = customer.to_dict()
        churn_prob = customer.get("churn_probability", 0)

        result["confidence_score"] = churn_prob

        for rule in self.rules["rules"]:
            if churn_prob >= rule["churn_threshold"]:
                result["recommendation"] = rule["recommendation"]
                break
        else:
            result["recommendation"] = "No action"

        return result
