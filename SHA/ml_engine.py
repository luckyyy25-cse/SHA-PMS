import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

DATA_FILE = "data/hospital_appointments.csv"

class MLEngine:
    def __init__(self, data_path=DATA_FILE):
        self.data_path = data_path
        self.raw_df = None
        self.clean_df = None
        self.lr_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.dept_cols = []
        self.metrics = {}
        self.eda_data = {}
        self.preprocessing_summary = {}

        self.load_and_preprocess()
        self.train_and_evaluate()

    def load_and_preprocess(self):
        self.raw_df = pd.read_csv(self.data_path)
        raw_count = len(self.raw_df)
        missing_age_count = int(self.raw_df["Age"].isna().sum())
        missing_lead_count = int(self.raw_df["Lead_Days"].isna().sum())

        # Raw sample before cleaning (150 rows for scroll view)
        raw_sample_df = self.raw_df.head(150).copy()
        raw_sample = raw_sample_df.astype(object).where(pd.notnull(raw_sample_df), None).to_dict(orient="records")

        df = self.raw_df.copy()

        # Step 1: Impute missing values
        median_age = float(df["Age"].median())
        median_lead = float(df["Lead_Days"].median())
        df["Age"] = df["Age"].fillna(median_age)
        df["Lead_Days"] = df["Lead_Days"].fillna(median_lead)

        # Step 2: Feature Engineering
        df["Age_Group"] = pd.cut(df["Age"], bins=[-1, 18, 40, 60, 100], labels=["Child", "Young Adult", "Middle Adult", "Senior"])
        df["Comorbidity_Count"] = df["Hypertension"] + df["Diabetes"] + df["Alcoholism"]

        self.clean_df = df.copy()

        # Clean sample after wrangling (150 rows for scroll view)
        clean_sample_df = self.clean_df.head(150).copy()
        clean_sample_df["Age_Group"] = clean_sample_df["Age_Group"].astype(str)
        clean_sample = clean_sample_df.to_dict(orient="records")

        self.preprocessing_summary = {
            "total_raw_records": raw_count,
            "total_cleaned_records": len(df),
            "missing_values_imputed": {
                "Age": {"count": missing_age_count, "method": f"Median Imputation ({median_age} yrs)"},
                "Lead_Days": {"count": missing_lead_count, "method": f"Median Imputation ({median_lead} days)"}
            },
            "features_engineered": [
                {"name": "Age_Group", "description": "Binned Age into Child (0-18), Young (19-40), Middle (41-60), Senior (60+)"},
                {"name": "Comorbidity_Count", "description": "Sum of Hypertension + Diabetes + Alcoholism indicators"}
            ],
            "categorical_encodings": [
                {"feature": "Gender", "type": "Binary Encoding (Male=0, Female=1)"},
                {"feature": "Department", "type": "One-Hot Encoding (6 Department Dummy Columns)"}
            ],
            "raw_sample": raw_sample,
            "clean_sample": clean_sample
        }

        # Prepare features X and target y
        dept_dummies = pd.get_dummies(df["Department"], prefix="Dept", drop_first=False).astype(int)
        self.dept_cols = list(dept_dummies.columns)

        X_df = pd.DataFrame({
            "Age": df["Age"],
            "Is_Female": (df["Gender"] == "Female").astype(int),
            "Lead_Days": df["Lead_Days"],
            "Hypertension": df["Hypertension"],
            "Diabetes": df["Diabetes"],
            "Alcoholism": df["Alcoholism"],
            "Previous_No_Shows": df["Previous_No_Shows"],
            "SMS_Reminder": df["SMS_Reminder"],
            "Comorbidity_Count": df["Comorbidity_Count"]
        })

        X_df = pd.concat([X_df, dept_dummies], axis=1)
        self.feature_cols = list(X_df.columns)

        y = df["No_Show"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_df, y, test_size=0.20, random_state=42, stratify=y
        )

        self.X_train_scaled = self.X_train.copy()
        self.X_test_scaled = self.X_test.copy()
        
        num_cols = ["Age", "Lead_Days", "Previous_No_Shows", "Comorbidity_Count"]
        self.scaler.fit(self.X_train[num_cols])
        self.X_train_scaled[num_cols] = self.scaler.transform(self.X_train[num_cols])
        self.X_test_scaled[num_cols] = self.scaler.transform(self.X_test[num_cols])

        self._compute_eda()

    def _compute_eda(self):
        df = self.clean_df
        total = len(df)
        no_show_count = int(df["No_Show"].sum())
        attended_count = total - no_show_count
        no_show_rate = round((no_show_count / total) * 100, 2)

        # 1. Department wise breakdown
        dept_stats = df.groupby("Department")["No_Show"].agg(
            Total="count", No_Shows="sum"
        ).reset_index()
        dept_stats["No_Show_Rate"] = round((dept_stats["No_Shows"] / dept_stats["Total"]) * 100, 1)

        # 2. Lead days vs No-show
        lead_buckets = pd.cut(df["Lead_Days"], bins=[-1, 0, 3, 7, 14, 30, 100], labels=["Same Day", "1-3 Days", "4-7 Days", "8-14 Days", "15-30 Days", "30+ Days"])
        lead_stats = df.groupby(lead_buckets, observed=False)["No_Show"].agg(
            Total="count", No_Shows="sum"
        ).reset_index()
        lead_stats["No_Show_Rate"] = round((lead_stats["No_Shows"] / lead_stats["Total"]) * 100, 1)

        # 3. Age Group breakdown
        age_stats = df.groupby("Age_Group", observed=False)["No_Show"].agg(
            Total="count", No_Shows="sum"
        ).reset_index()
        age_stats["No_Show_Rate"] = round((age_stats["No_Shows"] / age_stats["Total"]) * 100, 1)

        # 4. SMS impact
        sms_stats = df.groupby("SMS_Reminder")["No_Show"].agg(
            Total="count", No_Shows="sum"
        ).reset_index()
        sms_stats["No_Show_Rate"] = round((sms_stats["No_Shows"] / sms_stats["Total"]) * 100, 1)

        # 5. Comorbidities comparison
        comorbidity_stats = [
            {
                "Condition": "Hypertension",
                "Has_Condition_No_Show_Rate": round(float(df[df["Hypertension"] == 1]["No_Show"].mean() * 100), 1),
                "No_Condition_No_Show_Rate": round(float(df[df["Hypertension"] == 0]["No_Show"].mean() * 100), 1)
            },
            {
                "Condition": "Diabetes",
                "Has_Condition_No_Show_Rate": round(float(df[df["Diabetes"] == 1]["No_Show"].mean() * 100), 1),
                "No_Condition_No_Show_Rate": round(float(df[df["Diabetes"] == 0]["No_Show"].mean() * 100), 1)
            },
            {
                "Condition": "Alcoholism",
                "Has_Condition_No_Show_Rate": round(float(df[df["Alcoholism"] == 1]["No_Show"].mean() * 100), 1),
                "No_Condition_No_Show_Rate": round(float(df[df["Alcoholism"] == 0]["No_Show"].mean() * 100), 1)
            }
        ]

        # 6. Correlation Matrix for Heatmap
        corr_cols = ["Age", "Lead_Days", "Hypertension", "Diabetes", "Alcoholism", "Previous_No_Shows", "SMS_Reminder", "Comorbidity_Count", "No_Show"]
        corr_df = df[corr_cols].corr().round(2)
        correlation_matrix = {
            "columns": corr_cols,
            "matrix": corr_df.values.tolist()
        }

        # 7. Scatterplot Data (150 points sample)
        scatter_sample = df.head(150)[["Age", "Lead_Days", "No_Show", "Department", "Gender"]].to_dict(orient="records")

        # 8. Boxplot Data (Quantiles min, q1, median, q3, max)
        def get_boxplot_stats(series):
            q = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(1).tolist()
            return {"min": q[0], "q1": q[1], "median": q[2], "q3": q[3], "max": q[4]}

        boxplots_data = {
            "lead_days": {
                "Attended": get_boxplot_stats(df[df["No_Show"] == 0]["Lead_Days"]),
                "No_Show": get_boxplot_stats(df[df["No_Show"] == 1]["Lead_Days"])
            },
            "age": {
                "Attended": get_boxplot_stats(df[df["No_Show"] == 0]["Age"]),
                "No_Show": get_boxplot_stats(df[df["No_Show"] == 1]["Age"])
            }
        }

        # 9. Gender breakdown
        gender_counts = df["Gender"].value_counts().to_dict()

        self.eda_data = {
            "overview": {
                "total_patients": total,
                "attended_count": attended_count,
                "no_show_count": no_show_count,
                "no_show_rate": no_show_rate,
                "avg_age": round(float(df["Age"].mean()), 1),
                "avg_lead_days": round(float(df["Lead_Days"].mean()), 1)
            },
            "by_department": dept_stats.to_dict(orient="records"),
            "by_lead_days": lead_stats.to_dict(orient="records"),
            "by_age_group": age_stats.to_dict(orient="records"),
            "by_sms": sms_stats.to_dict(orient="records"),
            "by_comorbidity": comorbidity_stats,
            "correlation_matrix": correlation_matrix,
            "scatter_points": scatter_sample,
            "boxplots": boxplots_data,
            "gender_counts": gender_counts
        }

    def train_and_evaluate(self):
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train_scaled, self.y_train)

        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.rf_model.fit(self.X_train_scaled, self.y_train)

        self.metrics = {
            "Logistic_Regression": self._eval_model(self.lr_model, "Logistic Regression"),
            "Random_Forest": self._eval_model(self.rf_model, "Random Forest Classifier")
        }

    def _eval_model(self, model, name):
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1]

        acc = round(accuracy_score(self.y_test, y_pred) * 100, 2)
        prec = round(precision_score(self.y_test, y_pred) * 100, 2)
        rec = round(recall_score(self.y_test, y_pred) * 100, 2)
        f1 = round(f1_score(self.y_test, y_pred) * 100, 2)
        auc = round(roc_auc_score(self.y_test, y_proba), 3)

        cm = confusion_matrix(self.y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_data = [{"fpr": round(float(f), 3), "tpr": round(float(t), 3)} for f, t in zip(fpr, tpr)][::3]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_[0])

        feat_imp = [
            {"feature": col, "importance": round(float(imp), 4)}
            for col, imp in sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)
        ][:8]

        return {
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "roc_curve": roc_data,
            "top_features": feat_imp
        }

    def predict_no_show(self, input_dict):
        age = float(input_dict.get("Age", 30))
        is_female = 1 if input_dict.get("Gender", "Female") == "Female" else 0
        lead_days = float(input_dict.get("Lead_Days", 7))
        hypertension = int(input_dict.get("Hypertension", 0))
        diabetes = int(input_dict.get("Diabetes", 0))
        alcoholism = int(input_dict.get("Alcoholism", 0))
        prev_no_shows = int(input_dict.get("Previous_No_Shows", 0))
        sms = int(input_dict.get("SMS_Reminder", 1))
        dept = input_dict.get("Department", "General Medicine")

        comorbidity = hypertension + diabetes + alcoholism

        row_dict = {
            "Age": age,
            "Is_Female": is_female,
            "Lead_Days": lead_days,
            "Hypertension": hypertension,
            "Diabetes": diabetes,
            "Alcoholism": alcoholism,
            "Previous_No_Shows": prev_no_shows,
            "SMS_Reminder": sms,
            "Comorbidity_Count": comorbidity
        }

        for d_col in self.dept_cols:
            target_dept = d_col.replace("Dept_", "")
            row_dict[d_col] = 1 if dept == target_dept else 0

        single_df = pd.DataFrame([row_dict])[self.feature_cols]

        num_cols = ["Age", "Lead_Days", "Previous_No_Shows", "Comorbidity_Count"]
        single_df[num_cols] = self.scaler.transform(single_df[num_cols])

        proba = float(self.rf_model.predict_proba(single_df)[0][1])
        prediction = 1 if proba >= 0.50 else 0

        if proba >= 0.65:
            risk_level = "High Risk"
            risk_color = "#ef4444"
        elif proba >= 0.35:
            risk_level = "Moderate Risk"
            risk_color = "#f59e0b"
        else:
            risk_level = "Low Risk"
            risk_color = "#10b981"

        risk_factors = []
        if lead_days > 10:
            risk_factors.append(f"Long lead time ({int(lead_days)} days ahead)")
        if prev_no_shows > 0:
            risk_factors.append(f"History of {prev_no_shows} past missed appointment(s)")
        if sms == 0:
            risk_factors.append("No SMS reminder scheduled")
        if alcoholism == 1:
            risk_factors.append("Substance/Alcoholism factor logged")
        if not risk_factors:
            risk_factors.append("Short lead time with scheduled SMS reminder")

        recommendations = []
        if proba >= 0.65:
            recommendations.append("📞 Issue urgent 24-hour phone confirmation call")
            recommendations.append("📅 Schedule a double-booking backup buffer for this slot")
            recommendations.append("📱 Send automated SMS & WhatsApp notification with easy reschedule link")
        elif proba >= 0.35:
            recommendations.append("📱 Send automated 48-hour SMS reminder")
            recommendations.append("💻 Offer optional Telehealth / Virtual Consultation slot")
        else:
            recommendations.append("✅ Standard automated SMS reminder 24h prior")
            recommendations.append("👍 Slot reserved with high confidence")

        return {
            "prediction": "No-Show" if prediction == 1 else "Attended",
            "no_show_probability": round(proba * 100, 1),
            "attendance_probability": round((1 - proba) * 100, 1),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "risk_factors": risk_factors,
            "recommendations": recommendations,
            "patient_summary": {
                "Age": int(age),
                "Gender": "Female" if is_female else "Male",
                "Department": dept,
                "Lead_Days": int(lead_days),
                "Previous_No_Shows": prev_no_shows,
                "SMS_Reminder": "Yes" if sms == 1 else "No"
            }
        }
