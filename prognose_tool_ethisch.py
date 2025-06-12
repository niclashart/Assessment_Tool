#!/usr/bin/env python3
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Globale LabelEncoder zur Reproduzierbarkeit für Streamlit-Interface
le_quali = LabelEncoder().fit(["H", "S", "M", "A"])
le_schul = LabelEncoder().fit(["MS", "AS", "OS"])
le_beruf = LabelEncoder().fit(["ST", "K", "O", "HW"])

# Nur reduzierte Features (ethisch vertretbar)
def vorbereiten_ethisch(df, is_training=True):
    df = df.copy()
    df["Schulabschluss"] = df["Schulabschluss"].replace({"O": "OS"})
    if "Kündigungsdatum" in df.columns:
        df["IstGekündigt"] = df["Kündigungsdatum"].notnull().astype(int)

    df["Qualifikationstufe"] = le_quali.transform(df["Qualifikationstufe"].astype(str))
    df["Schulabschluss"] = le_schul.transform(df["Schulabschluss"].astype(str))
    df["Berufabschluss"] = le_beruf.transform(df["Berufabschluss"].astype(str))

    features = ["Monatsgehalt aktuell/ bzw. zuletz bezogenes Gehalt", "Monatsgehalt Einstieg",
                "Qualifikationstufe", "Schulabschluss", "Berufabschluss"]

    if is_training:
        return df[features], df["IstGekündigt"]
    else:
        return df[features], df[["Nachname", "Vorname"]] if "Nachname" in df.columns else df[features]

# Modelltraining
def trainiere_modell(train_excel):
    df_train = pd.read_excel(train_excel, sheet_name=None)["Tabelle1"]
    X, y = vorbereiten_ethisch(df_train, is_training=True)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "prognose_model_ethisch.pkl")
    print("Modell trainiert und gespeichert als 'prognose_model_ethisch.pkl'.")

# Einzelprognose für Streamlit
def prognose_manuell(monatsgehalt_aktuell, monatsgehalt_einstieg, quali, schul, beruf):
    model = joblib.load("prognose_model_ethisch.pkl")
    daten = pd.DataFrame([[
        monatsgehalt_aktuell,
        monatsgehalt_einstieg,
        le_quali.transform([quali])[0],
        le_schul.transform([schul])[0],
        le_beruf.transform([beruf])[0]
    ]], columns=["Monatsgehalt aktuell/ bzw. zuletz bezogenes Gehalt", "Monatsgehalt Einstieg",
                 "Qualifikationstufe", "Schulabschluss", "Berufabschluss"])

    p = model.predict_proba(daten)[:, 1]
    eignung_score = round((1 - p[0]) * 100, 2)
    return eignung_score

# Prognose über Excel-Testdatei
def prognose_excel(test_excel):
    df_test = pd.read_excel(test_excel, sheet_name=None)["Tabelle1"]
    X_test, namen = vorbereiten_ethisch(df_test, is_training=False)
    model = joblib.load("prognose_model_ethisch.pkl")
    prognosen = model.predict_proba(X_test)[:, 1]
    score = ((1 - prognosen) * 100).round(2)
    if isinstance(namen, pd.DataFrame):
        namen["Prognose-Score (%)"] = score
        return namen
    else:
        return pd.DataFrame({"Prognose-Score (%)": score})
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prognose-Tool (ethisch reduzierte Kriterien)")
    parser.add_argument("--train", help="Pfad zur Trainings-Excel-Datei")
    parser.add_argument("--test", help="Pfad zur Test-Excel-Datei (optional)")
    args = parser.parse_args()

    if args.train:
        trainiere_modell(args.train)
    if args.test:
        df_ergebnisse = prognose_excel(args.test)
        print(df_ergebnisse.to_string(index=False))