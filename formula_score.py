import pandas as pd
import numpy as np

# === Datei laden ===
file_path = "Datensatz HR-20240416 - Zusatzinfos.xlsx"
excel_file = pd.ExcelFile(file_path)
df = excel_file.parse(excel_file.sheet_names[0])

# === Scoring-Funktionen ===

# 1. Qualifikation
def qualifikations_score(row):
    q_map = {'H': 0.5, 'M': 0.5, 'S': 0.5, 'A': 0.5}
    s_map = {'OS': 0.25, 'MS': 0.6, 'AS': 1.0}
    b_map = {'O': 0.3, 'K': 0.6, 'ST': 1.0, 'HW': 0.6}
    
    q = q_map.get(str(row['Qualifikationstufe']).strip(), 0)
    s = s_map.get(str(row['Schulabschluss']).strip(), 0)
    b = b_map.get(str(row['Berufabschluss']).strip(), 0)
    
    return np.mean([q, s, b])

# 2. Leistung (Gehaltsentwicklung)
def leistungs_score(row):
    try:
        gehalt_aktuell = float(str(row['Monatsgehalt aktuell/ bzw. zuletz bezogenes Gehalt']).replace(",", "."))
        gehalt_einstieg = float(str(row['Monatsgehalt Einstieg']).replace(",", "."))
        if gehalt_einstieg <= 0:
            return 0.0
        ratio = gehalt_aktuell / gehalt_einstieg
        if ratio > 1.5:
            return 1.0
        elif ratio > 1.2:
            return 0.7
        else:
            return 0.4
    except:
        return 0.0

# 3. Kontinuität (Verbleibsdauer & Fehlzeiten)
def kontinuität_score(row):
    try:
        eintritt = pd.to_datetime(row['Einstellungsdatum'])
        austritt = pd.to_datetime(row['Kündigungsdatum']) if pd.notnull(row['Kündigungsdatum']) else pd.Timestamp.now()
        dauer_jahre = (austritt - eintritt).days / 365.25
        if dauer_jahre > 5:
            score = 1.0
        elif dauer_jahre > 2:
            score = 0.6
        else:
            score = 0.3

        fehlzeit = row['Fehlzeiten (Monaten)']
        if pd.notnull(fehlzeit) and fehlzeit > 3:
            score -= 0.2
        return max(score, 0.0)
    except:
        return 0.0

# === Score berechnen ===

def berechne_score(row):
    return round(100 * np.mean([
        qualifikations_score(row),
        leistungs_score(row),
        kontinuität_score(row)
    ]), 1)

df['Score (%)'] = df.apply(berechne_score, axis=1)

# === Ergebnis anzeigen ===

print(df[['Vorname', 'Nachname', 'Qualifikationstufe', 
          'Monatsgehalt Einstieg', 'Monatsgehalt aktuell/ bzw. zuletz bezogenes Gehalt', 
          'Fehlzeiten (Monaten)', 'Einstellungsdatum', 
          'Kündigungsdatum', 'Score (%)']].head(10))
