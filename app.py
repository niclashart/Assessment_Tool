import streamlit as st
import prognose_tool_ethisch
import pandas as pd
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="HR Prognose-Tool", 
    page_icon="📊", 
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
    }
    .info-text {
        font-size: 1rem;
    }
    .stProgress .st-bo {
        background-color: #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">HR Bewerber-Eignungs-Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Dieses Tool hilft bei der Einschätzung der Eignung von Bewerbern basierend auf HR-Daten.</p>', unsafe_allow_html=True)

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Wählen Sie eine Funktion:", 
                        ["Übersicht", "Modell trainieren", "Bewerber bewerten", "Info"])

# Overview page
if page == "Übersicht":
    st.markdown('<p class="sub-header">Willkommen beim HR Prognose-Tool!</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.write("""
        Mit diesem Tool können Sie:
        1. Ein Eignungsmodell mit Ihren eigenen HR-Daten trainieren
        2. Eignungsprognosen für neue Bewerber/innen erstellen
        3. Die Ergebnisse visualisieren und exportieren
        """)
        
        st.info("""
        **Anleitung:**
        - Nutzen Sie die Seitenleiste zur Navigation zwischen den Funktionen
        - Unter 'Modell trainieren' können Sie Ihre historischen Mitarbeiterdaten hochladen
        - Unter 'Bewerber bewerten' können Sie Bewerberdaten eingeben und eine Eignungsempfehlung erhalten
        """)
    
    with col2:
        # Display a sample chart
        if os.path.exists("prognose_model_ethisch.pkl"):
            st.success("Ein trainiertes Eignungsmodell wurde gefunden!")
            st.image("https://img.freepik.com/free-vector/business-team-putting-together-jigsaw-puzzle-isolated-flat-vector-illustration-cartoon-partners-working-connection-teamwork-partnership-cooperation-concept_74855-9814.jpg", 
                    caption="Effiziente Bewerberauswahl", width=300)
        else:
            st.warning("Kein trainiertes Eignungsmodell gefunden. Bitte trainieren Sie zuerst ein Modell.")
            st.image("https://img.freepik.com/free-vector/team-leader-teamwork-concept_74855-6671.jpg", 
                    caption="Starten Sie mit dem Training des Eignungsmodells", width=300)

# Train model page
elif page == "Modell trainieren":
    st.markdown('<p class="sub-header">Eignungsmodell trainieren</p>', unsafe_allow_html=True)
    st.write("Laden Sie eine Excel-Datei mit historischen Mitarbeiterdaten hoch, um das Eignungsmodell zu trainieren.")
    
    uploaded_file = st.file_uploader("Trainingsdaten hochladen (Excel-Datei)", type=["xlsx"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        # Preview data
        try:
            df = pd.read_excel(tmp_path, sheet_name="Tabelle1")
            st.write("Vorschau der Trainingsdaten:")
            st.dataframe(df.head())
            
            # Display data statistics
            st.write("Datenstatistiken:")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Anzahl der Datensätze: {df.shape[0]}")
                st.write(f"Anzahl der Spalten: {df.shape[1]}")
            with col2:
                gekuendigt = df["Kündigungsdatum"].notna().sum()
                st.write(f"Davon gekündigt: {gekuendigt}")
                st.write(f"Anteil gekündigt: {(gekuendigt / df.shape[0] * 100):.2f}%")
            
            # Train button
            if st.button("Modell trainieren"):
                with st.spinner('Modell wird trainiert...'):
                    prognose_tool_ethisch.trainiere_modell(tmp_path)
                st.success("Eignungsmodell erfolgreich trainiert!")
                
                # Data visualization after training
                X, y = prognose_tool_ethisch.vorbereiten_ethisch(df, is_training=True)
                
                # Feature importance visualization if model exists
                if os.path.exists("prognose_model_ethisch.pkl"):
                    model = prognose_tool_ethisch.joblib.load("prognose_model_ethisch.pkl")
                    feature_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                    ax.set_title('Bedeutung der Merkmale für die Eignungsprognose')
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der Datei: {e}")
        
        finally:
            # Clean up the temp file
            os.unlink(tmp_path)

# Predict page
elif page == "Bewerber bewerten":
    st.markdown('<p class="sub-header">Bewerber-Eignungsprognose</p>', unsafe_allow_html=True)
    
    if not os.path.exists("prognose_model_ethisch.pkl"):
        st.warning("Kein trainiertes Eignungsmodell gefunden. Bitte trainieren Sie zuerst ein Modell.")
    else:
        # Tabs für verschiedene Eingabemethoden
        tab1, tab2 = st.tabs(["Direkteingabe", "Excel-Upload"])
        
        with tab1:
            st.write("Geben Sie die Bewerberdaten direkt ein, um eine Eignungsprognose zu erhalten.")
            
            # Legende für Eingabefelder
            with st.expander("Legende für Eingabefelder", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Berufabschluss:**")
                    st.markdown("- **HW**: Hochschulabschluss")
                    st.markdown("- **K**: Kaufmännische Ausbildung")
                    st.markdown("- **ST**: Studium")
                    st.markdown("- **O**: Ohne Abschluss")
                
                with col2:
                    st.markdown("**Schulabschluss:**")
                    st.markdown("- **MS**: Mittlere Reife")
                    st.markdown("- **AS**: Abitur")
                    st.markdown("- **OS**: Ohne Schulabschluss")
                
                with col3:
                    st.markdown("**Qualifikationsstufe:**")
                    st.markdown("- **S**: Standard")
                    st.markdown("- **M**: Management")
                    st.markdown("- **H**: Höhere Qualifikation")
                    st.markdown("- **A**: Außertariflich")
            
            # Eingabefelder mit sinnvollen Dropdown-Optionen
            col1, col2 = st.columns(2)
            
            with col1:
                berufabschluss = st.selectbox(
                    "Berufabschluss",
                    options=["HW", "K", "ST", "O"],
                    help="HW = Hochschulabschluss, K = Kaufmännische Ausbildung, ST = Studium, O = Ohne Abschluss"
                )
                
                schulabschluss = st.selectbox(
                    "Schulabschluss",
                    options=["MS", "AS", "OS"],
                    help="MS = Mittlere Reife, AS = Abitur, OS = Ohne Schulabschluss"
                )
                
                qualifikationsstufe = st.selectbox(
                    "Qualifikationsstufe",
                    options=["S", "M", "H", "A"],
                    help="S = Standard, M = Management, H = Höhere Qualifikation, A = Außertariflich"
                )
            
            with col2:
                monatsgehalt_einstieg = st.number_input(
                    "Monatsgehalt Einstieg (€)",
                    min_value=2000,
                    max_value=20000,
                    value=3500,
                    step=100,
                    help="Einstiegsgehalt des Mitarbeiters in Euro"
                )
                
                monatsgehalt_aktuell = st.number_input(
                    "Monatsgehalt aktuell (€)",
                    min_value=2000,
                    max_value=20000,
                    value=4500,
                    step=100,
                    help="Aktuelles Gehalt des Mitarbeiters in Euro"
                )
            
            # Hinweis zu ethischen Aspekten
            st.info("Dieses Tool verwendet nur objektive Merkmale für die Bewerbereignungsprognose. Aus ethischen Gründen werden keine personenbezogenen Daten wie Alter, Geschlecht oder Fehlzeiten berücksichtigt.")
            
            # Eignungs-Button
            if st.button("Eignungsprognose erstellen", key="direct_predict"):
                try:
                    with st.spinner('Eignungsprognose wird erstellt...'):
                        # Benutze die neue Funktion für direkte Parametereingabe
                        score = prognose_tool_ethisch.prognose_manuell(
                            monatsgehalt_aktuell,
                            monatsgehalt_einstieg, 
                            qualifikationsstufe,
                            schulabschluss,
                            berufabschluss
                        )
                        
                        # Display result
                        st.success("Eignungsprognose erfolgreich erstellt!")
                        
                        # Score anzeigen
                        score_text = f"Eignungs-Score: {score:.2f}%"
                        
                        # Farbliche Darstellung und Empfehlungen
                        if score < 40:
                            st.error(score_text)
                            st.error("Nicht empfohlen: Der/die Bewerber/in hat eine niedrige prognostizierte Eignung für das Unternehmen.")
                        elif score < 70:
                            st.warning(score_text)
                            st.warning("Bedingt empfohlen: Der/die Bewerber/in könnte für das Unternehmen geeignet sein. Zusätzliche Faktoren prüfen.")
                        else:
                            st.success(score_text)
                            st.success("Empfohlen: Der/die Bewerber/in zeigt eine hohe prognostizierte Eignung für das Unternehmen.")
                        
                        # Fortschrittsbalken
                        st.progress(int(score) / 100)
                        
                        # Einfache Visualisierung
                        fig, ax = plt.subplots(figsize=(10, 2))
                        sns.barplot(x=[score], y=["Eignung"], ax=ax, palette=["green" if score >= 70 else "orange" if score >= 40 else "red"])
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Eignungs-Score (%)')
                        ax.set_ylabel('')
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Fehler bei der Prognose: {e}")
            
        with tab2:
            st.write("Alternativ können Sie eine Excel-Datei mit mehreren Bewerberdaten hochladen.")
            
            uploaded_file = st.file_uploader("Bewerberdaten hochladen (Excel-Datei)", type=["xlsx"])
            
            if uploaded_file is not None:
                # Save the uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Preview data
                try:
                    df = pd.read_excel(tmp_path, sheet_name="Tabelle1")
                    st.write("Vorschau der Bewerberdaten:")
                    st.dataframe(df.head())
                    
                    # Make predictions button
                    if st.button("Eignungsprognosen erstellen", key="excel_predict"):
                        with st.spinner('Eignungsprognosen werden erstellt...'):
                            # Benutze die neue Funktion für Excel-Prognosen
                            ergebnisse = prognose_tool_ethisch.prognose_excel(tmp_path)
                            
                            # Display results
                            st.success("Eignungsprognosen erfolgreich erstellt!")
                            st.write("Ergebnisse:")
                            
                            # Rename column to reflect purpose
                            if 'Prognose-Score (%)' in ergebnisse.columns:
                                ergebnisse = ergebnisse.rename(columns={'Prognose-Score (%)': 'Eignungs-Score (%)'})
                            
                            # Display results with conditional formatting
                            def color_score(val):
                                color = ''
                                if val < 40:
                                    color = 'red'
                                elif val < 70:
                                    color = 'orange'
                                else:
                                    color = 'green'
                                return f'background-color: {color}; color: white;'
                            
                            styled_output = ergebnisse.style.applymap(
                                color_score, subset=['Eignungs-Score (%)']
                            )
                            
                            st.dataframe(styled_output)
                            
                            # Download button for results
                            csv = ergebnisse.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Eignungsergebnisse als CSV herunterladen", 
                                csv, 
                                "bewerber_eignungsergebnisse.csv", 
                                "text/csv", 
                                key='download-csv'
                            )
                            
                            # Visualization
                            if 'Eignungs-Score (%)' in ergebnisse.columns:
                                score_values = ergebnisse['Eignungs-Score (%)']
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(score_values, bins=10, kde=True, ax=ax)
                                ax.set_title('Verteilung der Eignungs-Scores')
                                ax.set_xlabel('Eignungs-Score (%)')
                                ax.set_ylabel('Anzahl der Bewerber')
                                st.pyplot(fig)
                                
                                # Display statistics
                                st.write(f"Durchschnittlicher Eignungs-Score: {score_values.mean():.2f}%")
                                st.write(f"Niedrigster Eignungs-Score: {score_values.min():.2f}%")
                                st.write(f"Höchster Eignungs-Score: {score_values.max():.2f}%")
                    
                except Exception as e:
                    st.error(f"Fehler beim Verarbeiten der Datei: {e}")
                
                finally:
                    # Clean up the temp file
                    os.unlink(tmp_path)

# Info page
elif page == "Info":
    st.markdown('<p class="sub-header">Über dieses Tool</p>', unsafe_allow_html=True)
    
    st.write("""
    ## HR Bewerber-Eignungs-Tool
    
    Dieses Tool unterstützt die Personalabteilung bei der Bewertung von Bewerbern, um zu prognostizieren,
    wie produktiv und geeignet ein Kandidat für das Unternehmen sein wird.
    
    ### Einsatzbereich:
    Das Tool gibt eine rasche Empfehlung, ob ein Bewerber/eine Bewerberin für das Unternehmen 
    voraussichtlich eine produktive Arbeitskraft wird und deshalb eingestellt werden sollte.
    
    ### Ethische Grundsätze:
    Aus ethischen Gründen werden nur objektive, nicht-diskriminierende Merkmale verwendet.
    
    ### Verwendete Merkmale:
    - **Monatsgehalt Einstieg**: Einstiegsgehalt in Euro
    - **Monatsgehalt aktuell**: Aktuelles/letztes Gehalt des Bewerbers in Euro
    - **Qualifikationsstufe**: Kodiert als S (Standard), M (Management), H (Höhere Qualifikation), A (Außertariflich)
    - **Schulabschluss**: Kodiert als MS (Mittlere Reife), AS (Abitur), OS (Ohne Schulabschluss)
    - **Berufabschluss**: Kodiert als HW (Hochschulabschluss), K (Kaufmännische Ausbildung), ST (Studium), O (Ohne Abschluss)
    
    ### Nicht verwendete Merkmale:
    Um eine faire und ethisch vertretbare Eignungsbewertung zu gewährleisten, werden bewusst
    keine demographischen oder potenziell diskriminierenden Merkmale verwendet, wie:
    - Alter
    - Geschlecht
    - Ethnische Herkunft
    - Fehlzeiten
    - Familienstand
    
    ### Modelldetails:
    Das Tool verwendet einen Random Forest Classifier für die Eignungsprognose und zeigt 
    als Ergebnis einen Eignungs-Score in Prozent an.
    
    ### Datenschutz:
    Die hochgeladenen Bewerberdaten werden nur temporär verarbeitet und nicht dauerhaft gespeichert.
    """)
    
    st.info("""
    **Hinweis**: Die Eignungsprognosen dieses Tools sind statistische Schätzungen und 
    sollten als Entscheidungshilfe, nicht als alleinige Grundlage für Einstellungsentscheidungen dienen.
    Es wird empfohlen, die Ergebnisse im Kontext aller verfügbaren Bewerbungsunterlagen und Gespräche zu bewerten.
    """)

# Footer
st.markdown("""---""")
st.markdown(
    """<div style="text-align: center; color: gray; font-size: 0.8rem;">
    HR Bewerber-Eignungs-Tool © 2024 | Version 1.0
    </div>""", 
    unsafe_allow_html=True
)

