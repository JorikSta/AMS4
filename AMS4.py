import streamlit as st
import requests
import pandas as pd
import json
import ast # Nodig voor het veilig evalueren van string-representaties van Python-structuren (dictionaries)
import plotly.express as px # Nodig voor het maken van interactieve plots
import plotly.graph_objects as go # Nodig voor geoplots met lijnen

# --- Streamlit Pagina Configuratie ---
st.set_page_config(
    page_title="Schiphol Data Dashboard",
    layout="wide" # Gebruik de volledige breedte van de pagina
)

# --- Functies voor Data Ophalen en Verwerken ---

@st.cache_data(show_spinner=False) # show_spinner=False because we're creating our own progress bar inside
def get_schiphol_flight_data(max_pages_to_load):
    """
    Haalt vluchtdata op van de Schiphol API met paginering.
    Deze versie creëert zijn eigen Streamlit voortgangsindicatoren binnen de functie.

    Args:
        max_pages_to_load (int): Het maximale aantal pagina's dat opgehaald moet worden.

    Returns:
        list: Een lijst met alle opgehaalde vluchtgegevens.
              Retourneert een lege lijst als er een fout optreedt.
    """
    url_base = "https://api.schiphol.nl/public-flights/flights"
    headers = {
        'accept': 'application/json',
        'resourceversion': 'v4',
        'app_id': 'b1ff0af7', # **BELANGRIJK: VERVANG DIT MET JE EIGEN app_id**
        'app_key': '43567fff2ced7e77e947c1b71ebc5f38' # **BELANGRIJK: VERVANG DIT MET JE EIGEN app_key**
    }

    all_flights_data = []
    page_counter = 0
    next_page_url = url_base

    # Creëer de progress indicators BINNEN de gecachede functie
    # Deze zullen alleen verschijnen als de functie daadwerkelijk wordt uitgevoerd (cache miss)
    progress_text_placeholder = st.empty()
    progress_bar = st.progress(0)

    for i in range(max_pages_to_load):
        if not next_page_url:
            break

        progress_text_placeholder.text(f"Laden van pagina {i + 1} van {max_pages_to_load}...")
        progress_bar.progress((i + 1) / max_pages_to_load)

        try:
            response = requests.get(next_page_url, headers=headers)
            response.raise_for_status() # Genereert een HTTPError voor slechte antwoorden (4xx of 5xx)
            data = response.json()

            if 'flights' in data:
                all_flights_data.extend(data['flights'])
            else:
                st.warning(f"Geen 'flights' sleutel gevonden in het antwoord van pagina {i+1}. Stopt met laden.")
                break

            next_link = response.headers.get('Link')
            next_page_url = None
            if next_link:
                links = next_link.split(',')
                for link in links:
                    if 'rel="next"' in link:
                        next_page_url = link.split(';')[0].strip('<> ')
                        break
            page_counter += 1

        except requests.exceptions.RequestException as e:
            st.error(f"Fout bij het ophalen van data van pagina {i+1}: {e}. Stopt met laden.")
            break
        except ValueError as e:
            st.error(f"Fout bij het parsen van JSON: {e}. Stopt met laden.")
            break
        except Exception as e:
            st.error(f"Een onverwachte fout trad op tijdens het laden van pagina {i+1}: {e}. Stopt met laden.")
            break

    progress_text_placeholder.text(f"Laden voltooid! Totaal {page_counter} pagina's geladen.")
    progress_bar.progress(1.0) # Zet de balk op 100%
    return all_flights_data

@st.cache_data(show_spinner="Data wordt verwerkt en voorbereid...") # Cache de dataverwerking om tijd te besparen
def process_flight_dataframe(df_raw, airports_df):
    """
    Verwerkt de ruwe vlucht DataFrame door kolommen te transformeren en toe te voegen.
    Ontvangt nu ook de airports_df als argument.
    """
    df = df_raw.copy() # Werk op een kopie om de originele data niet te wijzigen

    # --- Converteer 'route' kolom van string naar dictionary en bepaal origin/destination ---
    if 'route' in df.columns and 'flightDirection' in df.columns:
        df['route'] = df['route'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        def get_origin_destination_iata(row):
            route_data = row['route']
            flight_direction = row['flightDirection']
            origin = None
            destination = None

            if isinstance(route_data, dict) and 'destinations' in route_data and len(route_data['destinations']) > 0:
                if flight_direction == 'A': # Arrival
                    origin = route_data['destinations'][0] # For arrivals, 'destinations' in routes refers to origin
                    destination = 'AMS' # Schiphol is the destination
                elif flight_direction == 'D': # Departure
                    origin = 'AMS' # Schiphol is the origin
                    destination = route_data['destinations'][0] # For departures, 'destinations' is the actual destination
            return origin, destination

        df[['origin_airport_iata', 'destination_airport_iata']] = df.apply(get_origin_destination_iata, axis=1, result_type='expand')

        if 'destination' not in df.columns:
            df['destination'] = df['route'].apply(
                lambda x: x['destinations'][0] if isinstance(x, dict) and 'destinations' in x and len(x['destinations']) > 0 else None
            )

    else:
        st.warning("Kolommen 'route' of 'flightDirection' niet gevonden. Kan geen 'origin_airport_iata' of 'destination_airport_iata' kolommen toevoegen.")


    # --- Map afkortingen naar volledige namen en voeg coördinaten toe ---
    if 'destination' in df.columns and not airports_df.empty: # Zorg dat airports_df geladen is
        airport_name_map = airports_df.set_index('iata_code')['name'].to_dict()
        df['destination_full_name'] = df['destination'].map(airport_name_map)
        df['destination_full_name'] = df['destination_full_name'].fillna(df['destination'])

        lat_map = airports_df.set_index('iata_code')['latitude_deg'].to_dict()
        lon_map = airports_df.set_index('iata_code')['longitude_deg'].to_dict()
        df['destination_latitude'] = df['destination'].map(lat_map)
        df['destination_longitude'] = df['destination'].map(lon_map)

        if (df['destination_latitude'].isna().sum() / len(df) > 0.5) or \
           (df['destination_longitude'].isna().sum() / len(df) > 0.5):
            st.warning("Waarschuwing: Zeer weinig luchthavencoördinaten gevonden na het koppelen met 'airports.csv'.")
            st.info("Dit kan komen doordat de IATA-codes in de Schiphol-data (in 'destination' kolom) niet overeenkomen met die in 'airports.csv', of door problemen met de kolomnamen/opmaak in 'airports.csv'.")
            st.dataframe(df[['destination', 'destination_full_name', 'destination_latitude', 'destination_longitude']].head())

    else:
        st.warning("De 'destination' kolom of 'airports.csv' data is niet beschikbaar. Kan geen volledige namen of coördinaten toevoegen.")


    # --- Datum/Tijd conversie en vertraging berekenen ---
    planned_landing_time_col = 'estimatedLandingTime'
    actual_landing_time_col = 'actualLandingTime'
    scheduled_departure_time_col = 'scheduleDateTime'

    # Prioritised list of date formats to try
    date_formats_to_try = [
        "%Y-%m-%dT%H:%M:%S.%f%z",   # e.g., 2025-06-27T00:05:00.000+02:00 (with milliseconds and 'T')
        "%Y-%m-%d %H:%M:%S%z",      # e.g., 2025-06-27 00:42:37+02:00 (with space and no milliseconds)
        "%Y-%m-%dT%H:%M:%S%z",      # e.g., 2025-06-27T00:05:00+02:00 (no milliseconds)
        "%Y-%m-%d %H:%M:%S",        # e.g., 2025-06-27 00:42:37 (no timezone)
        "%Y-%m-%dT%H:%M:%S",        # e.g., 2025-06-27T00:05:00 (no timezone, no milliseconds)
        "%Y-%m-%d",                 # e.g., 2025-06-27 (only date)
    ]

    # Function to apply multiple date formats
    def parse_datetime_robustly(series):
        series_as_str = series.apply(lambda x: str(x).strip() if pd.notna(x) else '')
        
        # Attempt 1: Infer format and directly get UTC-aware datetime
        # This is the most flexible initial attempt
        parsed_utc_series = pd.to_datetime(series_as_str, errors='coerce', infer_datetime_format=True, utc=True)

        # Attempt 2: For values that failed in Attempt 1, try specific known formats directly to UTC
        unparsed_mask = parsed_utc_series.isna()
        if unparsed_mask.any():
            for fmt in date_formats_to_try:
                if unparsed_mask.any(): # Re-check if there are still unparsed values
                    attempt_parse_fmt = pd.to_datetime(series_as_str[unparsed_mask], format=fmt, errors='coerce', utc=True)
                    parsed_utc_series.loc[unparsed_mask] = attempt_parse_fmt.dropna()
                    unparsed_mask = parsed_utc_series.isna()

        # Final step: Convert all successfully parsed UTC-aware datetimes to naive datetimes (representing UTC time)
        # This ensures all datetime columns are of dtype 'datetime64[ns]' for consistent operations.
        if pd.api.types.is_datetime64_any_dtype(parsed_utc_series) and parsed_utc_series.notna().any():
            # Apply tz_localize(None) only if the series is timezone-aware
            if parsed_utc_series.dt.tz is not None:
                final_series = parsed_utc_series.dt.tz_localize(None)
            else:
                # If it's already naive (but expected to be UTC due to utc=True in parsing), keep it as is.
                # This path should ideally not be hit if utc=True worked as expected.
                final_series = parsed_utc_series
            
            # Ensure the dtype is 'datetime64[ns]'
            return final_series.astype('datetime64[ns]')
        else:
            # If no datetimes were successfully parsed to datetime dtype, return a series of NaT
            if parsed_utc_series.isna().all():
                 st.warning(f"Failed to parse ALL datetime values for this series. Returning all NaT.")
                 return pd.Series([pd.NaT] * len(series), index=series.index, dtype='datetime64[ns]')
            
            st.warning(f"Some datetime values failed to parse to a proper datetime dtype after all attempts for this series.")
            st.warning(f"Sample of values that couldn't be fully parsed: {series_as_str[parsed_utc_series.isna()].head().to_string()}")
            return parsed_utc_series.astype('datetime64[ns]', errors='ignore') # Coerce to datetime64[ns], but will leave NaT if unparseable


    # Process estimatedLandingTime and actualLandingTime
    if planned_landing_time_col in df.columns:
        df[planned_landing_time_col] = parse_datetime_robustly(df[planned_landing_time_col])
        if not pd.api.types.is_datetime64_any_dtype(df[planned_landing_time_col]):
            df['arrival_hour'] = pd.NA
            st.warning(f"Could not convert '{planned_landing_time_col}' to datetime. 'arrival_hour' not created or is all NaT.")
        else:
            df['arrival_hour'] = df[planned_landing_time_col].dt.hour
    else:
        st.warning(f"Kolom '{planned_landing_time_col}' niet gevonden. Kan geen 'arrival_hour' toevoegen.")

    if actual_landing_time_col in df.columns and planned_landing_time_col in df.columns:
        df[actual_landing_time_col] = parse_datetime_robustly(df[actual_landing_time_col])
        if pd.api.types.is_datetime64_any_dtype(df[planned_landing_time_col]) and \
           pd.api.types.is_datetime64_any_dtype(df[actual_landing_time_col]):
            df['delay_minutes'] = ((df[actual_landing_time_col] - df[planned_landing_time_col]).dt.total_seconds() / 60).fillna(0).astype(int)
        else:
            st.warning(f"Kan 'delay_minutes' niet berekenen. Kolommen '{actual_landing_time_col}' of '{planned_landing_time_col}' zijn geen geldige datetime types.")
    else:
        st.warning(f"Kolommen '{actual_landing_time_col}' of '{planned_landing_time_col}' ontbreken. Geen vertragingskolom toegevoegd.")

    # Verwerk scheduleDateTime voor departure_hour
    if scheduled_departure_time_col in df.columns:
        df[scheduled_departure_time_col] = parse_datetime_robustly(df[scheduled_departure_time_col])
        if not pd.api.types.is_datetime64_any_dtype(df[scheduled_departure_time_col]):
            df['departure_hour'] = pd.NA
            st.warning(f"Could not convert '{scheduled_departure_time_col}' to datetime. 'departure_hour' not created or is all NaT.")
        else:
            df['departure_hour'] = df[scheduled_departure_time_col].dt.hour
    else:
        st.warning(f"Kolom '{scheduled_departure_time_col}' niet gevonden. Kan geen 'departure_hour' toevoegen.")


    # --- Verwerking van 'aircraftType' kolom en toevoegen 'main_aircraft_type' ---
    if 'aircraftType' in df.columns:
        def get_main_aircraft_type(aircraft_type_data):
            if isinstance(aircraft_type_data, dict):
                if 'iataMain' in aircraft_type_data:
                    return aircraft_type_data['iataMain']
            elif isinstance(aircraft_type_data, str):
                try:
                    parsed_type = ast.literal_eval(aircraft_type_data)
                    if isinstance(parsed_type, dict) and 'iataMain' in parsed_type:
                        return parsed_type['iataMain']
                except (ValueError, SyntaxError):
                    return None
            return None

        df['main_aircraft_type'] = df['aircraftType'].apply(get_main_aircraft_type)
    else:
        st.warning("Kolom 'aircraftType' niet gevonden. Kan 'main_aircraft_type' kolom niet toevoegen.")

    # --- Verwijder de gespecificeerde kolommen ---
    columns_to_drop = [
        'schemaVersion', 'actualOffBlockTime', 'aircraftRegistration',
        'transferPositions', 'checkinAllocations', 'expectedSecurityFilter',
        'expectedTimeBoarding', 'expectedTimeGateClosing', 'expectedTimeGateOpen'
    ]
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    if existing_columns_to_drop:
        df.drop(columns=existing_columns_to_drop, inplace=True)

    return df

def create_hourly_flight_traffic_plot(df_processed, selected_direction='Totaal'):
    """
    Creëert een lijnplot van de vluchtenactiviteit per uur, gesplitst per richting.
    """
    df_plot = df_processed.copy()

    # Deze checks zijn cruciaal, als ze mislukken, betekent het dat de datetime parsing in process_flight_dataframe faalde
    # Check of arrival_hour kolom bestaat en numeriek is, anders waarschuwing
    if 'arrival_hour' not in df_plot.columns or not pd.api.types.is_numeric_dtype(df_plot['arrival_hour']) or df_plot['arrival_hour'].isnull().all():
        st.warning("Geen geldige 'arrival_hour' data beschikbaar voor plot. Controleer 'Data Laden en Verwerken'.")
        hourly_arrival_counts = pd.Series([], dtype='int64') # Lege serie
    else:
        # Filter op 'A' flightDirection en gebruik arrival_hour
        hourly_arrival_counts = df_plot[df_plot['flightDirection'] == 'A']['arrival_hour'].value_counts().sort_index()

    # Check of departure_hour kolom bestaat en numeriek is, anders waarschuwing
    if 'departure_hour' not in df_plot.columns or not pd.api.types.is_numeric_dtype(df_plot['departure_hour']) or df_plot['departure_hour'].isnull().all():
        st.warning("Geen geldige 'departure_hour' data beschikbaar voor plot. Controleer 'Data Laden en Verwerken'.")
        hourly_departure_counts = pd.Series([], dtype='int64') # Lege serie
    else:
        # Filter op 'D' flightDirection en gebruik departure_hour
        hourly_departure_counts = df_plot[df_plot['flightDirection'] == 'D']['departure_hour'].value_counts().sort_index()


    if hourly_arrival_counts.empty and hourly_departure_counts.empty:
        st.warning("Geen aankomst- of vertrekdata gevonden om de vluchtenactiviteit per uur te plotten. Plot kan leeg zijn.")
        return None

    # Combineer naar één DataFrame
    hourly_activity = pd.DataFrame({
        'Aankomsten': hourly_arrival_counts,
        'Vertrekken': hourly_departure_counts
    }).fillna(0) # Vul ontbrekende uren op met 0

    # Zorg dat de index (uur) volledig is van 0 tot 23
    all_hours = pd.Series(range(24))
    hourly_activity = hourly_activity.reindex(all_hours, fill_value=0)
    hourly_activity.index.name = 'Uur van de dag'
    hourly_activity['Totaal'] = hourly_activity['Aankomsten'] + hourly_activity['Vertrekken']
    hourly_activity = hourly_activity.reset_index()

    # Kies welke lijnen getoond moeten worden
    if selected_direction == 'Aankomsten':
        columns_to_plot = ['Aankomsten']
    elif selected_direction == 'Vertrekken':
        columns_to_plot = ['Vertrekken']
    else: # Totaal of alle
        columns_to_plot = ['Aankomsten', 'Vertrekken', 'Totaal']

    # Maak de lijnplot
    fig = px.line(
        hourly_activity,
        x='Uur van de dag',
        y=columns_to_plot,
        title='Vluchtenactiviteit per uur van de dag',
        labels={'Uur van de dag': 'Uur van de dag (24-uurs)', 'value': 'Aantal vluchten', 'variable': 'Vluchtrichting'},
        markers=True # Toon punten op de lijnen
    )

    fig.update_layout(xaxis_title="Uur van de dag", yaxis_title="Aantal vluchten")
    fig.update_xaxes(tickvals=list(range(0, 24, 2))) # Toon om de 2 uur een label
    return fig


def create_aircraft_type_pie_plot(df_processed):
    """
    Creëert een donut plot van de verdeling van vliegtuigtypes.
    Gebruikt de 'main_aircraft_type' kolom.
    """
    if 'main_aircraft_type' in df_processed.columns:
        aircraft_type_counts = df_processed['main_aircraft_type'].dropna().value_counts().reset_index()
        aircraft_type_counts.columns = ['Vliegtuigtype', 'Aantal']

        if not aircraft_type_counts.empty:
            fig_pie = px.pie(
                aircraft_type_counts,
                values='Aantal',
                names='Vliegtuigtype',
                title='Verdeling van Vliegtuigtypes',
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            return fig_pie
        else:
            st.warning("Geen geldige vliegtuigtypes gevonden om de ronde plot te maken.")
            return None
    else:
        st.warning("Kolom 'main_aircraft_type' niet gevonden. Zorg dat 'aircraftType' correct geparsed wordt.")
        return None

def create_delay_over_day_plot(df_processed):
    """
    Creëert een plot van de gemiddelde vertraging over de uren van de dag.
    Gebruikt 'estimatedLandingTime' om het uur te berekenen.
    """
    if 'estimatedLandingTime' in df_processed.columns and 'delay_minutes' in df_processed.columns:
        df_valid = df_processed.dropna(subset=['estimatedLandingTime', 'delay_minutes'])
        if not df_valid.empty:
            # Controleren of estimatedLandingTime een datetime type is
            if pd.api.types.is_datetime64_any_dtype(df_valid['estimatedLandingTime']):
                df_valid['arrival_hour_derived'] = df_valid['estimatedLandingTime'].dt.hour # Tijdelijke kolom
                df_valid['delay_minutes'] = pd.to_numeric(df_valid['delay_minutes'], errors='coerce').fillna(0).clip(lower=0)
                avg_delay_hourly = df_valid.groupby('arrival_hour_derived')['delay_minutes'].mean().reset_index()
                avg_delay_hourly.columns = ['Uur van de dag', 'Gemiddelde vertraging (minuten)']

                fig_delay = px.line(
                    avg_delay_hourly,
                    x='Uur van de dag',
                    y='Gemiddelde vertraging (minuten)',
                    title='Gemiddelde vertraging per uur van de dag',
                    labels={'Uur van de dag': 'Uur van de dag (24-uurs)', 'Gemiddelde vertraging (minuten)': 'Gemiddelde vertraging (minuten)'}
                )
                fig_delay.update_layout(xaxis_title="Uur van de dag", yaxis_title="Gemiddelde vertraging (minuten)")
                return fig_delay
            else:
                st.warning("Kolom 'estimatedLandingTime' is geen datetime type. Kan gemiddelde vertraging per uur niet plotten.")
                return None
        else:
            st.warning("Geen geldige data voor gemiddelde vertraging per uur.")
            return None
    else:
        st.warning("Kolommen 'estimatedLandingTime' of 'delay_minutes' niet gevonden. Kan vertraging over de dag niet plotten.")
        return None

def create_top_10_destinations_plot(df_processed):
    """
    Creëert een bar plot van de top 10 aankomstplaatsen.
    Gebruikt 'destination_full_name' voor de namen.
    """
    if 'destination_full_name' in df_processed.columns and 'flightDirection' in df_processed.columns:
        # Filter alleen voor aankomende vluchten ('A')
        df_arrivals = df_processed[df_processed['flightDirection'] == 'A'].copy()
        destination_counts = df_arrivals['destination_full_name'].dropna().value_counts().nlargest(10).reset_index()
        destination_counts.columns = ['Bestemming', 'Aantal vluchten']

        if not destination_counts.empty:
            fig_top10 = px.bar(
                destination_counts,
                x='Aantal vluchten',
                y='Bestemming',
                orientation='h', # Horizontale balken
                title='Top 10 Aankomstplaatsen',
                labels={'Aantal vluchten': 'Aantal aankomsten', 'Bestemming': 'Aankomstplaats'},
                text='Aantal vluchten'
            )
            fig_top10.update_layout(yaxis={'categoryorder':'total ascending'}) # Sorteer van klein naar groot
            return fig_top10
        else:
            st.warning("Geen geldige data voor top 10 aankomstbestemmingen.")
            return None
    else:
        st.warning("Kolom 'destination_full_name' of 'flightDirection' niet gevonden. Kan Top 10 Aankomstplaatsen niet plotten.")
        return None

def create_top_10_departures_plot(df_processed):
    """
    Creëert een bar plot van de top 10 vertrekbestemmingen.
    Gebruikt 'destination_airport_iata' (geparst uit 'route') als de bestemming voor vertrekken.
    """
    if 'destination_airport_iata' in df_processed.columns and 'flightDirection' in df_processed.columns and not df_processed.empty:
        # Filter alleen voor vertrekkende vluchten ('D')
        df_departures = df_processed[df_processed['flightDirection'] == 'D'].copy()
        
        # Gebruik de IATA code voor vertrekbestemmingen en map naar volledige naam indien mogelijk
        # We moeten de 'destination_airport_iata' kolom gebruiken en deze mappen naar volledige namen
        if 'airports_data_df' in st.session_state and not st.session_state['airports_data_df'].empty:
            airports_df = st.session_state['airports_data_df']
            airport_name_map = airports_df.set_index('iata_code')['name'].to_dict()
            df_departures['destination_full_name_for_plot'] = df_departures['destination_airport_iata'].map(airport_name_map)
            # Vul NaN op met de IATA code als de volledige naam niet gevonden is
            df_departures['destination_full_name_for_plot'] = df_departures['destination_full_name_for_plot'].fillna(df_departures['destination_airport_iata'])
            col_to_count = 'destination_full_name_for_plot'
        else:
            # Als airports_data_df niet beschikbaar is, gebruik dan de IATA code direct
            col_to_count = 'destination_airport_iata'
            st.warning("Luchthavengegevens niet volledig geladen, gebruikt IATA-codes voor vertrekbestemmingen.")

        departure_counts = df_departures[col_to_count].dropna().value_counts().nlargest(10).reset_index()
        departure_counts.columns = ['Bestemming', 'Aantal vluchten']

        if not departure_counts.empty:
            fig_top10_dep = px.bar(
                departure_counts,
                x='Aantal vluchten',
                y='Bestemming',
                orientation='h',
                title='Top 10 Vertrekbestemmingen',
                labels={'Aantal vluchten': 'Aantal vertrekken', 'Bestemming': 'Bestemming'},
                text='Aantal vluchten'
            )
            fig_top10_dep.update_layout(yaxis={'categoryorder':'total ascending'})
            return fig_top10_dep
        else:
            st.warning("Geen geldige data voor top 10 vertrekbestemmingen.")
            return None
    else:
        st.warning("Kolom 'destination_airport_iata', 'flightDirection' of geen verwerkte data beschikbaar. Kan Top 10 Vertrekbestemmingen niet plotten.")
        return None

def create_cancellation_plot(df_processed):
    """
    Creëert een pie chart van de verhouding tussen operationele en niet-operationele (geannuleerde) vluchten.
    """
    if 'isOperationalFlight' in df_processed.columns:
        df_processed['flight_status'] = df_processed['isOperationalFlight'].map({True: 'Operationeel', False: 'Niet-Operationeel (Geannuleerd)'})
        status_counts = df_processed['flight_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Aantal vluchten']

        if not status_counts.empty:
            fig_cancellation = px.pie(
                status_counts,
                values='Aantal vluchten',
                names='Status',
                title='Verhouding Operationele vs. Geannuleerde Vluchten',
                hole=0.4
            )
            fig_cancellation.update_traces(textposition='inside', textinfo='percent+label')
            return fig_cancellation
        else:
            st.warning("Geen geldige data voor vluchtstatus (operationeel/geannuleerd).")
            return None
    else:
        st.warning("Kolom 'isOperationalFlight' niet gevonden. Kan annuleringsplot niet maken.")
        return None

@st.cache_resource(show_spinner="Laden van luchthaven gegevens...") # Cache het laden van de airports.csv file
def load_airports_data():
    """
    Laadt de airports.csv file en retourneert de DataFrame.
    Voegt een robuuste controle toe voor de aanwezigheid van verwachte kolommen
    en waarschuwt de gebruiker als de CSV niet correct is geparsed.
    """
    try:
        airports_df = pd.read_csv('airports.csv')

        required_cols = ['iata_code', 'name', 'latitude_deg', 'longitude_deg']
        if not all(col in airports_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in airports_df.columns]
            st.error(f"Fout: De volgende benodigde kolommen zijn niet gevonden in 'airports.csv': {', '.join(missing_cols)}.")
            st.info("Dit kan komen doordat het 'airports.csv' bestand niet correct is opgemaakt (bijv. verkeerde scheidingstekens), waardoor Pandas de kolommen niet goed kan inlezen.")
            st.dataframe(airports_df.head(1))
            return pd.DataFrame()

        airports_df = airports_df.dropna(subset=required_cols)

        if airports_df['latitude_deg'].empty or airports_df['longitude_deg'].empty:
            st.warning("Na het laden en opschonen van 'airports.csv', zijn er geen geldige luchthaven met coördinaten overgebleven.")
            st.info("Dit kan betekenen dat je 'airports.csv' geen geldige rijen met latitude/longitude bevat of dat de waarden niet numeriek zijn.")

        return airports_df
    except FileNotFoundError:
        st.error("Fout: 'airports.csv' niet gevonden. Kan luchthavengegevens niet laden voor geoplot.")
        st.info("Zorg ervoor dat 'airports.csv' in dezelfde map staat als de Streamlit-app.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Een onverwachte fout trad op bij het laden of verwerken van 'airports.csv': {e}")
        return pd.DataFrame()

def create_geoplot_hourly(df_processed, selected_hour, ams_lat, ams_lon, selected_flight_direction):
    """
    Creëert een geoplot met lijnen naar bestemmingen voor een specifiek uur,
    gebruikmakend van Scattergeo voor een bolvormige, zoombare kaart gecentreerd op AMS.
    Inclusief filter voor vluchtrichting.
    """
    if 'estimatedLandingTime' not in df_processed.columns:
        st.error("Kolom 'estimatedLandingTime' is niet beschikbaar voor geoplot. Zorg dat de data correct is verwerkt.")
        return None, pd.DataFrame() # Return empty DataFrame as well

    df_temp = df_processed.copy()
    
    # Filter by flight direction first
    if selected_flight_direction == "A": # Arrivals
        df_filtered_direction = df_temp[df_temp['flightDirection'] == 'A'].copy()
        plot_title_direction = "Aankomsten naar Schiphol"
    elif selected_flight_direction == "D": # Departures
        df_filtered_direction = df_temp[df_temp['flightDirection'] == 'D'].copy()
        plot_title_direction = "Vertrekken vanaf Schiphol"
    else: # All ("Alle")
        df_filtered_direction = df_temp.copy()
        plot_title_direction = "Vluchten van/naar Schiphol"

    # Then filter by hour
    # Ensure estimatedLandingTime is datetime type for hour extraction
    if not pd.api.types.is_datetime64_any_dtype(df_filtered_direction['estimatedLandingTime']):
        st.error("estimatedLandingTime is geen datetime type in de verwerkte data. Kan uur niet extraheren voor geoplot.")
        st.info("Controleer de 'Data Laden en Verwerken' pagina voor parsing errors.")
        return None, pd.DataFrame() # Return empty DataFrame as well

    df_hour = df_filtered_direction[df_filtered_direction['estimatedLandingTime'].dt.hour == selected_hour].copy()

    df_plot = df_hour.dropna(subset=['destination_latitude', 'destination_longitude', 'destination_full_name'])

    if df_plot.empty:
        st.warning(f"Geen {plot_title_direction.lower()} gevonden om {selected_hour}:00 uur met geldige locatiegegevens om te plotten.")
        return None, pd.DataFrame() # Return empty DataFrame as well

    fig = go.Figure()

    # Add flight paths
    for index, row in df_plot.iterrows():
        # Adjust start/end points based on flight direction for lines
        if row['flightDirection'] == 'A': # Arrival: destination to AMS
            start_lon, start_lat = row['destination_longitude'], row['destination_latitude']
            end_lon, end_lat = ams_lon, ams_lat
        else: # Departure: AMS to destination
            start_lon, start_lat = ams_lon, ams_lat
            end_lon, end_lat = row['destination_longitude'], row['destination_latitude']

        fig.add_trace(go.Scattergeo(
            lon=[start_lon, end_lon],
            lat=[start_lat, end_lat],
            mode="lines",
            line=dict(width=1, color='red'),
            opacity=0.6,
            name=f"Vlucht {row['flightName']}",
            hoverinfo='text',
            text=f"Vlucht: {row['flightName']}<br>Richting: {'Aankomst' if row['flightDirection'] == 'A' else 'Vertrek'}<br>Bestemming: {row['destination_full_name']}<br>Type: {row.get('main_aircraft_type', 'N/A')}",
            showlegend=False # Avoid too many legends for individual lines
        ))

    # Add Schiphol marker
    fig.add_trace(go.Scattergeo(
        lon=[ams_lon],
        lat=[ams_lat],
        mode="markers",
        marker=dict(size=10, color='blue', symbol='star'),
        name='Schiphol (AMS)',
        hoverinfo='text',
        text='Amsterdam Airport Schiphol (AMS)'
    ))

    # Add destination markers
    # Filter for unique destinations to avoid overlapping markers
    unique_destinations = df_plot.groupby(['destination_full_name', 'destination_latitude', 'destination_longitude']).size().reset_index(name='count')
    
    fig.add_trace(go.Scattergeo(
        lon=unique_destinations['destination_longitude'],
        lat=unique_destinations['destination_latitude'],
        mode="markers",
        marker=dict(size=5, color='green', symbol='circle'),
        name='Bestemmingen',
        hoverinfo='text',
        text=unique_destinations.apply(lambda r: f"{r['destination_full_name']} ({r['count']} vluchten)", axis=1)
    ))

    fig.update_layout(
        title_text=f"{plot_title_direction} om {selected_hour}:00 uur",
        hovermode='closest',
        showlegend=True,
        # Adjust legend properties for compactness
        legend=dict(
            orientation="h", # Horizontal legend
            yanchor="bottom",
            y=0.99, # Place at top, just under title
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="Black",
            borderwidth=1,
            font=dict(size=10) # Smaller font
        ),
        geo=dict(
            scope='world',
            projection_type='orthographic', # Changed to orthographic projection
            center=dict(lat=ams_lat, lon=ams_lon), # Center on AMS
            projection_scale=1.5, # Adjust this value to zoom in/out (higher = more zoomed in)
            showland=True,
            landcolor="rgb(243, 243, 243)",
            countrycolor="rgb(204, 204, 204)",
            showocean=True,
            oceancolor="rgb(188, 222, 240)",
            showcountries=True,
            showcoastlines=True,
            coastlinecolor="rgb(0,0,0)",
            coastlinewidth=0.5
        ),
        margin={"r":0,"t":50,"l":0,"b":0}, # Adjust margins to give more space to the map
        height=700 # Keep height consistent
    )
    return fig, df_plot # Return df_plot as well for the table below

# --- Streamlit UI Hoofdpagina ---

st.title("Schiphol Data Dashboard")
st.write("Welkom bij het interactieve dashboard voor Schiphol vluchtdata.")

# Laad airports_df globaal (en cache deze)
airports_data_df = load_airports_data()
# Sla airports_data_df op in session_state zodat andere functies deze kunnen benaderen
st.session_state['airports_data_df'] = airports_data_df

# Zoek Schiphol's coördinaten
ams_coords = airports_data_df[airports_data_df['iata_code'] == 'AMS']
if not ams_coords.empty:
    AMS_LAT = ams_coords['latitude_deg'].iloc[0]
    AMS_LON = ams_coords['longitude_deg'].iloc[0]
else:
    AMS_LAT, AMS_LON = 52.3086, 4.7638
    st.warning("Schiphol (AMS) coördinaten niet gevonden in 'airports.csv'. Gebruikt standaardwaarden.")


# Sectie selectie via radio buttons in de sidebar
st.sidebar.header("Navigatie")
page_selection = st.sidebar.radio(
    "Ga naar:",
    ("Data Laden en Verwerken", "Vlucht Plots", "Geoplot Vluchtroutes")
)

# --- Inhoud voor "Data Laden en Verwerken" Sectie ---
if page_selection == "Data Laden en Verwerken":
    st.subheader("Data Laden en Voorbereiden")

    num_pages_to_load = st.slider("Aantal pagina's om te laden:", 1, 50, 10)

    if st.button("Start Laden en Verwerken"):
        with st.spinner("Data aan het laden en verwerken..."):
            raw_flight_data = get_schiphol_flight_data(num_pages_to_load)

        if raw_flight_data:
            st.success(f"Ruwe data succesvol geladen! Totaal {len(raw_flight_data)} vluchten.")
            st.info("Start verwerking van de data...")
            df_processed = process_flight_dataframe(pd.DataFrame(raw_flight_data), airports_data_df)
            st.success(f"Dataverwerking voltooid! DataFrame heeft nu {len(df_processed)} rijen.")

            st.session_state['schiphol_df'] = df_processed

            st.subheader("Voorbeeld van de Verwerkte Data:")
            display_cols = [
                'flightName', 'flightDirection', 'estimatedLandingTime', 'arrival_hour',
                'scheduleDateTime', 'departure_hour', 'actualLandingTime', 'delay_minutes',
                'destination', 'destination_full_name', 'origin_airport_iata', 'destination_airport_iata',
                'main_aircraft_type', 'pier', 'gate'
            ]
            display_cols_exist = [col for col in df_processed.columns if col in display_cols]
            st.dataframe(df_processed[display_cols_exist].head(10))

            st.subheader("DataFrame Informatie:")
            st.text("Type gegevens per kolom:")
            st.dataframe(df_processed.dtypes.astype(str))
            st.text("Statistische samenvatting (numerieke kolommen):")
            st.dataframe(df_processed.describe())
        else:
            st.error("Geen data geladen uit de Schiphol API.")

# --- Inhoud voor "Vlucht Plots" Sectie ---
elif page_selection == "Vlucht Plots":
    st.subheader("Analyse en Visualisaties")

    if 'schiphol_df' not in st.session_state:
        st.warning("Geen data beschikbaar. Ga eerst naar de sectie 'Data Laden en Verwerken' om de data te laden.")
    else:
        df = st.session_state['schiphol_df']

        if df.empty:
            st.warning("De geladen DataFrame is leeg. Laad data op de 'Data Laden en Verwerken' pagina.")
        else:
            # Filter voor vluchtrichting voor de tijdplot
            selected_direction = st.selectbox(
                "Filter op Vluchtrichting voor tijdplot:",
                ("Totaal", "Aankomsten", "Vertrekken")
            )
            st.markdown("---")

            # Row 1: Vluchtstatus & Gemiddelde Vertraging
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Vluchtstatus: Operationeel vs. Geannuleerd")
                fig_cancellation = create_cancellation_plot(df)
                if fig_cancellation:
                    st.plotly_chart(fig_cancellation, use_container_width=True)
            with col2:
                st.subheader("Gemiddelde Vertraging per Uur")
                fig_delay = create_delay_over_day_plot(df)
                if fig_delay:
                    st.plotly_chart(fig_delay, use_container_width=True)

            st.markdown("---")

            # Row 2: Vluchtenactiviteit per uur (breed)
            st.subheader("Vluchtenactiviteit per uur")
            fig_traffic = create_hourly_flight_traffic_plot(df, selected_direction)
            if fig_traffic:
                st.plotly_chart(fig_traffic, use_container_width=True)

            st.markdown("---")

            # Row 3: Vliegtuigtypes (alleenstaand)
            st.subheader("Verdeling Vliegtuigtypes (IATA Main)")
            fig_pie = create_aircraft_type_pie_plot(df)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")

            # Row 4: Top 10 Aankomsten & Vertrekbestemmingen (naast elkaar)
            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Top 10 Aankomstplaatsen")
                fig_top10 = create_top_10_destinations_plot(df)
                if fig_top10:
                    st.plotly_chart(fig_top10, use_container_width=True)
            with col4:
                st.subheader("Top 10 Vertrekbestemmingen")
                fig_top10_dep = create_top_10_departures_plot(df)
                if fig_top10_dep:
                    st.plotly_chart(fig_top10_dep, use_container_width=True)


elif page_selection == "Geoplot Vluchtroutes":
    st.subheader("Interactieve Vluchtroutes op de Wereldkaart")

    if 'schiphol_df' not in st.session_state:
        st.warning("Geen data beschikbaar. Ga eerst naar de sectie 'Data Laden en Verwerken' om de data te laden.")
    else:
        df = st.session_state['schiphol_df']

        if df.empty:
            st.warning("De geladen DataFrame is leeg. Laad data op de 'Data Laden en Verwerken' pagina.")
        elif 'estimatedLandingTime' not in df.columns:
            st.warning("Kolom 'estimatedLandingTime' niet gevonden. Kan geoplot niet maken.")
            st.info("Controleer de 'Data Laden en Verwerken' pagina voor parsing errors.")
        elif 'destination_latitude' not in df.columns or 'destination_longitude' not in df.columns:
            st.warning("Locatiegegevens (breedtegraad/lengtegraad) voor bestemmingen ontbreken. Zorg dat 'airports.csv' correct is geladen en verwerkt.")
        else:
            # Filter voor vluchtrichting voor de geoplot
            selected_flight_direction = st.selectbox(
                "Filter op Vluchtrichting:",
                ("Alle", "Aankomsten", "Vertrekken"),
                key="map_flight_direction_filter"
            )
            # Map selected text to 'A', 'D', or 'All'
            direction_map = {"Aankomsten": "A", "Vertrekken": "D", "Alle": "All"}
            actual_direction = direction_map[selected_flight_direction]


            df_for_slider = df.dropna(subset=['estimatedLandingTime'])
            if not df_for_slider.empty:
                # Extra controle op dtype voor het extraheren van uren
                if pd.api.types.is_datetime64_any_dtype(df_for_slider['estimatedLandingTime']):
                    unique_hours = sorted(df_for_slider['estimatedLandingTime'].dt.hour.unique().astype(int))
                else:
                    st.warning("De kolom 'estimatedLandingTime' is geen datetime-type, kan geen unieke uren extraheren voor de slider.")
                    unique_hours = []
            else:
                unique_hours = []

            if not unique_hours:
                st.warning("Geen geldige uren gevonden in de 'estimatedLandingTime' kolom om op te filteren.")
            else:
                selected_hour = st.slider(
                    "Selecteer het uur van de dag:",
                    min_value=int(min(unique_hours)),
                    max_value=int(max(unique_hours)),
                    value=int(min(unique_hours)),
                    step=1
                )

                fig_geo, df_plot_filtered_for_map = create_geoplot_hourly(df, selected_hour, AMS_LAT, AMS_LON, actual_direction)
                if fig_geo:
                    st.plotly_chart(fig_geo, use_container_width=True)

                    st.subheader(f"Details van {selected_flight_direction.lower()} om {selected_hour}:00 uur")
                    if not df_plot_filtered_for_map.empty:
                        # Select relevant columns for display in the table
                        display_cols_table = [
                            'flightName', 'flightDirection', 'origin_airport_iata',
                            'destination_full_name', 'estimatedLandingTime', 'actualLandingTime',
                            'delay_minutes', 'main_aircraft_type', 'pier', 'gate'
                        ]
                        # Ensure columns exist before displaying
                        table_cols_exist = [col for col in display_cols_table if col in df_plot_filtered_for_map.columns]
                        st.dataframe(df_plot_filtered_for_map[table_cols_exist].sort_values('estimatedLandingTime'))
                    else:
                        st.info(f"Geen {selected_flight_direction.lower()} gevonden om {selected_hour}:00 uur in de geselecteerde data.")
