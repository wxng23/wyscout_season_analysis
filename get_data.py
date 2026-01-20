import requests
import pandas as pd
import os
from datetime import datetime
import ast
from dotenv import load_dotenv

load_dotenv()

api_version = 3
WYSCOUT_API_USERNAME = os.getenv('WYSCOUT_API_USERNAME')
WYSCOUT_API_PASSWORD = os.getenv('WYSCOUT_API_PASSWORD')
BASE_PATH = os.getenv('BASE_PATH')
SEASON_DATA_PATH = os.getenv('SEASON_DATA_PATH')
verbose = True

def make_get_request(
    request_url,
    username = WYSCOUT_API_USERNAME,
    password = WYSCOUT_API_PASSWORD,
    verbose = False
):
    response = requests.get(request_url, auth=(username, password))

    if verbose == True:
        print(f'HTTP Status Code of GET Request : {response.status_code}')

    response_data = response.json()

    if response_data == []:
        if verbose == True:
            print(f'Received Empty Response. Raising Exception')
        raise Exception('Received Empty Response')
    else:
        return response_data
    
def print_message_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_current_big_ten_season():
    comp_id = 43236
    url = f'https://apirest.wyscout.com/v{api_version}/competitions/{comp_id}/seasons?active=true'
    
    try:
        response = make_get_request(url)
        seasons = response.get('seasons', [])
        
        if seasons:
            current_id = seasons[0]['seasonId']
            season_name = seasons[0].get('season').get('name')
            print_message_with_timestamp(f"Live Season Detected: {season_name} (ID: {current_id})")
            return current_id
        else:
            print_message_with_timestamp("Warning: No active season found via API flag.")
            return None
    except Exception as e:
        print(f"Error auto-detecting season: {e}")
        return None

def get_areas(api_version = 3, verbose = False):
    if verbose == True:
        print_message_with_timestamp(f'Retrieving Areas JSON.')
    areas_api = f'https://apirest.wyscout.com/v{api_version}/areas'
    return make_get_request(request_url = areas_api, verbose = verbose)

def get_competitions(area_id, api_version = 3, verbose = False):
    if verbose == True:
        print_message_with_timestamp(f'Retrieving Competitions JSON.')
    comps_api = f'https://apirest.wyscout.com/v{api_version}/competitions?areaId={area_id}'
    return make_get_request(request_url = comps_api, verbose = verbose)

def get_seasons(competition_id, api_version = 3, verbose = False):
    if verbose == True:
        print_message_with_timestamp(f'Retrieving Seasons JSON.')
    seasons_api = f'https://apirest.wyscout.com/v{api_version}/competitions/{competition_id}/seasons'
    return make_get_request(request_url = seasons_api, verbose = verbose)

def get_season_matches(season_id, api_version = 3, verbose = False):
    if verbose == True:
        print_message_with_timestamp(f'Retrieving Season Matches JSON.')
    matches_api = f'https://apirest.wyscout.com/v{api_version}/seasons/{season_id}/matches'
    return make_get_request(request_url = matches_api, verbose = verbose)

def get_match_events(match_id, api_version = 3, verbose = False):
    if verbose == True:
        print_message_with_timestamp(f'Retrieving Events JSON.')
    match_events_api = f'https://apirest.wyscout.com/v{api_version}/matches/{match_id}/events?fetch=teams,players,match,coaches,referees,formations,substitutions'
    return make_get_request(request_url = match_events_api, verbose = verbose)

def clean_and_save(events_list, filename):
    if not events_list:
        print("No events found to save.")
        return
    
    df = pd.DataFrame(events_list)
    # cols_to_drop = ['matchTimestamp', 'videoTimestamp', 'relatedEventId', 'possession']
    # df = df.drop(columns=cols_to_drop, errors='ignore')
    
    output_path = os.path.join(BASE_PATH, filename)
    df.to_csv(output_path, index=False)
    print(f"--- Success! Saved {len(df)} events to {output_path} ---")

def getSeason(season_id):
    """Retrieves events for every team in the season."""
    print(f"Starting Full Season Download (ID: {season_id})")
    matches = get_season_matches(season_id).get('matches', [])
    all_events = []
    
    for match in matches:
        try:
            data = get_match_events(match['matchId'])
            all_events.extend(data.get('events', []))
            print(f"Downloaded: {match.get('label')}")
        except Exception as e:
            print(f"Skipping {match['matchId']}: {e}")
            
    clean_and_save(all_events, 'seasonEvents25.csv')

def getUmichGame(season_id):
    """Retrieves all events (Both Teams) for only Michigan matches."""
    print(f"Retrieving all Michigan Games (ID: {season_id})")
    matches = get_season_matches(season_id).get('matches', [])
    umich_events = []
    
    for match in matches:
        if "Michigan Wolverines" in match.get('label', ''):
            try:
                data = get_match_events(match['matchId'])
                umich_events.extend(data.get('events', []))
                print(f"Downloaded Michigan Match: {match.get('label')}")
            except Exception as e:
                print(f"Error on Match {match['matchId']}: {e}")
                
    clean_and_save(umich_events, 'umichGameEvents25.csv')

def getUmichOnly(season_id):
    """Retrieves only Michigan player events from Michigan matches."""
    print(f"Retrieving Michigan-Only Events (ID: {season_id})")
    matches = get_season_matches(season_id).get('matches', [])
    all_events = []
    
    for match in matches:
        if "Michigan Wolverines" in match.get('label', ''):
            try:
                data = get_match_events(match['matchId'])
                all_events.extend(data.get('events', []))
                print(f"Downloaded Michigan Match: {match.get('label')}")
            except Exception as e:
                print(f"Error: {e}")

    if all_events:
        df = pd.DataFrame(all_events)
        df['team_name'] = df['team'].apply(lambda x: x.get('name') if isinstance(x, dict) else x)
        df = df[df['team_name'].str.contains("Michigan Wolverines", na=False)]
        
        clean_and_save(df.to_dict('records'), 'umichOnlyEvents25.csv')