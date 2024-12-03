# %%
import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st
import pickle

import plotly.graph_objects as go

import streamlit as st
import pandas as pd
import pickle
from typing import Tuple, List

# Caching models
#@st.cache_resource
def load_models():
    with open('yardage_model.pkl', 'rb') as file:
        yardage_model = pickle.load(file)
    
    with open('touchdown_model.pkl', 'rb') as file:
        touchdown_model = pickle.load(file)


    with open('Receiving_Yards_Predictor/pass_volume_model.pkl', 'rb') as file:
        pass_volume_model = pickle.load(file)

    return yardage_model, touchdown_model, pass_volume_model




# Caching function for predictions
#@st.cache_data
def predict_columns(data, yardage_model, touchdown_model):
    new_predictors = [
        'air_yards', 'yardline_100', 'ydstogo',
        'down', 'pass_location', 'season', 'qb_hit', 'end_zone_target', 'distance_to_EZ_after_target'
    ]
    
    new_X = data[new_predictors]
    new_X = pd.get_dummies(new_X, columns=['pass_location'], drop_first=True)
    
    # Perform predictions
    predictions = {
        'xYards': yardage_model.predict(new_X),
        'xTDs': touchdown_model.predict(new_X),
        'xFPs': (yardage_model.predict(new_X) * 0.1) + (touchdown_model.predict(new_X) * 6) + data['cp']
    }
    
    return pd.DataFrame(predictions)

# Main processing function
#@st.cache_data
def process_data(data, yardage_model, touchdown_model):
    rec_data = data[data['air_yards'].notna() & data['receiver_player_name'].notna()]
    current_szn = rec_data[rec_data['season'] == 2024]

    # Get predictions and concatenate with current_szn
    new_columns_current = predict_columns(current_szn, yardage_model, touchdown_model)
    current_szn = pd.concat([current_szn, new_columns_current], axis=1)

    # Calculate fantasy points
    data['fantasy_points'] = (
        data['complete_pass'] * 1 +          # 1 point per completion
        data['touchdown'] * 6 +               # 6 points per touchdown
        data['yards_gained'] * 0.1            # 0.1 points per yard gained
    )

    # Prepare game by game receivers data
    games = current_szn['game_id'].unique()
    receivers_list = []

    for game in games:
        current_game = current_szn[current_szn['game_id'] == game]
        teams = current_game['posteam'].unique()

        for team in teams:
            offense = current_game[(current_game['posteam'] == team) & (current_game['pass'] == 1) & (current_game['play_type'] == 'pass')]
            
            if offense.empty:
                continue  # Skip if there's no offense data

            throws = offense[['complete_pass', 'incomplete_pass', 'interception']].sum().sum()
            team_air_yards = offense['air_yards'].sum()

            receivers = offense.groupby(['receiver_player_name', 'posteam', 'game_id', 'week'])[['pass', 'fantasy_points', 'xFPs', 'complete_pass', 'cp', 'yards_gained', 'xYards', 'air_yards', 'touchdown', 'xTDs', 'end_zone_target','fumble_lost']].sum()
            receivers['team_attempts'] = throws
            receivers['team_air_yards'] = team_air_yards
            
            receivers_list.append(receivers)

    # Concatenate all receivers data at once
    game_by_game_receivers = pd.concat(receivers_list).rename(columns={'pass': 'targets'})

    # Calculate shares and WOPR
    game_by_game_receivers['target_share'] = round(game_by_game_receivers['targets'] / game_by_game_receivers['team_attempts'], 3)
    game_by_game_receivers['air_yards_share'] = round(game_by_game_receivers['air_yards'] / game_by_game_receivers['team_air_yards'], 3)
    game_by_game_receivers['WOPR'] = round(1.5 * game_by_game_receivers['target_share'] + 0.7 * game_by_game_receivers['air_yards_share'], 3)

    return game_by_game_receivers

# Aggregating season receivers
#@st.cache_data
def aggregate_season_receivers(game_by_game_receivers):
    szn_receivers = game_by_game_receivers.reset_index().groupby(['receiver_player_name', 'posteam'])[['targets', 'fantasy_points', 'xFPs', 'complete_pass', 'cp', 'yards_gained', 'xYards', 'air_yards', 'touchdown', 'xTDs', 'end_zone_target', 'fumble_lost', 'team_attempts', 'team_air_yards']].sum()
    
    szn_receivers['target_share'] = round(szn_receivers['targets'] / szn_receivers['team_attempts'], 3)
    szn_receivers['air_yards_share'] = round(szn_receivers['air_yards'] / szn_receivers['team_air_yards'], 3)
    szn_receivers['WOPR'] = round(1.5 * szn_receivers['target_share'] + 0.7 * szn_receivers['air_yards_share'], 3)
    szn_receivers['aDOT'] = round(szn_receivers['air_yards'] / szn_receivers['targets'], 1)
    
    # Round columns
    szn_receivers[['xFPs', 'xYards', 'xTDs', 'cp']] = szn_receivers[['xFPs', 'xYards', 'xTDs', 'cp']].round(1)

    return szn_receivers.sort_values('WOPR', ascending=False).reset_index()











# Caching data loading
#@st.cache_data
def load_data():
    data_all = pd.DataFrame()
    YEARS = [2023, 2024]

    for i in YEARS:
        i_data = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/pbp/' \
                             'play_by_play_' + str(i) + '.csv.gz',
                             compression='gzip', low_memory=False)

        data_all = pd.concat([data_all, i_data])

    return data_all

# Caching data processing
#@st.cache_data
def preprocess_data(data):
    data = data.loc[(data.play_type.isin(['no_play', 'pass', 'run'])) & (data.epa.isna() == False)]
    data.loc[data['pass'] == 1, 'play_type'] = 'pass'
    data.loc[data.rush == 1, 'play_type'] = 'run'
    data.reset_index(drop=True, inplace=True)
    data = data.dropna(subset=['posteam'])

    # Creating new columns
    new_columns_data = pd.DataFrame({
        'turnover': data['interception'] + data['fumble_lost'],
        '20+_play': (data['yards_gained'] > 19).astype(int),
        'short_pass': (data['air_yards'] < 10).astype(int),
        'medium_pass': ((data['air_yards'] > 9) & (data['air_yards'] < 20)).astype(int),
        'deep_pass': (data['air_yards'] > 19).astype(int),
        'fantasy_points': (
            data['complete_pass'] * 1 +  # 1 point per completion
            data['fumble_lost'] * -2 +
            data['touchdown'] * 6 +       # 6 points per touchdown
            data['yards_gained'] * 0.1     # 0.1 points per yard gained
        ),
        'end_zone_target': (data['yardline_100'] - data['air_yards']) <= 0,
        'distance_to_EZ_after_target': data['yardline_100'] - data['air_yards'],
        'goal_to_go': (data['yardline_100'] < 10).astype(int)
    })

    # Concatenate the new columns to the original DataFrame
    data = pd.concat([data, new_columns_data], axis=1)
    return data




# Caching the function to avoid recomputation when the same game_id is used
#@st.cache_data
def wp_graph(dataframe, game_id):
    # Filter the dataframe once
    df = dataframe[dataframe['game_id'] == game_id].copy()
    
    # Vectorized calculation of seconds and minutes
    df['second#'] = np.where(df['qtr'] != 5, 3600 - df['game_seconds_remaining'], 600 - df['game_seconds_remaining'] + 3600)
    df['minute#'] = df['second#'] / 60

    # Resetting index and extracting required values for plotting
    df = df.reset_index().reset_index()
    host = df['home_team'].iloc[0]  # Use .iloc for single value extraction
    visitor = df['away_team'].iloc[0]
    week = df['week'].iloc[0]
    
    # Efficient creation of quarter marker values without .apply()
    df['quarter_marker'] = np.where(df['desc'] == 'END QUARTER', df['level_0'], np.nan)
    quarter_list = list(df['quarter_marker'].dropna().unique())

    # Creating the figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['level_0'],
        y=df['home_wp_post'],
        mode='lines+markers',
        connectgaps=True,
        text=[f"{row['desc']}; {host}: {row['total_home_score']}, {visitor}: {row['total_away_score']}" for _, row in df.iterrows()],
        hovertemplate='%{text}<extra></extra>'
    ))

    # Adding quarter and halfway line markers more efficiently
    for x in quarter_list:
        fig.add_shape(type="line", line=dict(width=1, color="red"), x0=x, x1=x, y0=df['home_wp_post'].min(), y1=1)
    fig.add_shape(type="line", line=dict(width=1, color="black"), x0=df['level_0'].min(), x1=df['level_0'].max(), y0=0.5, y1=0.5)

    # Set layout properties
    fig.update_layout(
        title_text=f"{host} Win Probability vs. {visitor} (Week {week})", 
        title_x=0.5,
        xaxis=dict(showticklabels=False)  # Combine axis updates into layout settings
    )

    return fig



# %%


#szn_receivers.head(50)

# %%

def game_review(game_id):
#game_id = '2023_02_MIN_PHI'

    yardage_model, touchdown_model = load_models()[:2]

    # Load and preprocess data
    data_all = load_data()
    data = preprocess_data(data_all)
    
    game_by_game_receivers = process_data(data, yardage_model, touchdown_model)
    szn_receivers = aggregate_season_receivers(game_by_game_receivers)




    game = data[data['game_id']==game_id]

# %%
    host = game['home_team'].max()
    visitor = game['away_team'].max()
    home_score = game['home_score'].max()
    away_score = game['away_score'].max()
    score_dict = {host:home_score,visitor:away_score}


    game['total_points'] = game['posteam'].map(score_dict).astype(int)


# %%
    game_db = game.groupby('posteam').agg({'total_points':'max','posteam':'count','epa':['mean','sum'],'success':'mean','pass_oe':'mean','yards_gained':['mean','sum'],'turnover':'sum'}).round(2)

    game_db = game_db.rename(columns={'posteam':'plays'})

    pass_db = game[game['pass']==1].rename(columns={'epa':'pass_epa'})

    pass_db = pass_db.groupby('posteam').agg({'pass_epa':['mean','sum']})

    rush_db = game[game['rush']==1].rename(columns={'epa':'rush_epa'})

    rush_db = rush_db.groupby('posteam').agg({'rush_epa':['mean','sum']})

    game_db1 = game_db.merge(pass_db,right_index=True,left_index=True).merge(rush_db,right_index=True,left_index=True).T.round(2)



# %%
    def epa_retriever(category):
        if category == 'turnover':
            new_df = game[game['turnover']==1]
            to_df = new_df.groupby('posteam').agg({'epa':'sum'}).round(2)
            to_df = to_df.reset_index()
            if len(to_df['posteam'])==2:
                host_lost = to_df[to_df['posteam']==host]['epa'].values[0]
                visitor_lost = to_df[to_df['posteam']==visitor]['epa'].values[0]
                return host_lost - visitor_lost
            elif len(to_df['posteam'])==1:
                if to_df['posteam'][0] == host:
                    to_lost = to_df['epa'].values[0]
                if to_df['posteam'][0] == visitor:
                    to_lost = to_df['epa'].values[0] * -1
                return to_lost
        if category == 'late_downs':
            new_df = game[game['down']>2]
            to_df = new_df.groupby('posteam').agg({'epa':'sum'}).round(2)
            to_df = to_df.reset_index()
            host_lost = to_df[to_df['posteam']==host]['epa'].values[0]
            visitor_lost = to_df[to_df['posteam']==visitor]['epa'].values[0]
            return host_lost - visitor_lost
        if category == 'red_zone':
            new_df = game[game['yardline_100']<21]
            to_df = new_df.groupby('posteam').agg({'epa':'sum'}).round(2)
            to_df = to_df.reset_index()
            if len(to_df['posteam'])==2:
                host_lost = to_df[to_df['posteam']==host]['epa'].values[0]
                visitor_lost = to_df[to_df['posteam']==visitor]['epa'].values[0]
                return host_lost - visitor_lost
            elif len(to_df['posteam'])==1:
                if to_df['posteam'][0] == host:
                    to_lost = to_df['epa'].values[0]
                if to_df['posteam'][0] == visitor:
                    to_lost = to_df['epa'].values[0] * -1
        if category == 'other':
            misc = data_all[data_all['game_id']==game_id]
            misc = misc[(misc['play_type']!='qb_kneel') & (misc['play_type']!='run') & (misc['play_type']!='pass')]
            to_df = misc.groupby('posteam').agg({'epa':'sum'}).round(2)
            to_df = to_df.reset_index()
            if len(to_df['posteam'])==2:
                host_lost = to_df[to_df['posteam']==host]['epa'].values[0]
                visitor_lost = to_df[to_df['posteam']==visitor]['epa'].values[0]
                return host_lost - visitor_lost
            elif len(to_df['posteam'])==1:
                if to_df['posteam'][0] == host:
                    to_lost = to_df['epa'].values[0]
                if to_df['posteam'][0] == visitor:
                    to_lost = to_df['epa'].values[0] * -1


        






# %%
    home_mov = game_db[game_db.index==host]['total_points']['max'].values[0] - game_db[game_db.index==visitor]['total_points']['max'].values[0]

    home_epa_diff = game_db[game_db.index==host]['epa']['sum'].values[0] - game_db[game_db.index==visitor]['epa']['sum'].values[0]

    home_turnover_diff = epa_retriever('turnover')

    home_late_down_diff = epa_retriever('late_downs')

    home_red_zone_diff = epa_retriever('red_zone')

    home_misc_diff = epa_retriever('other')

    points_list = [round(home_misc_diff,2), home_red_zone_diff, home_late_down_diff, home_turnover_diff, home_epa_diff, home_mov]


#margin_of_victory

# %%
    points_df = pd.DataFrame()
    points_df['difference (home team)'] = ['epa_ST/penalties','epa_red_zone','epa_late_downs','epa_turnovers','epa_all','points_for']
    points_df['epa'] = points_list


# Assuming points_df is your DataFrame
    points_fig = px.bar(points_df, x='epa', y='difference (home team)', orientation='h')

    #fig.show()




# %%
    overall_sr = game.groupby('posteam').agg({'success':'mean'})

    early_down = game[game['down']<3].groupby('posteam').agg({'success':'mean'}).rename(columns={'success':'early down'})

    late_down = game[game['down']>2].groupby('posteam').agg({'success':'mean'}).rename(columns={'success':'late down'})

    short_yardage = game[game['ydstogo']<3].groupby('posteam').agg({'success':'mean'}).rename(columns={'success':'short yardage'})

    redzone = game[game['yardline_100']<21].groupby('posteam').agg({'success':'mean'}).rename(columns={'success':'red zone'})

    if redzone.shape[0] == 2:
        pass
    else: 
        t_list = []
        for team in [host,visitor]:
            if team not in list(redzone.index):
                t_list.append(team)
                ndf = pd.DataFrame()
                ndf['posteam'] = t_list
                ndf['red zone'] = 0
                ndf.set_index('posteam',inplace=True)
            
                redzone = pd.concat([redzone,ndf])

    
    success_rate = overall_sr.merge(early_down,right_index=True,left_index=True).merge(late_down,right_index=True,left_index=True).merge(short_yardage,right_index=True,left_index=True).merge(redzone, right_index=True,left_index=True)
    success_rate = overall_sr.merge(early_down,right_index=True,left_index=True).merge(late_down,right_index=True,left_index=True).merge(short_yardage,right_index=True,left_index=True).merge(redzone, right_index=True,left_index=True)

# %%
    team_success_rate = success_rate.reset_index()


    visitor_sr = team_success_rate[team_success_rate['posteam']==visitor].round(3).set_index('posteam')
    host_sr = team_success_rate[team_success_rate['posteam']==host].round(3).set_index('posteam')

# %%
    lg_success = data['success'].mean()

    lg_early = data[data['down']<3]['success'].mean()

    lg_late = data[data['down']>2]['success'].mean()

    lg_short = data[data['ydstogo']<3]['success'].mean()

    lg_red = data[data['yardline_100']<21]['success'].mean()


# %%
    last_szn = data[data['season']==2024]

    szn_sr = last_szn.groupby('posteam').agg({'success':'mean'}).reset_index()

    szn_early = last_szn[last_szn['down']<3].groupby('posteam').agg({'success':'mean'}).reset_index()

    szn_late = last_szn[last_szn['down']>2].groupby('posteam').agg({'success':'mean'}).reset_index()

    szn_short = last_szn[last_szn['ydstogo']<3].groupby('posteam').agg({'success':'mean'}).reset_index()
    szn_red = last_szn[last_szn['yardline_100']<21].groupby('posteam').agg({'success':'mean'}).reset_index()




    szn_sr_d = last_szn.groupby('defteam').agg({'success':'mean'}).reset_index()

    szn_early_d = last_szn[last_szn['down']<3].groupby('defteam').agg({'success':'mean'}).reset_index()

    szn_late_d = last_szn[last_szn['down']>2].groupby('defteam').agg({'success':'mean'}).reset_index()

    szn_short_d = last_szn[last_szn['ydstogo']<3].groupby('defteam').agg({'success':'mean'}).reset_index()
    szn_red_d = last_szn[last_szn['yardline_100']<21].groupby('defteam').agg({'success':'mean'}).reset_index()


# %%
    success_all = data.groupby(['posteam','game_id']).agg({'success':'mean'})

    success_all['sr_perc'] = success_all['success'].rank(pct=True).round(2)

    sr_perc = str(int((success_all[success_all.index==(host, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_early = data[data['down']<3].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_early['sr_perc'] = success_early['success'].rank(pct=True).round(2)

    sr_early_perc = str(int((success_early[success_early.index==(host, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_late = data[data['down']>2].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_late['sr_perc'] = success_late['success'].rank(pct=True).round(2)

    sr_late_perc = str(int((success_late[success_late.index==(host, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_short = data[data['ydstogo']<3].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_short['sr_perc'] = success_short['success'].rank(pct=True).round(2)

    sr_short_perc = str(int((success_short[success_short.index==(host, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_red = data[data['yardline_100']<21].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_red['sr_perc'] = success_red['success'].rank(pct=True).round(2)

    if host not in list(success_red.reset_index()[success_red.reset_index()['game_id']==game_id]['posteam']):
        sr_red_perc = "No red zone attempts"
    else: 
        sr_red_perc = str(int((success_red[success_red.index==(host, game_id)]['sr_perc']*100).values[0]))+"th percentile"

# %%
    host_percentile_list = [sr_perc, sr_early_perc, sr_late_perc, sr_short_perc, sr_red_perc]



# %%
    success_all = data.groupby(['posteam','game_id']).agg({'success':'mean'})

    success_all['sr_perc'] = success_all['success'].rank(pct=True).round(2)

    sr_perc = str(int((success_all[success_all.index==(visitor, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_early = data[data['down']<3].groupby(['posteam','game_id']).agg({'success':'mean'}) 

    success_early['sr_perc'] = success_early['success'].rank(pct=True).round(2)

    sr_early_perc = str(int((success_early[success_early.index==(visitor, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_late = data[data['down']>2].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_late['sr_perc'] = success_late['success'].rank(pct=True).round(2)

    sr_late_perc = str(int((success_late[success_late.index==(visitor, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_short = data[data['ydstogo']<3].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_short['sr_perc'] = success_short['success'].rank(pct=True).round(2)

    sr_short_perc = str(int((success_short[success_short.index==(visitor, game_id)]['sr_perc']*100).values[0]))+"th percentile"


    success_red = data[data['yardline_100']<21].groupby(['posteam','game_id']).agg({'success':'mean'})

    success_red['sr_perc'] = success_red['success'].rank(pct=True).round(2)

    if visitor not in list(success_red.reset_index()[success_red.reset_index()['game_id']==game_id]['posteam']):
        sr_red_perc = "No red zone attempts"
    else: 
        sr_red_perc = str(int((success_red[success_red.index==(visitor, game_id)]['sr_perc']*100).values[0]))+"th percentile"

# %%
    visitor_percentile_list = [sr_perc, sr_early_perc, sr_late_perc, sr_short_perc, sr_red_perc]


# %%


    off_avg_sr = szn_sr[szn_sr['posteam']==host]['success'].values[0]
    off_early = szn_early[szn_early['posteam']==host]['success'].values[0]
    off_late = szn_late[szn_late['posteam']==host]['success'].values[0]
    off_short = szn_short[szn_short['posteam']==host]['success'].values[0]
    off_red = szn_red[szn_red['posteam']==host]['success'].values[0]

    def_avg_sr = szn_sr_d[szn_sr_d['defteam']==visitor]['success'].values[0]
    def_early = szn_early_d[szn_early_d['defteam']==visitor]['success'].values[0]
    def_late = szn_late_d[szn_late_d['defteam']==visitor]['success'].values[0]
    def_short = szn_short_d[szn_short_d['defteam']==visitor]['success'].values[0]
    def_red = szn_red_d[szn_red_d['defteam']==visitor]['success'].values[0]


    columns = ['overall', 'early down', 'late down', 'short yardage', 'red zone']
    bar_width = 0.25
    r = np.arange(len(columns))
    line_values = [lg_success,lg_early,lg_late,lg_short,lg_red]
    off_avg = [off_avg_sr,off_early,off_late,off_short,off_red]
    def_avg = [def_avg_sr,def_early,def_late,def_short,def_red]

    fig2 = go.Figure()

    for i, idx in enumerate(host_sr.index):
        fig2.add_trace(go.Bar(x=columns, y=host_sr.loc[idx], name=str(idx)))

    for i, val in enumerate(line_values):
        fig2.add_shape(type="line", xref="x", x0=i-0.5, y0=val, x1=i+0.5, y1=val,
                  line=dict(color="Blue", width=0.75, dash="dash"))
    
    for i, value in enumerate(off_avg):
        fig2.add_shape(type="line", xref="x1", x0=i-0.5, y0=value, x1=i+0.5, y1=value,
                  line=dict(color="Green", width=0.75, dash="dash"))

    for i, value in enumerate(def_avg):
        fig2.add_shape(type="line", xref="x1", x0=i-0.5, y0=value, x1=i+0.5, y1=value,
                  line=dict(color="Red", width=0.75, dash="dash"))
    


    for i in range(len(columns)):
        fig2.add_annotation(x=columns[i], y=host_sr.loc[host_sr.index[0]][i]+0.01,
                       text=host_percentile_list[i],
                       showarrow=False)    

    fig2.update_layout(barmode='group', title_text=f"{host} Offense Week {game['week'].max()} -- Success Rate")
    fig2.update_yaxes(range=[0.2, 0.7])
    #fig2.show()


# %%


    off_avg_sr = szn_sr[szn_sr['posteam']==visitor]['success'].values[0]
    off_early = szn_early[szn_early['posteam']==visitor]['success'].values[0]
    off_late = szn_late[szn_late['posteam']==visitor]['success'].values[0]
    off_short = szn_short[szn_short['posteam']==visitor]['success'].values[0]
    off_red = szn_red[szn_red['posteam']==visitor]['success'].values[0]

    def_avg_sr = szn_sr_d[szn_sr_d['defteam']==host]['success'].values[0]
    def_early = szn_early_d[szn_early_d['defteam']==host]['success'].values[0]
    def_late = szn_late_d[szn_late_d['defteam']==host]['success'].values[0]
    def_short = szn_short_d[szn_short_d['defteam']==host]['success'].values[0]
    def_red = szn_red_d[szn_red_d['defteam']==host]['success'].values[0]


    columns = ['overall', 'early down', 'late down', 'short yardage', 'red zone']
    bar_width = 0.25
    r = np.arange(len(columns))
    line_values = [lg_success,lg_early,lg_late,lg_short,lg_red]
    off_avg = [off_avg_sr,off_early,off_late,off_short,off_red]
    def_avg = [def_avg_sr,def_early,def_late,def_short,def_red]

    fig3 = go.Figure()

    for i, idx in enumerate(visitor_sr.index):
        fig3.add_trace(go.Bar(x=columns, y=visitor_sr.loc[idx], name=str(idx)))

    for i, val in enumerate(line_values):
        fig3.add_shape(type="line", xref="x", x0=i-0.5, y0=val, x1=i+0.5, y1=val,
                  line=dict(color="Blue", width=0.75, dash="dash"))
    
    for i, value in enumerate(off_avg):
        fig3.add_shape(type="line", xref="x1", x0=i-0.5, y0=value, x1=i+0.5, y1=value,
                  line=dict(color="Green", width=0.75, dash="dash"))

    for i, value in enumerate(def_avg):
        fig3.add_shape(type="line", xref="x1", x0=i-0.5, y0=value, x1=i+0.5, y1=value,
                  line=dict(color="Red", width=0.75, dash="dash"))


    for i in range(len(columns)):
        fig3.add_annotation(x=columns[i], y=visitor_sr.loc[visitor_sr.index[0]][i]+0.01,
                       text=visitor_percentile_list[i],
                       showarrow=False)    


    fig3.update_layout(barmode='group', title_text=f"{visitor} Offense Week {game['week'].max()} -- Success Rate")
    fig3.update_yaxes(range=[0.2, 0.7])

    #fig.show()


# %%
    pass_show = game[game['pass']==1].groupby('posteam').agg({'pass':'sum','epa':['mean','sum'],'success':'mean','yards_gained':['mean','sum'],'air_yards':'mean','cpoe':'mean','interception':'sum','fumble_lost':'sum','20+_play':'sum','sack':'sum','qb_hit':'sum'}).round(2)

# %%
    passing_viz_table = game[game['pass']==1].groupby('posteam').agg({'epa':'mean','success':'mean','20+_play':'mean','sack':'mean'}).round(2)


    host_passing = passing_viz_table[passing_viz_table.index==host]

    visitor_passing = passing_viz_table[passing_viz_table.index==visitor]

# %%
    def get_league_stats(type):
        lg_df = data[data[type]==1]
        if type == 'pass':
            return lg_df['epa'].mean(), lg_df['success'].mean(), lg_df['20+_play'].mean(), lg_df['sack'].mean(),lg_df['qb_hit'].mean()
        else:
            return lg_df['epa'].mean(), lg_df['success'].mean(), lg_df['20+_play'].mean()


# %%

    #lg_epa_db, lg_pass_sr, lg_pass_20, lg_sack,lg_qb_hit = get_league_stats('pass')

    #lg_epa_rush, lg_rush_sr, lg_rush_20 = get_league_stats('rush')


# %%
    def capitalize_first_letter(s):
        return s[0].upper() + s[1:]

# %%
    def team_stats(type, team, off_def):
        team_df = last_szn[(last_szn[type]==1) & (last_szn[off_def]==team)]
        if type == 'rush':
            return team_df['epa'].mean(), team_df['success'].mean(), team_df['20+_play'].mean()
        else:
            return team_df['epa'].mean(), team_df['success'].mean(), team_df['20+_play'].mean(), team_df['sack'].mean(), team_df['qb_hit'].mean()



# %%
    def pass_or_rush_viz(offense, defense, pass_or_rush,game,df):
    
        if pass_or_rush == 'rush':
            offense_table = df[df[pass_or_rush]==1].groupby('posteam').agg({'epa':'mean','success':'mean','20+_play':'mean'}).round(2)
        else:
            offense_table = df[df[pass_or_rush]==1].groupby('posteam').agg({'epa':'mean','success':'mean','20+_play':'mean','sack':'mean','qb_hit':'mean'}).round(2)


        offense_table = offense_table[offense_table.index==offense]

    
    
    
    
        week = game.split("_")[1]
    
        if pass_or_rush == 'pass':    
            columns = ['epa_per_dropback', 'pass_success_rate', '20+_pass_plays','sack','qb_hit']
        else:
            columns = ['epa_per_designed_run','rush_success_rate','20+_designed_runs']
        bar_width = 0.25
        r = np.arange(len(columns))
        line_values = list(get_league_stats(pass_or_rush))
        off_avg = list(team_stats(pass_or_rush,offense,'posteam'))
        def_avg = list(team_stats(pass_or_rush,defense,'defteam'))

        fig4 = go.Figure()

        for i, idx in enumerate(offense_table.index):
            fig4.add_trace(go.Bar(x=columns, y=offense_table.loc[idx], name=str(idx)))

        for i, val in enumerate(line_values):
            fig4.add_shape(type="line", xref="x", x0=i-0.5, y0=val, x1=i+0.5, y1=val,line=dict(color="Blue", width=0.75, dash="dash"))
    
        for i, value in enumerate(off_avg):
            fig4.add_shape(type="line", xref="x1", x0=i-0.5, y0=value, x1=i+0.5, y1=value, line=dict(color="Green", width=0.75, dash="dash"))

        for i, value in enumerate(def_avg):
            fig4.add_shape(type="line", xref="x1", x0=i-0.5, y0=value, x1=i+0.5, y1=value, line=dict(color="Red", width=0.75, dash="dash"))

        new_df = data[data[pass_or_rush]==1]

        if pass_or_rush == 'rush':
            perc_df = new_df.groupby(['posteam','game_id']).agg({'epa':'mean','success':'mean','20+_play':'mean'}).rank(pct=True)
        else:
            perc_df = new_df.groupby(['posteam','game_id']).agg({'epa':'mean','success':'mean','20+_play':'mean','sack':'mean','qb_hit':'mean'})
            perc_df['sack'] = perc_df['sack']*-1
            perc_df['qb_hit'] = perc_df['qb_hit']*-1
            perc_df = perc_df.rank(pct=True) 
        perc_list = []
    
        for col in perc_df.columns:
            perc = str(int((perc_df[perc_df.index==(offense,game)][col]*100).values[0]))+"th percentile"
            perc_list.append(perc) 

        for i in range(len(columns)):
            fig4.add_annotation(x=columns[i], y=offense_table.loc[offense_table.index[0]][i]+0.01, text=perc_list[i], showarrow=False)    


        fig4.update_layout(barmode='group', title_text=f"{offense} Offense Week {week} -- {capitalize_first_letter(pass_or_rush)}ing")

        return fig4

# %%
    scramble_df = game[(game['pass']==1) & (game['rusher_player_name']!=False)].groupby('rusher_player_name').agg({'pass':'sum','epa':'sum','touchdown':'sum','turnover':'sum'}).round(2)


    scramble_df.rename(columns={'pass':'scrambles','epa':'scramble_epa','touchdown':'scramble_TD','turnover':'scramble_fumble'},inplace=True)

# %%    
    throws_df = game[(game['complete_pass']==1)|(game['incomplete_pass']==1)].groupby('posteam')[['epa']].sum().rename(columns={'epa':'throw_epa (w/o interceptions)'})

    sack_df = game[game['sack']==1].groupby('posteam')[['epa']].sum().rename(columns={'epa':'sack_epa'})

    turnover_df = game[game['turnover']==1].groupby('posteam')[['epa']].sum().rename(columns={'epa':'turnover_epa'})

    team_scramble_df = game[(game['pass']==1) & (game['rusher_player_name'].isna()==False)].groupby('posteam').agg({'epa':'sum'}).rename(columns={'epa':'scramble_epa'})


    passing_epa_df = throws_df.merge(team_scramble_df,right_index=True,left_index=True,how='outer').merge(sack_df,right_index=True,left_index=True, how='outer').merge(turnover_df,right_index=True,left_index=True, how='outer').fillna(0)




# %%

# %%

    fig5 = go.Figure()

    for i, idx in enumerate(passing_epa_df.index):
        fig5.add_trace(go.Bar(x=passing_epa_df.columns, y=passing_epa_df.loc[idx], name=str(idx)))



    fig5.update_layout(barmode='group', title_text=game_id+" Passing EPA Breakdown")

    #fig5.show()


# %%
    home_pass = pass_or_rush_viz(host,visitor,'pass',game_id,game)

# %%
    away_pass = pass_or_rush_viz(visitor,host,'pass',game_id,game)

# %%
    def pass_length_finder(length):
        new_df = game[game[(length+"_pass")]==1]
        new_df = new_df.groupby('posteam').agg({'pass':'sum','success':'mean','cpoe':'mean','epa':'sum'})
        new_df['cpoe'] = round(new_df['cpoe']/100,2)
        col_list = []
        for col in new_df.columns:
            col_list.append(length+"_"+col)
        new_df.columns = col_list
        return new_df

# %%
    length_df = pass_length_finder('short').merge(pass_length_finder('medium'),right_index=True,left_index=True,how='outer').merge(pass_length_finder('deep'),right_index=True,left_index=True,how='outer').round(3).fillna(0)

    length_df['total_throws'] = length_df['short_pass']+length_df['medium_pass']+length_df['deep_pass']

    length_df['short_pass%'] = length_df['short_pass']/length_df['total_throws']

    length_df['medium_pass%'] = length_df['medium_pass']/length_df['total_throws']

    length_df['deep_pass%'] = length_df['deep_pass']/length_df['total_throws']

    length_df = length_df.round(3)


    length_df['short_pass'] = length_df['short_pass'].astype(str)+" ("+(length_df['short_pass%']*100).astype(str)+"%)"

    length_df['medium_pass'] = length_df['medium_pass'].astype(str)+" ("+(length_df['medium_pass%']*100).astype(str)+"%)"

    length_df['deep_pass'] = length_df['deep_pass'].astype(str)+" ("+(length_df['deep_pass%']*100).astype(str)+"%)"


    length_df = length_df[['short_pass','short_epa','short_success','short_cpoe','medium_pass','medium_epa','medium_success','medium_cpoe','deep_pass','deep_epa','deep_success','deep_cpoe']]

# %%
    pass_df = data[data['air_yards'].isna()==False]

    short_list = [round(pass_df['short_pass'].mean()*100,1).astype(str)+"%"]


    lg_length_df = pd.DataFrame()
    lg_length_df['short_pass'] = short_list     
    lg_length_df['short_success'] = pass_df[pass_df['short_pass']==1]['success'].mean()
    lg_length_df['short_cpoe'] = 0



    lg_length_df['medium_pass'] = round(pass_df['medium_pass'].mean()*100,1).astype(str)+"%"                              
    lg_length_df['medium_success'] = pass_df[pass_df['medium_pass']==1]['success'].mean()
    lg_length_df['medium_cpoe'] = 0

    lg_length_df['deep_pass'] = round(pass_df['deep_pass'].mean()*100,1).astype(str)+"%"                             
    lg_length_df['deep_success'] = pass_df[pass_df['deep_pass']==1]['success'].mean()

    lg_length_df['deep_cpoe'] = 0


    lg_length_df['index'] = 'lg. average'

    lg_length_df.set_index('index',inplace=True)

# %%
    length_show = pd.concat([length_df, lg_length_df]).round(3)

# %%
    qb_show = game[game['pass']==1].groupby('passer_player_name').agg({'pass':'sum','epa':'mean','success':'mean','yards_gained':'mean','air_yards':'mean','cpoe':'mean','sack':'sum','turnover':'sum'}).round(2).merge(scramble_df, how='outer',right_index=True,left_index=True).fillna(0)

# %%
    host_rush = pass_or_rush_viz(host,visitor,'rush',game_id,game)

# %%
    visitor_rush = pass_or_rush_viz(visitor,host,'rush',game_id,game)

# %%
    rush_show = game[game['rush']==1].groupby('posteam').agg({'rush':'sum','epa':['mean','sum'],'success':'mean','yards_gained':['mean','sum','max'],'turnover':'sum','20+_play':'sum'}).round(2)

# %%
    rushers = game[game['rush']==1].groupby('rusher_player_name').agg({'posteam':'max','rush':'sum','epa':'sum','success':'mean','yards_gained':'sum','turnover':'sum','touchdown':'sum','goal_to_go':'sum','20+_play':'sum'}).round(2).sort_values(['posteam','rush'],ascending=False)

# %%
    game_receivers = game_by_game_receivers.reset_index()
    game_receivers['aDOT'] = round(game_receivers['air_yards']/game_receivers['targets'],1)
    receiver_show = game_receivers[game_receivers['game_id']==game_id].sort_values(['posteam','xFPs'],ascending=False)[['receiver_player_name','posteam','fantasy_points','xFPs','WOPR','targets','target_share','complete_pass','cp', 'yards_gained','xYards', 'aDOT', 'touchdown','xTDs','end_zone_target','fumble_lost']]
    receiver_show[['xFPs', 'xYards', 'xTDs','cp']] = receiver_show[['xFPs', 'xYards', 'xTDs','cp']].round(1)
    #[['receiver_player_name','posteam','WOPR','target_share','targets','complete_pass','yards_gained','aDOT','touchdown','goal_to_go']].round(2)

# %%
    misc = data_all[data_all['game_id']==game_id]
    misc = misc[(misc['play_type']!='qb_kneel') & (misc['play_type']!='run') & (misc['play_type']!='pass')]
    misc_show = misc.groupby(['posteam','play_type']).agg({'epa':'sum'}).round(2)
    

    win_prob = wp_graph(data_all,game_id)

#


    return st.write(game_db1), st.write(win_prob), st.plotly_chart(points_fig), st.plotly_chart(fig2), st.plotly_chart(fig3), st.write(pass_show), st.plotly_chart(fig5), st.plotly_chart(home_pass), st.plotly_chart(away_pass), st.write(length_show),st.write(qb_show), st.write(rush_show), st.plotly_chart(host_rush), st.plotly_chart(visitor_rush), st.write(receiver_show), st.write(rushers), st.write(misc_show)
# %%
header = st.container()


def overall_creator(data, szn, offense, defense):
    szn_df = data[data['season']==szn]

    szn_df = szn_df.loc[szn_df.season_type=='REG']

    overall = szn_df.groupby('posteam')[['epa','success','20+_play','turnover','pass_oe']].mean()
    overall = (overall - overall.mean()) / overall.std()
    overall['turnover'] = overall['turnover']*-1
    off_df = overall[overall.index==offense].round(1)

    overall_df = szn_df.groupby('defteam')[['epa','success','20+_play','turnover','pass_oe']].mean()
    overall_df = (overall_df - overall_df.mean()) / overall_df.std()
    overall_df['epa'] = overall_df['epa']*-1
    overall_df['success'] = overall_df['success']*-1
    overall_df['pass_oe'] = overall_df['pass_oe']*-1
    overall_df['20+_play'] = overall_df['20+_play']*-1

    def_df = overall_df[overall_df.index==defense].round(1)

    matchup_df = pd.concat([off_df,def_df])


    matchup_df_transposed = matchup_df.transpose()


# Create a custom color map for NFL teams
    team_colors = {
        'ARI': 'red',
        'ATL': 'black',
        'BAL': 'purple',
        'BUF': 'red',
        'CAR': 'black',
        'CHI': 'blue',
        'CIN': 'orange',
        'CLE': 'orange',
        'DAL': 'gray',
        'DEN': 'orange',
        'DET': 'gray',
        'GB': 'green',
        'HOU': 'red',
        'IND': 'blue',
        'JAX': 'teal',
        'KC': 'red',
        'LAC': 'yellow',
        'LA': 'yellow',
        'MIA': 'aqua',
        'MIN': 'purple',
        'NE': 'red',
        'NO': 'black',
        'NYG': 'blue',
        'NYJ': 'green',
        'LV': 'black',
        'PHI': 'green',
        'PIT': 'black',
        'SEA': 'blue',
        'SF': 'red',
        'TB': 'red',
        'TEN': 'red',
        'WAS': 'red',
    }

# Create a bar chart with custom colors
    fig = px.bar(
        matchup_df_transposed, 
        x=matchup_df_transposed.index, 
        y=matchup_df_transposed.columns, 
        barmode='group',
        color_discrete_map=team_colors  # Apply custom colors
    )

# Set the chart title
    if szn == 2023:
        fig.update_layout(title=f"{offense} Offense vs. {defense} Defense")
    else:
        fig.update_layout(title=f"{offense} Offense vs. {defense} Defense ({szn} stats)")


# Update the y-axis title
    fig.update_yaxes(title_text="z-score")

# Remove the x-axis title
    fig.update_xaxes(title_text="")

# Add a footnote
    footnote = "Higher is better for both offense and defense; a higher defensive PROE means opposing offenses have been more run-heavy"
    fig.add_annotation(
        text=footnote,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10),
    )

# Display the chart
    return fig

def pass_matchup(data, szn, offense, defense):
    szn_df = data[data['season']==szn]

    szn_df = szn_df.loc[szn_df.season_type=='REG']

    szn_df = szn_df[szn_df['pass']==1]
    overall1 = szn_df.groupby('posteam')[['epa','success','turnover','air_yards','20+_play','sack','qb_hit']].mean()
    overall1 = (overall1 - overall1.mean()) / overall1.std()
    overall1['turnover'] = overall1['turnover']*-1
    overall1['sack'] = overall1['sack']*-1
    overall1['qb_hit'] = overall1['qb_hit']*-1



    off_df = overall1[overall1.index==offense].round(1)

    overall_df2 = szn_df.groupby('defteam')[['epa','success','turnover','air_yards','20+_play','sack','qb_hit']].mean()
    overall_df2 = (overall_df2 - overall_df2.mean()) / overall_df2.std()
    overall_df2['epa'] = overall_df2['epa']*-1
    overall_df2['success'] = overall_df2['success']*-1
    overall_df2['air_yards'] = overall_df2['air_yards']*-1
    overall_df2['20+_play'] = overall_df2['20+_play']*-1



    def_df = overall_df2[overall_df2.index==defense].round(1)

    matchup_df = pd.concat([off_df,def_df])


    matchup_df_transposed = matchup_df.transpose()


# Create a custom color map for NFL teams
    team_colors = {
        'ARI': 'red',
        'ATL': 'black',
        'BAL': 'purple',
        'BUF': 'red',
        'CAR': 'black',
        'CHI': 'blue',
        'CIN': 'orange',
        'CLE': 'orange',
        'DAL': 'gray',
        'DEN': 'orange',
        'DET': 'gray',
        'GB': 'green',
        'HOU': 'red',
        'IND': 'blue',
        'JAX': 'teal',
        'KC': 'red',
        'LAC': 'yellow',
        'LA': 'yellow',
        'MIA': 'aqua',
        'MIN': 'purple',
        'NE': 'red',
        'NO': 'black',
        'NYG': 'blue',
        'NYJ': 'green',
        'LV': 'black',
        'PHI': 'green',
        'PIT': 'black',
        'SEA': 'blue',
        'SF': 'red',
        'TB': 'red',
        'TEN': 'red',
        'WAS': 'red',
    }

# Create a bar chart with custom colors
    fig = px.bar(
        matchup_df_transposed, 
        x=matchup_df_transposed.index, 
        y=matchup_df_transposed.columns, 
        barmode='group',
        color_discrete_map=team_colors  # Apply custom colors
    )

# Set the chart title
    if szn == 2023:
        fig.update_layout(title=f"{offense} Pass Offense vs. {defense} Pass Defense")
    else:
        fig.update_layout(title=f"{offense} Pass Offense vs. {defense} Pass Defense ({szn} stats)")


# Update the y-axis title
    fig.update_yaxes(title_text="z-score")

# Remove the x-axis title
    fig.update_xaxes(title_text="")

# Add a footnote
    footnote = "Higher is better for both offense and defense"
    fig.add_annotation(
        text=footnote,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10),
    )

# Display the chart
    return fig

def rush_matchup(data, szn, offense, defense):
    szn_df = data[data['season']==szn]

    szn_df = szn_df.loc[szn_df.season_type=='REG']

    szn_df = szn_df[szn_df['rush']==1]
    overall = szn_df.groupby('posteam')[['epa','success','turnover','20+_play']].mean()
    overall = (overall - overall.mean()) / overall.std()
    overall['turnover'] = overall['turnover']*-1

    off_df = overall[overall.index==offense].round(1)

    overall_df = szn_df.groupby('defteam')[['epa','success','turnover','20+_play']].mean()
    overall_df = (overall_df - overall_df.mean()) / overall_df.std()
    overall_df['epa'] = overall_df['epa']*-1
    overall_df['success'] = overall_df['success']*-1
    overall_df['20+_play'] = overall_df['20+_play']*-1



    def_df = overall_df[overall_df.index==defense].round(1)

    matchup_df = pd.concat([off_df,def_df])


    matchup_df_transposed = matchup_df.transpose()


# Create a custom color map for NFL teams
    team_colors = {
        'ARI': 'red',
        'ATL': 'black',
        'BAL': 'purple',
        'BUF': 'red',
        'CAR': 'black',
        'CHI': 'blue',
        'CIN': 'orange',
        'CLE': 'orange',
        'DAL': 'gray',
        'DEN': 'orange',
        'DET': 'gray',
        'GB': 'green',
        'HOU': 'red',
        'IND': 'blue',
        'JAX': 'teal',
        'KC': 'red',
        'LAC': 'yellow',
        'LA': 'yellow',
        'MIA': 'aqua',
        'MIN': 'purple',
        'NE': 'red',
        'NO': 'black',
        'NYG': 'blue',
        'NYJ': 'green',
        'LV': 'black',
        'PHI': 'green',
        'PIT': 'black',
        'SEA': 'blue',
        'SF': 'red',
        'TB': 'red',
        'TEN': 'red',
        'WAS': 'red',
    }

# Create a bar chart with custom colors
    fig = px.bar(
        matchup_df_transposed, 
        x=matchup_df_transposed.index, 
        y=matchup_df_transposed.columns, 
        barmode='group',
        color_discrete_map=team_colors  # Apply custom colors
    )

# Set the chart title
    if szn == 2024:
        fig.update_layout(title=f"{offense} Rush Offense vs. {defense} Rush Defense")
    else:
        fig.update_layout(title=f"{offense} Rush Offense vs. {defense} Rush Defense ({szn} stats)")


# Update the y-axis title
    fig.update_yaxes(title_text="z-score")

# Remove the x-axis title
    fig.update_xaxes(title_text="")

# Add a footnote
    footnote = "Higher is better for both offense and defense"
    fig.add_annotation(
        text=footnote,
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        font=dict(size=10),
    )

# Display the chart
    return fig



with header:
    st.title("FOOTBALL PREVIEW/REVIEW")
    st.write("All data from NFLVerse.")


def get_off_stats(team,data,last_or_this):




    # Filter data for the specified team
    #     
    team_data = data[(data['posteam'] == team)]

    # Calculate statistics
    epa_per_play = team_data['epa'].mean()
    success_rate = (team_data['success'] == 1).mean()
    epa_per_dropback = team_data[team_data['pass'] == 1]['epa'].mean()
    passing_success_rate = (team_data[team_data['pass'] == 1]['success'] == 1).mean()
    epa_per_rush = team_data[team_data['rush']== 1]['epa'].mean()
    rushing_success_rate = (team_data[team_data['rush'] == 1]['success'] == 1).mean()
    early_down_success_rate = (team_data[team_data['down'].isin([1, 2])]['success'] == 1).mean()
    late_down_success_rate = (team_data[team_data['down'].isin([3, 4])]['success'] == 1).mean()
    red_zone_success_rate = (team_data[team_data['yardline_100'] <= 20]['success'] == 1).mean()
    success_rate_outside_red_zone = (team_data[team_data['yardline_100'] > 20]['success'] == 1).mean()
    adot = team_data['air_yards'].mean()
    epa_lost_on_turnovers = team_data[team_data['turnover'] == 1]['epa'].sum()
    turnovers_per_play_rate = (team_data['turnover'] == 1).mean()
    proe = team_data['pass_oe'].mean()
    
    # Create table
    stats = ['EPA per play', 'Success rate', 'Pass Rate Over Expected','EPA per dropback', 'Passing success rate', 'EPA per rush', 'Rushing success rate', 'Early down success rate', 'Late down success rate', 'Red zone success rate', 'Success rate outside red zone', 'ADOT', 'EPA lost on turnovers', 'Turnovers per play rate']
    values = [epa_per_play, success_rate, proe, epa_per_dropback, passing_success_rate, epa_per_rush, rushing_success_rate, early_down_success_rate, late_down_success_rate, red_zone_success_rate, success_rate_outside_red_zone, adot, epa_lost_on_turnovers, turnovers_per_play_rate]
    
    # Calculate ranks
    teams = data['posteam'].unique()
    ranks = []
    
    for stat in stats:
        stat_values = []
        
        for t in teams:
            t_data = data[(data['posteam'] == t)]
            
            if stat == 'EPA per play':
                stat_values.append(t_data['epa'].mean())
            elif stat == 'Success rate':
                stat_values.append((t_data['success'] == 1).mean())
            elif stat == 'Pass Rate Over Expected':
                stat_values.append(t_data['pass_oe'].mean())
            elif stat == 'EPA per dropback':
                stat_values.append(t_data[t_data['pass'] == 1]['epa'].mean())
            elif stat == 'Passing success rate':
                stat_values.append((t_data[t_data['pass'] == 1]['success'] == 1).mean())
            elif stat == 'EPA per rush':
                stat_values.append(t_data[t_data['rush'] == 1]['epa'].mean())
            elif stat =='Rushing success rate':
                stat_values.append((t_data[t_data['rush']==1]['success']==1).mean())
            elif stat == 'Early down success rate':
                stat_values.append((t_data[t_data['down'].isin([1, 2])]['success'] == 1).mean())
            elif stat == 'Late down success rate':
                stat_values.append((t_data[t_data['down'].isin([3, 4])]['success'] == 1).mean())
            elif stat == 'Red zone success rate':
                stat_values.append((t_data[t_data['yardline_100'] <= 20]['success'] == 1).mean())
            elif stat == 'Success rate outside red zone':
                stat_values.append((t_data[t_data['yardline_100'] > 20]['success'] == 1).mean())
            elif stat == 'ADOT':
                stat_values.append(t_data['air_yards'].mean())
            elif stat == 'EPA lost on turnovers':
                stat_values.append(t_data[t_data['turnover']==1]['epa'].sum())
            elif stat =='Turnovers per play rate':
                stat_values.append((t_data['turnover']==1).mean())
        
        if stat in ['Turnovers per play rate']:
            rank = pd.Series(stat_values).rank(ascending=True)[teams.tolist().index(team)]
        else:
            rank = pd.Series(stat_values).rank(ascending=False)[teams.tolist().index(team)]
        ranks.append(rank)
    
    df = pd.DataFrame({'Stat': stats, 'Value': values, 'Rank': ranks})
    df['Value'] = df['Value'].round(3)
    df['Rank'] = df['Rank'].astype(int)


    team_passing = team_data.groupby('passer_player_name').agg({'pass':'sum','epa':['sum','mean'],'success':'mean','air_yards':'mean', 'cpoe':'mean','touchdown':['sum','mean'],'interception':['sum','mean']})
    team_rushing = team_data.groupby('rusher_player_name').agg({'rush':'sum','epa':['sum','mean'],'success':'mean','yards_gained':['sum','mean']})
    
    if last_or_this == 'this':
        yardage_model, touchdown_model = load_models()[0:2]

    
        game_by_game_receivers = process_data(data, yardage_model, touchdown_model)
        szn_receivers = aggregate_season_receivers(game_by_game_receivers)

        rec_data = szn_receivers.reset_index()
    
        team_receiving = rec_data[rec_data['posteam']==team]
        return df, team_passing, team_rushing, team_receiving
    if last_or_this == 'last':
        return df, team_passing, team_rushing


def get_def_stats(team,data):

    
    # Filter data for the specified team
    #team_data = data[(data['posteam'] == team) | (pbp['defteam'] == team)]
    team_data = data[(data['defteam'] == team)]

    # Calculate statistics
    epa_per_play = team_data['epa'].mean()
    success_rate = (team_data['success'] == 1).mean()
    epa_per_dropback = team_data[team_data['pass'] == 1]['epa'].mean()
    passing_success_rate = (team_data[team_data['pass'] == 1]['success'] == 1).mean()
    epa_per_rush = team_data[team_data['rush'] == 1]['epa'].mean()
    rushing_success_rate = (team_data[team_data['rush'] == 1]['success'] == 1).mean()
    early_down_success_rate = (team_data[team_data['down'].isin([1, 2])]['success'] == 1).mean()
    late_down_success_rate = (team_data[team_data['down'].isin([3, 4])]['success'] == 1).mean()
    red_zone_success_rate = (team_data[team_data['yardline_100'] <= 20]['success'] == 1).mean()
    success_rate_outside_red_zone = (team_data[team_data['yardline_100'] > 20]['success'] == 1).mean()
    adot = team_data['air_yards'].mean()
    epa_lost_on_turnovers = team_data[team_data['turnover'] == 1]['epa'].sum()
    turnovers_per_play_rate = (team_data['turnover'] == 1).mean()
    proe = team_data['pass_oe'].mean()
    
    # Create table
    stats = ['EPA per play', 'Success rate', 'Pass Rate Over Expected','EPA per dropback', 'Passing success rate', 'EPA per rush', 'Rushing success rate', 'Early down success rate', 'Late down success rate', 'Red zone success rate', 'Success rate outside red zone', 'ADOT', 'EPA lost on turnovers', 'Turnovers per play rate']
    values = [epa_per_play, success_rate, proe, epa_per_dropback, passing_success_rate, epa_per_rush, rushing_success_rate, early_down_success_rate, late_down_success_rate, red_zone_success_rate, success_rate_outside_red_zone, adot, epa_lost_on_turnovers, turnovers_per_play_rate]
    
    # Calculate ranks
    teams = data['defteam'].unique()
    ranks = []
    
    for stat in stats:
        stat_values = []
        
        for t in teams:
            t_data = data[(data['defteam'] == t)]
            
            if stat == 'EPA per play':
                stat_values.append(t_data['epa'].mean())
            elif stat == 'Success rate':
                stat_values.append((t_data['success'] == 1).mean())
            elif stat == 'Pass Rate Over Expected':
                stat_values.append(t_data['pass_oe'].mean())
            elif stat == 'EPA per dropback':
                stat_values.append(t_data[t_data['pass'] == 1]['epa'].mean())
            elif stat == 'Passing success rate':
                stat_values.append((t_data[t_data['pass'] == 1]['success'] == 1).mean())
            elif stat == 'EPA per rush':
                stat_values.append(t_data[t_data['rush'] == 1]['epa'].mean())
            elif stat =='Rushing success rate':
                stat_values.append((t_data[t_data['rush']==1]['success']==1).mean())
            elif stat == 'Early down success rate':
                stat_values.append((t_data[t_data['down'].isin([1, 2])]['success'] == 1).mean())
            elif stat == 'Late down success rate':
                stat_values.append((t_data[t_data['down'].isin([3, 4])]['success'] == 1).mean())
            elif stat == 'Red zone success rate':
                stat_values.append((t_data[t_data['yardline_100'] <= 20]['success'] == 1).mean())
            elif stat == 'Success rate outside red zone':
                stat_values.append((t_data[t_data['yardline_100'] > 20]['success'] == 1).mean())
            elif stat == 'ADOT':
                stat_values.append(t_data['air_yards'].mean())
            elif stat == 'EPA lost on turnovers':
                stat_values.append(t_data[t_data['turnover']==1]['epa'].sum())
            elif stat =='Turnovers per play rate':
                stat_values.append((t_data['turnover']==1).mean())
        
        if stat in ['Turnovers per play rate']:
            rank = pd.Series(stat_values).rank(ascending=False)[teams.tolist().index(team)]
        else:
            rank = pd.Series(stat_values).rank(ascending=True)[teams.tolist().index(team)]
        ranks.append(rank)
    
    df = pd.DataFrame({'Stat': stats, 'Value': values, 'Rank': ranks})
    df['Value'] = df['Value'].round(3)
    df['Rank'] = df['Rank'].astype(int)

    return df


def get_team_stats(team, year):




    previous = year - 1    
    # Load and preprocess data
    data_all = load_data()
    data_df = preprocess_data(data_all)
    
    
    
    data = data_df[(data_df['season'] == year) & (data_df['season_type'] == 'REG')]




    off_df = get_off_stats(team, data,'this')[0]
    year1 = off_df.copy()

    off_df = off_df.rename(columns={'Stat':f'{year} {team} Offense'})

    pass_df = get_off_stats(team,data,'this')[1]
    rush_df = get_off_stats(team,data,'this')[2]
    rec_df = get_off_stats(team,data,'this')[3]

    def_df = get_def_stats(team,data)
    year1_def = def_df.copy()
    def_df = def_df.rename(columns={'Stat':f'{year} {team} Defense'})

    last_year_stats = data_df[data_df['season']==previous]
    last_year_stats = last_year_stats.loc[last_year_stats.season_type=='REG']



    year2 = get_off_stats(team,last_year_stats,'last')[0]
    year2_def = get_def_stats(team,last_year_stats)


    year_over_year = year1.merge(year2,on='Stat')
    year_over_year = year_over_year.rename(columns={'Rank_x':year,'Rank_y':previous})

    year_over_year_def = year1_def.merge(year2_def,on='Stat')
    year_over_year_def = year_over_year_def.rename(columns={'Rank_x':year,'Rank_y':previous})


    # Assuming 'year_over_year' is your DataFrame
    df = year_over_year
    df = df.reset_index()
    df.sort_values('index',ascending=False,inplace=True)
# Create figure
    fig = go.Figure()

# Add traces for 'year' and 'previous'
    fig.add_trace(go.Scatter(
        x=df[year],
        y=df['Stat'],
        mode='markers',
        name=str(int(year)),
        text=df.apply(lambda row: f"{row['Stat']}: {row['Value_x']} ({row[year]})", axis=1),
        marker=dict(size=10, color='blue'),
        hovertemplate='%{text}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=df[previous],
        y=df['Stat'],
        mode='markers',
        name=str(int(previous)),
        text=df.apply(lambda row: f"{row['Stat']}: {row['Value_y']} ({row[previous]})", axis=1),
        marker=dict(size=10, color='red'),
        hovertemplate='%{text}<extra></extra>'

    ))

# Add lines connecting the dots
    for i in range(len(df)):
        fig.add_shape(type="line",
            x0=df[year].iloc[i], y0=df['Stat'].iloc[i], x1=df[previous].iloc[i], y1=df['Stat'].iloc[i],
            line=dict(color="RebeccaPurple",width=1.5)
        )

    # Set layout properties
    fig.update_layout(title_text=f"{team} Offense -- {int(previous)} to {int(year)}",title_x=0.5)

# Show plot
    df = year_over_year_def
    df = df.reset_index()
    df.sort_values('index',ascending=False,inplace=True)
# Create figure
    def_fig = go.Figure()

# Add traces for 'year' and 'previous'
    def_fig.add_trace(go.Scatter(
        x=df[year],
        y=df['Stat'],
        mode='markers',
        name=str(int(year)),
        text=df.apply(lambda row: f"{row['Stat']}: {row['Value_x']} ({row[year]})", axis=1),
        hovertemplate='%{text}<extra></extra>',
        marker=dict(size=10, color='blue')
    ))

    def_fig.add_trace(go.Scatter(
        x=df[previous],
        y=df['Stat'],
        mode='markers',
        name=str(int(previous)),
        text=df.apply(lambda row: f"{row['Stat']}: {row['Value_y']} ({row[previous]})", axis=1),
        hovertemplate='%{text}<extra></extra>',
        marker=dict(size=10, color='red')
    ))

# Add lines connecting the dots
    for i in range(len(df)):
        def_fig.add_shape(type="line",
            x0=df[year].iloc[i], y0=df['Stat'].iloc[i], x1=df[previous].iloc[i], y1=df['Stat'].iloc[i],
            line=dict(color="RebeccaPurple",width=1.5)
        )

    # Set layout properties
    def_fig.update_layout(title_text=f"{team} Defense -- {int(previous)} to {int(year)}",title_x=0.5)







    return off_df,def_df, pass_df, rec_df, rush_df,fig,def_fig



def preview_maker(season, team_a, team_b):

    # Load and preprocess data
    data_all = load_data()
    data = preprocess_data(data_all)
    

    #game_by_game_receivers = process_data(data, yardage_model, touchdown_model)
    #szn_receivers = aggregate_season_receivers(game_by_game_receivers)

    overall_result = overall_creator(data, season, team_a, team_b)
    overall_result2 = overall_creator(data, season, team_b, team_a)
    pass_matchup1 = pass_matchup(data, season, team_a, team_b)
    rush_matchup1 = rush_matchup(data, season, team_a, team_b)
    pass_matchup2 = pass_matchup(data, season, team_b, team_a)
    rush_matchup2 = rush_matchup(data, season, team_b, team_a)

    return overall_result, overall_result2, pass_matchup1, rush_matchup1, pass_matchup2, rush_matchup2

def df_creator(sample_df, team,spread,total):
    team_df = sample_df[sample_df.index==team]
    team_total = total/2 - spread/2


    team_df['total_line'] = total
    team_df['pos_team_total'] = team_total
    team_df['pos_spread'] = spread

    return team_df

def total_finder(home_or_away,home_total,away_total):
    if home_or_away == 'home':
        total = home_total
    else:
        total = away_total 
    return total


def receiver_simulator(chosen_team, spread, total, excluded_receiver1, excluded_receiver2, receiver_name,trailing_games,attempts_input,target_share_input):


    yardage_model, touchdown_model, pass_volume_model = load_models()




    data_all = load_data()
    data = data_all.loc[data_all.season_type=='REG']
    data = data[data['season'] == 2024]

    starting_week = (data[data['posteam']==chosen_team]['week'].max() - trailing_games)+1

    data.reset_index(drop=True, inplace=True)
    data['turnover'] = data['interception'] + data['fumble_lost']
    data = data.dropna(subset=['posteam'])
    data['inside_10'] = (data['yardline_100'] < 10).astype(int)
    data['20+_play'] = (data['yards_gained'] > 19).astype(int)
    data['short_pass'] = (data['air_yards'] < 10).astype(int)
    data['medium_pass'] = ((data['air_yards'] > 9)&(data['air_yards']<20)).astype(int)
    data['deep_pass'] = (data['air_yards'] > 19).astype(int)
    data['end_zone_target'] = (data['yardline_100'] - data['air_yards']) <= 0

    data['distance_to_EZ_after_target'] = data['yardline_100'] - data['air_yards']
    data.reset_index(drop=True, inplace=True)

    data = data[data['two_point_attempt']==0]

    data['total_plays'] = data['pass'] + data['rush']
    data.loc[data['receiver_player_name'].isna()==False, 'target']=1   
    
    # derive implied team total from betting market data
    data['home_implied_total'] = abs(data['total_line'] / 2 + data['spread_line'] / 2)
    data['away_implied_total'] = abs(data['total_line'] / 2 - data['spread_line'] / 2)
    data = data[(data['play_type']=='pass')|(data['play_type']=='run')]
    # Use list comprehension with zip for more efficient row-wise operations
    data['implied_posteam_total'] = [
    total_finder(has_ball, home_number, away_number)
        for has_ball, home_number, away_number in zip(data['posteam_type'], data['home_implied_total'], data['away_implied_total'])
    ]



    #sample = data[data['week']>=starting_week].groupby('posteam').agg({'pass':'mean','total_plays':'sum','pass_oe':'mean','game_id':'nunique'})

    sample = data[data['week']>=starting_week].groupby('posteam').agg(    
    target_total = ('target','sum'),
    pass_total=('pass', 'sum'),
    pass_rate=('pass', 'mean'),
    pass_oe=('pass_oe', 'mean'),
    plays=('total_plays', 'sum'),
    game_id = ('game_id','nunique'))




    sample['trailing_total_plays_avg'] = sample['plays']/sample['game_id']
    sample['pass_total'] = sample['pass_total']/sample['game_id']


    sample = sample.rename(columns={'pass_rate':'trailing_pass_avg','pass_total':'trailing_pass_total','pass_oe':'trailing_pass_oe_avg'})

    throws = data[data['air_yards'].notna()]

    throws = throws[throws['receiver_player_name'].notna()]
    throws = throws[throws['pass_location'].notna()]
    throws.loc[throws['receiver_player_name'].isna()==False, 'target']=1   

    
    df = throws[['receiver_player_name','receiver_player_id','posteam','pass','cp','game_id','complete_pass','inside_10','air_yards','yardline_100','ydstogo','implied_posteam_total','yards_gained','pass_touchdown','down','pass_location','week','season','home_implied_total','away_implied_total','posteam_type','qb_hit','end_zone_target', 'distance_to_EZ_after_target','target']]




    team_df = df_creator(sample,chosen_team,spread,total)
    
    if attempts_input == 0:
        predicted_attempts = pass_volume_model.predict(team_df[['trailing_pass_total','trailing_pass_avg','trailing_pass_oe_avg','trailing_total_plays_avg','total_line','pos_team_total','pos_spread']])[0]
    else:
        predicted_attempts = attempts_input


    new_columns_current = predict_columns(df, yardage_model, touchdown_model)
    current_szn = pd.concat([df, new_columns_current], axis=1)

    xYardsmean = current_szn[current_szn['receiver_player_name']==receiver_name]['xYards'].mean()

    xYards_sd = current_szn[current_szn['receiver_player_name']==receiver_name]['xYards'].std()


    szn_stats = current_szn[(current_szn['posteam']==chosen_team)].groupby('receiver_player_name').agg({'pass':'sum','xYards':'sum','game_id':'nunique','yards_gained':'sum','target':'sum'})
    team_games = current_szn[(current_szn['posteam']==chosen_team)]['game_id'].nunique()
    szn_targets_per_game = szn_stats['target'].sum()/team_games





    team_period = current_szn[(current_szn['posteam']==chosen_team)&(current_szn['week']>=starting_week)].groupby('receiver_player_name').agg({'pass':'sum','xYards':'sum','game_id':'nunique','yards_gained':'sum','target':'sum'})

    period_targets_per_game = team_period['target'].sum()/team_period['game_id'].nunique()

    team_targets = team_period[team_period.index!=excluded_receiver1]

    team_targets = team_targets[team_targets.index!=excluded_receiver2]['pass'].sum()


    team_period['target_share'] = team_period['pass']/team_targets

    team_period['xYards_game'] = team_period['xYards']/team_period['game_id']

    team_period['yards_game'] = team_period['yards_gained']/team_period['game_id']

    team_rec_df = team_period.round(2).sort_values('xYards_game',ascending=False)[['game_id','pass','target_share','xYards','xYards_game','yards_game']]



    rec_df = current_szn[(current_szn['receiver_player_name']==receiver_name)&(current_szn['posteam']==chosen_team)].groupby('week').agg({'pass':'sum','xYards':'sum','yards_gained':'sum'}).round(1)

    receiver_string = (f"Season median: {rec_df['xYards'].median()}; Last four games median: {rec_df.tail(4)['xYards'].median()}")

    # get game by game data
    rec_data = data[data['air_yards'].notna() & data['receiver_player_name'].notna()]
    current_szn_1 = rec_data[rec_data['season'] == 2024]
    
    
    games = current_szn_1['game_id'].unique()
    receivers_list = []

    for game in games:
        current_game = current_szn_1[current_szn_1['game_id'] == game]
        teams = current_game['posteam'].unique()

        for team in teams:
            offense = current_game[(current_game['posteam'] == team) & (current_game['pass'] == 1) & (current_game['play_type'] == 'pass')]
            
            if offense.empty:
                continue  # Skip if there's no offense data

            throws = offense[['complete_pass', 'incomplete_pass', 'interception']].sum().sum()

            receivers = offense.groupby(['receiver_player_name', 'posteam', 'game_id', 'week'])[['pass','complete_pass', 'cp', 'yards_gained']].sum()
            receivers['team_attempts'] = throws
            
            receivers_list.append(receivers)

    # Concatenate all receivers data at once
    game_by_game_receivers = pd.concat(receivers_list).rename(columns={'pass': 'targets'})

    # Calculate shares and WOPR
    game_by_game_receivers['target_share'] = round(game_by_game_receivers['targets'] / game_by_game_receivers['team_attempts'], 3)

    individual_df = game_by_game_receivers.reset_index()
    individual_df = individual_df[individual_df['receiver_player_name']==receiver_name]

    rec_df = rec_df.merge(individual_df,on='week')[['week','game_id','targets','xYards','yards_gained_x','target_share','team_attempts']].set_index('week').rename(columns={'yards_gained_x':'yards_gained'})



    individual_period = rec_df.copy()

    individual_period = individual_period.tail(int(trailing_games))

    if target_share_input == 0:
        rec_target_share = individual_period['targets'].sum()/individual_period['team_attempts'].sum()
    else:
        rec_target_share = target_share_input

    



    team_passes = predicted_attempts 
    player_target_rate = rec_target_share 
    yards_per_target_mean = xYardsmean
    yards_per_target_std = xYards_sd
    
    # Run simulation
    targets, yards = simulate_receiver_game(
        team_pass_attempts=team_passes,
        target_rate=player_target_rate,
        yards_mean=yards_per_target_mean,
        yards_std=yards_per_target_std
    )
    
    # Analyze results
    results = analyze_simulation_results(predicted_attempts,targets, yards)
    
    median_yards = f"Median Yards: {results['median_yards']:.1f}"




    return team_rec_df, rec_df, receiver_string, median_yards, results, predicted_attempts, period_targets_per_game, szn_targets_per_game, rec_target_share



def simulate_receiver_game(
    team_pass_attempts: int,
    target_rate: float,
    yards_mean: float,
    yards_std: float,
    num_simulations: int = 1000
) -> Tuple[List[int], List[float]]:
    """
    Simulate receiving statistics for a player based on team passing attempts and player metrics.
    
    Args:
        team_pass_attempts: Number of team pass attempts
        target_rate: Rate at which player is targeted (between 0 and 1)
        yards_mean: Mean yards per target
        yards_std: Standard deviation of yards per target
        num_simulations: Number of games to simulate
        
    Returns:
        Tuple containing:
        - List of targets for each simulation
        - List of receiving yards for each simulation
    """
    
    simulated_targets = []
    simulated_yards = []
    
    for _ in range(num_simulations):
        # Simulate targets using binomial distribution
        targets = np.random.binomial(n=team_pass_attempts, p=target_rate)
        
        # Simulate yards for each target using normal distribution
        if targets > 0:
            yards = np.random.normal(yards_mean, yards_std, targets)
            # Round to 1 decimal place and ensure no negative yards
            yards = np.maximum(0, np.round(yards, 1))
            total_yards = sum(yards)
        else:
            total_yards = 0
            
        simulated_targets.append(targets)
        simulated_yards.append(total_yards)
    
    return simulated_targets, simulated_yards

def analyze_simulation_results(
    predicted_attempts,
    targets: List[int],
    yards: List[float]
) -> dict:
    """
    Analyze the results of the simulation.
    
    Args:
        targets: List of simulated target counts
        yards: List of simulated receiving yards
        
    Returns:
        Dictionary containing summary statistics
    """
    return {
        'avg_targets': np.mean(targets),
        'median_targets': np.median(targets),
        'target_share': round(np.mean(targets)/predicted_attempts,3),
        'target_percentiles': np.percentile(targets, [10, 25, 75, 90]),
        'avg_yards': np.mean(yards),
        'median_yards': np.median(yards),
        'yard_percentiles': np.percentile(yards, [10, 25, 75, 90])
    }



# Streamlit app
def main():


    # Create a select box for user to choose between Preview and Review
    choice = st.selectbox("Select an Option", ["Preview", "Review","Team Analysis","Receiving Yards Simulation"])

    if choice == "Preview":
        with st.container():
            st.write("Please enter the following information:")
            season = st.number_input("Season (Integer)")
            team_a = st.text_input("Team A (String)")
            team_b = st.text_input("Team B (String)")

            # Create a button to confirm the selection
            if st.button("Confirm Selection"):
                if season and team_a and team_b:
                    # Call the overall_creator function and display the result
                    
                    overall_result, overall_result2, pass_matchup1, rush_matchup1,pass_matchup2, rush_matchup2 = preview_maker(season, team_a, team_b)
                    # overall_result = overall_creator(season, team_a, team_b)
                    # overall_result2 = overall_creator(season, team_b, team_a)
                    # pass_matchup1 = pass_matchup(season, team_a, team_b)
                    # rush_matchup1 = rush_matchup(season, team_a, team_b)
                    # pass_matchup2 = pass_matchup(season, team_b, team_a)
                    # rush_matchup2 = rush_matchup(season, team_b, team_a)

                    st.write(overall_result)
                    st.write(pass_matchup1)
                    st.write(rush_matchup1)
                    st.write(overall_result2)
                    st.write(pass_matchup2)
                    st.write(rush_matchup2)
                else:
                    st.warning("Please enter all required information.")

    elif choice == "Review":
        # If "Review" is selected, create a text input for Game ID
        game_id = st.text_input("Enter Game ID")
        
        # Create a button to trigger the game review
        if st.button("Submit"):
            if game_id:
                # Call the game_review function and display the result
                review_result = game_review(game_id)
                st.write(review_result)
            else:
                st.warning("Please enter a Game ID.")

    elif choice == "Team Analysis":
        with st.container():
            st.write("Please enter the following information:")
            season = st.number_input("Season")
            team = st.text_input("Team")

        
        # Create a button to trigger the game review
        if st.button("Submit"):
            if team:
                # Call the game_review function and display the result
                analysis_1, analysis_2, analysis_3, analysis_4, analysis_5, analysis_6, analysis_7 = get_team_stats(team, season)
                st.write(analysis_1)
                st.write(analysis_2)
                st.write(analysis_3)
                st.write(analysis_4)
                st.write(analysis_5)
                st.plotly_chart(analysis_6)
                st.plotly_chart(analysis_7)


            else:
                st.warning("Please enter valid inputs.")

    elif choice == "Receiving Yards Simulation":
        with st.container():
            st.write("Please enter the following information:")
            spread = st.number_input("Spread",key="number_input_1")
            total = st.number_input("Total",key="number_input_2")
            chosen_team = st.text_input("Team")
            receiver_name = st.text_input("Receiver Name",key="name_input_1")
            excluded_receiver1 = st.text_input("Excluded Receiver","",key="name_input_2")
            excluded_receiver2 = st.text_input("Excluded Receiver","",key="name_input_3")
            starting_week = st.number_input("Number of Trailing Weeks",key="number_input_3")
            team_attempts = st.number_input("Team Targets (leave '0' for model to predict targets)",key="number_input_4")
            player_target_share = st.number_input("Player Target Share (leave '0' for model to use average target share over trailing games window)",key="number_input_5")

                    
            if st.button("Submit"):
                team_rec_df, rec_df, receiver_string, median_yards, results, team_attempts, team_targets_in_period, szn_targets_per_game, t_share = receiver_simulator(chosen_team, spread, total, excluded_receiver1, excluded_receiver2, receiver_name,starting_week,team_attempts,player_target_share)
                st.write(team_rec_df)
                st.write(rec_df)
                st.write(f"Predicted team targets: {team_attempts}")
                st.write(f"Team targets per game in period {team_targets_in_period}")
                st.write(f"Team targets per game this season {szn_targets_per_game}")
                st.write(f"Target share: {t_share}")
                st.write(receiver_string)
                st.write(median_yards)
                st.write(results)
 
                


if __name__ == "__main__":
    main()




#input_string = st.text_input("Enter a string")

#if st.button('Submit'):
#    result = game_review(input_string)
#    st.write(result)