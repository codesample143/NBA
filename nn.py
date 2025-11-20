import torch
import requests
import re
import pandas as pd
from nba_api.stats.endpoints import playerdashboardbyyearoveryear
from nba_api.stats.endpoints import teamdashlineups
from nba_api.stats.endpoints import commonteamroster
from nba_api.stats.static import teams
from nba_api.stats.endpoints.boxscoreadvancedv3 import BoxScoreAdvancedV3
from nba_api.stats.static import players

"""
    Goal of this project is to return an output value for the most likely team to win a championship. 
    Network will take in a multidimensional torch tensor grouping the best 9 players and their points, rebounds, assists, steals, blocks, ts%, ft%, LEBRON, RAPTOR
    There will be 30 of these such tensors, resulting output will be tested against historical victories, meaning a binary outcome.
"""

"""
    Get a list of all teams in the NBA, then iterate to get the top 8 players. This will be more meaningful come playoff time, and modern teams use more of their bench. 
"""
nba_team_list = teams.get_teams()
team_id_name = []
nba_intermediate_data_for_names = []

nba_team_roster = []

"""API structure: Indexed in a singular list that is fully returned """
for i in range(len((nba_team_list))):
    team_id_name.append(nba_team_list[i]['id'])

"""for id in team_id_name:
    print(id)
"""

"""every team, need to organize into player stats for each team"""
nba_intermediate_data_for_names.append(commonteamroster.CommonTeamRoster(team_id='1610612757')) 


"""get names of players, must repeat for all 30 teams"""
nba_team_roster.append([nba_intermediate_data_for_names[0].get_data_frames()[0]['PLAYER']])

"""get stats and advanced stats of every nba player and stick that in a list of list of lists"""
roster = []
temp = "".join(str(nba_team_roster).split(maxsplit=1))
names = re.findall(r"[A-Z][A-Za-z' .]+(?:III|II|IV)?", temp)
names = [n.strip() for n in names]

"""fix this later to make sure loop will run"""
for i in range(30):
    team_roster = []
    for n in range(min(len(names), 15)):
        value = players.find_players_by_full_name(names[n])[0]['id']
        team_roster.append(value)
    roster.append(team_roster)
print(roster)