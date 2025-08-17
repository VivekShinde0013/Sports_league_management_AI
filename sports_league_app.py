
import os, sys
from datetime import datetime
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEAMS = os.path.join(DATA_DIR, "teams.csv")
PLAYERS = os.path.join(DATA_DIR, "players.csv")
MATCHES = os.path.join(DATA_DIR, "matches.csv")
MODEL = os.path.join(DATA_DIR, "match_model.npz")

POINTS_WIN, POINTS_DRAW = 3, 1
ELO_START, ELO_K = 1500, 20

def _load(path, cols=None):
    if not os.path.exists(path):
        pd.DataFrame(columns=cols or []).to_csv(path, index=False)
    return pd.read_csv(path)

def _save(df, path): df.to_csv(path, index=False)
def _next_id(df, col): return 1 if df.empty else int(df[col].max())+1
def _tm(teams): return {int(r.team_id): str(r.team_name) for _,r in teams.iterrows()}

def header(t): 
    print("="*72); print(t); print("="*72)

# ---------- Add/List/Delete ----------
def list_teams():
    teams = _load(TEAMS, ["team_id","team_name"])
    if teams.empty: print("No teams yet."); return
    print(tabulate(teams.rename(columns={"team_id":"ID","team_name":"Team"}), headers="keys", tablefmt="github", showindex=False))

def add_team():
    teams = _load(TEAMS, ["team_id","team_name"])
    name = input("Enter new team name: ").strip()
    if not name: print("Name required."); return
    if (teams["team_name"]==name).any():
        print("Team already exists."); return
    tid = _next_id(teams, "team_id")
    teams = pd.concat([teams, pd.DataFrame([{"team_id":tid,"team_name":name}])], ignore_index=True)
    _save(teams, TEAMS)
    print(f"Added team '{name}' with ID {tid}.")

def delete_team():
    teams = _load(TEAMS, ["team_id","team_name"])
    if teams.empty: print("No teams to delete."); return
    list_teams()
    try:
        tid = int(input("Enter Team ID to delete: "))
    except Exception:
        print("Invalid ID."); return
    if tid not in teams["team_id"].values:
        print("Team not found."); return

    players = _load(PLAYERS, ["player_id","team_id","player_name","position","rating"])
    matches = _load(MATCHES, ["match_id","date","home_team_id","away_team_id","played","home_goals","away_goals"])
    pcount = int((players["team_id"]==tid).sum())
    mcount = int(((matches["home_team_id"]==tid) | (matches["away_team_id"]==tid)).sum())

    confirm = input(f"Delete team ID {tid} and cascade remove {pcount} players and {mcount} matches? (yes/no): ").strip().lower()
    if confirm not in ("y","yes"): 
        print("Cancelled."); return

    teams = teams[teams["team_id"]!=tid]; _save(teams, TEAMS)
    players = players[players["team_id"]!=tid]; _save(players, PLAYERS)
    matches = matches[~((matches["home_team_id"]==tid)|(matches["away_team_id"]==tid))]; _save(matches, MATCHES)
    if os.path.exists(MODEL):
        try: os.remove(MODEL)
        except Exception: pass
    print(f"Team {tid} and related players/matches removed.")

def list_players():
    players = _load(PLAYERS, ["player_id","team_id","player_name","position","rating"])
    teams = _load(TEAMS, ["team_id","team_name"])
    if players.empty: print("No players yet."); return
    tm = _tm(teams)
    df = players.copy()
    df["team"] = df["team_id"].map(lambda t: tm.get(int(t), f"T{t}"))
    df = df[["player_id","player_name","team","position","rating"]]
    df.columns = ["ID","Player","Team","Pos","Rating"]
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

def add_player():
    teams = _load(TEAMS, ["team_id","team_name"])
    if teams.empty: print("Add a team first."); return
    list_teams()
    try:
        team_id = int(input("Enter team ID: "))
    except Exception:
        print("Invalid team ID."); return
    if team_id not in teams["team_id"].values: print("Team not found."); return
    name = input("Player name: ").strip()
    pos = input("Position (GK/DF/MF/FW): ").strip().upper() or "MF"
    try:
        rating = int(input("Rating (40-99): "))
    except Exception:
        rating = 70
    players = _load(PLAYERS, ["player_id","team_id","player_name","position","rating"])
    pid = _next_id(players, "player_id")
    players = pd.concat([players, pd.DataFrame([{
        "player_id":pid,"team_id":team_id,"player_name":name,"position":pos,"rating":rating
    }])], ignore_index=True)
    _save(players, PLAYERS)
    if os.path.exists(MODEL):
        try: os.remove(MODEL)
        except Exception: pass
    print(f"Added player '{name}' (ID {pid}) to team {team_id}.")

def delete_player():
    players = _load(PLAYERS, ["player_id","team_id","player_name","position","rating"])
    if players.empty: print("No players to delete."); return
    list_players()
    try:
        pid = int(input("Enter Player ID to delete: "))
    except Exception:
        print("Invalid ID."); return
    if pid not in players["player_id"].values:
        print("Player not found."); return
    confirm = input(f"Delete player ID {pid}? (yes/no): ").strip().lower()
    if confirm not in ("y","yes"):
        print("Cancelled."); return
    players = players[players["player_id"]!=pid]
    _save(players, PLAYERS)
    if os.path.exists(MODEL):
        try: os.remove(MODEL)
        except Exception: pass
    print(f"Player {pid} removed.")

# ---------- Matches & Results ----------
def list_matches(upcoming_only=False):
    teams = _load(TEAMS, ["team_id","team_name"])
    matches = _load(MATCHES, ["match_id","date","home_team_id","away_team_id","played","home_goals","away_goals"])
    if matches.empty:
        print("No matches."); return
    tm = _tm(teams)
    df = matches.copy()
    if upcoming_only:
        today = datetime.today().date().isoformat()
        df = df[(df["played"]==0) & (df["date"]>=today)]
    df["Home"] = df["home_team_id"].map(lambda x: tm.get(int(x), f"T{x}"))
    df["Away"] = df["away_team_id"].map(lambda x: tm.get(int(x), f"T{x}"))
    df = df[["match_id","date","Home","Away","played","home_goals","away_goals"]].sort_values(["date","match_id"])
    df.columns = ["ID","Date","Home","Away","Played","H","A"]
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))

def record_result():
    matches = _load(MATCHES, ["match_id","date","home_team_id","away_team_id","played","home_goals","away_goals"])
    list_matches(upcoming_only=False)
    try:
        mid = int(input("Enter Match ID to record: "))
    except Exception:
        print("Invalid Match ID."); return
    row = matches[matches["match_id"]==mid]
    if row.empty: print("Match not found."); return
    if int(row.iloc[0]["played"])==1: print("Already recorded."); return
    try:
        hg = int(input("Home goals: ")); ag = int(input("Away goals: "))
    except Exception:
        print("Invalid scores."); return
    matches.loc[matches["match_id"]==mid, ["home_goals","away_goals","played"]] = [hg, ag, 1]
    _save(matches, MATCHES)
    print("Result saved.")

# ---------- Standings & Elo ----------
POINTS_WIN, POINTS_DRAW = 3, 1
ELO_START, ELO_K = 1500, 20

def compute_elo(teams, matches):
    elo = {int(t.team_id): ELO_START for _,t in teams.iterrows()}
    played = matches[matches["played"]==1].sort_values("date")
    for _, m in played.iterrows():
        h,a = int(m.home_team_id), int(m.away_team_id)
        hg,ag = int(m.home_goals), int(m.away_goals)
        res = 1 if hg>ag else (0 if hg<ag else 0.5)
        Eh = 1/(1+10**((elo[a]-elo[h])/400))
        elo[h] = elo[h] + ELO_K*(res - Eh)
        elo[a] = elo[a] + ELO_K*((1-res) - (1-Eh))
    return elo

def standings():
    teams = _load(TEAMS, ["team_id","team_name"])
    matches = _load(MATCHES, ["match_id","date","home_team_id","away_team_id","played","home_goals","away_goals"])
    if teams.empty: 
        print("No teams."); return
    rows=[]
    for _,t in teams.iterrows():
        tid=int(t.team_id); W=D=L=GF=GA=0
        played = matches[(matches["played"]==1) & ((matches["home_team_id"]==tid)|(matches["away_team_id"]==tid))]
        for _,m in played.iterrows():
            hg,ag=int(m.home_goals),int(m.away_goals)
            if int(m.home_team_id)==tid:
                GF+=hg; GA+=ag
                W+=hg>ag; D+=hg==ag; L+=hg<ag
            else:
                GF+=ag; GA+=hg
                W+=ag>hg; D+=ag==hg; L+=ag<hg
        P = int(W)*POINTS_WIN + int(D)*POINTS_DRAW
        rows.append({"Team":t.team_name,"P":int(W)+int(D)+int(L),"W":int(W),"D":int(D),"L":int(L),"GF":GF,"GA":GA,"GD":GF-GA,"Pts":P})
    df = pd.DataFrame(rows).sort_values(["Pts","GD","GF"],ascending=[False,False,False]).reset_index(drop=True)
    print(tabulate(df, headers="keys", tablefmt="github", showindex=True))

    # Elo table
    elo = compute_elo(teams, matches)
    erows = [{"Team": teams[teams.team_id==tid].iloc[0].team_name, "Elo": round(r,1)} for tid,r in sorted(elo.items(), key=lambda x:x[1], reverse=True)]
    header("Power Ratings (Elo)")
    print(tabulate(erows, headers="keys", tablefmt="github", showindex=True))

# ---------- ML ----------
def _avg_ratings():
    players = _load(PLAYERS, ["player_id","team_id","player_name","position","rating"])
    return players.groupby("team_id")["rating"].mean().to_dict() if not players.empty else {}

def _build_training():
    teams = _load(TEAMS, ["team_id","team_name"])
    matches = _load(MATCHES, ["match_id","date","home_team_id","away_team_id","played","home_goals","away_goals"])
    played = matches[matches["played"]==1]
    if played.empty: return np.zeros((0,3)), np.zeros((0,))
    elo = compute_elo(teams, matches)
    avg = _avg_ratings()
    X=[]; y=[]
    for _,m in played.iterrows():
        h,a=int(m.home_team_id), int(m.away_team_id)
        elo_diff = elo.get(h, ELO_START) - elo.get(a, ELO_START)
        rat_diff = avg.get(h,70) - avg.get(a,70)
        goal_diff = int(m.home_goals) - int(m.away_goals)
        X.append([elo_diff, rat_diff, goal_diff])
        y.append(1 if int(m.home_goals)>int(m.away_goals) else 0)
    return np.array(X), np.array(y)

def train_model():
    X,y = _build_training()
    if len(y) < 8 or len(np.unique(y))<2:
        print("Not enough varied results yet — predictions will use Elo heuristic.")
        return None
    model = LogisticRegression(max_iter=1000)
    model.fit(X,y)
    np.savez(MODEL, coef_=model.coef_, intercept_=model.intercept_)
    print(f"Model trained on {len(y)} matches.")
    return model

def _load_model():
    if not os.path.exists(MODEL): return None
    d = np.load(MODEL, allow_pickle=True)
    class M:
        coef_ = d["coef_"]; intercept_ = d["intercept_"]
        def predict_proba(self, X):
            z = X @ self.coef_.T + self.intercept_
            p = 1/(1+np.exp(-z))
            return np.hstack([1-p, p])
    return M()

def predict_upcoming():
    teams = _load(TEAMS, ["team_id","team_name"])
    matches = _load(MATCHES, ["match_id","date","home_team_id","away_team_id","played","home_goals","away_goals"])
    if matches.empty:
        print("No matches."); return
    tm = _tm(teams)
    model = _load_model()
    elo = compute_elo(teams, matches)
    avg = _avg_ratings()

    upcoming = matches[matches["played"]==0].sort_values(["date","match_id"])
    rows=[]
    for _,m in upcoming.iterrows():
        h,a=int(m.home_team_id), int(m.away_team_id)
        X = np.array([[
            elo.get(h,ELO_START) - elo.get(a,ELO_START),
            avg.get(h,70) - avg.get(a,70),
            0
        ]])
        if model is not None:
            p_home = float(model.predict_proba(X)[0,1])
            method = "ML"
        else:
            Rh,Ra = elo.get(h,ELO_START), elo.get(a,ELO_START)
            p_home = 1/(1+10**((Ra-Rh)/400))
            method = "Elo"
        rows.append({"ID":int(m.match_id),"Date":m.date,"Home":tm.get(h,f'T{h}'),"Away":tm.get(a,f'T{a}'),
                     "P(HomeWin)":round(p_home,3),"Pick": tm.get(h,f'T{h}') if p_home>=0.5 else tm.get(a,f'T{a}'),
                     "Method":method})
    if not rows: print("No upcoming matches."); return
    print(tabulate(rows, headers="keys", tablefmt="github", showindex=False))

# ---------- Menu ----------
def menu():
    while True:
        header("Sports League Management System — AI/ML (v4)")
        print("1) List Teams")
        print("2) Add Team")
        print("3) Delete Team")
        print("4) List Players")
        print("5) Add Player")
        print("6) Delete Player")
        print("7) List Matches")
        print("8) Record Result")
        print("9) Train/Update ML Model")
        print("10) Predict Upcoming Matches")
        print("11) Standings & Elo Ratings")
        print("0) Exit")
        c = input("Choose: ").strip()
        if c=='1': list_teams()
        elif c=='2': add_team()
        elif c=='3': delete_team()
        elif c=='4': list_players()
        elif c=='5': add_player()
        elif c=='6': delete_player()
        elif c=='7': list_matches(upcoming_only=False)
        elif c=='8': record_result()
        elif c=='9': train_model()
        elif c=='10': predict_upcoming()
        elif c=='11': standings()
        elif c=='0': print("Goodbye!"); break
        else: print("Invalid.")
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    menu()
