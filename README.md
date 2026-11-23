## COMP4116-Final-Project-Early-Proposal

# What type of project are you pursuing (Groundwork, Application, Development, other)?

This is an Application project. We apply multiagent systems (MAS) concepts, specifically sequential game theory, repeated play with a shrinking action set, and agent-based simulation of competing strategies, to the champion draft in League of Legends, using the 2026 World Championship (Worlds 2026) as the reference setting.

# What is your topic and what is the research question that you are trying to answer?

Our topic is League of Legends champion drafting, specifically focused on ban strategy in a Fearless Draft series, restricted to the top lane.
Research Question: In a Fearless Draft Bo3 or Bo5 restricted to top lane, does banning the opponent's comfort champions (opponent-targeted banning) give a team a higher series win probability than banning the champions that are strongest in general (generic banning), and how do the real top-lane bans at Worlds 2026 compare with each strategy?

# What resources (tools, datasets, literature) have you identified that will you answer
your research question?

Tools: We will use Python and necessary libraries (pandas, numpy, etc.) to program a simple draft simulation targeted exclusively at Top Lane champions. 

Data: We will use data from the 2026 League of Legends World Championships and regional tournaments that occurred during 2026, including drafts from each match, champion matchup win rates (before Worlds 2026), and player’s individual win rates on each champion. The tournament and draft data can be found by looking through gol.gg, and the champion matchup win rates can be found on u.gg.  

Literature: On Draft Composition Recommendations in League of Legends: Explores how real-time draft recommendation software operates by evaluating statistical impacts of specific champion combinations and synergy factors during the pick-ban phase
 https://ieeexplore.ieee.org/document/10585636 

DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games (2022): A hierarchical recommendation model trained on 280,000 League of Legends matches and 50,000 Dota 2 matches. It combines a player network (capturing individual champion preferences) with a match network (modeling complex hero synergies and opponent counters) to predict picks and match outcomes.
https://arxiv.org/abs/2204.12750 

# What aspects of the topic interest you? How do you plan on exploring that aspect? This
may incorporate additional theoretical frameworks or simulations and experiments.

I (Brian) have spent a considerable amount of time playing League of Legends. I am ranked in the top 0.5% of players in North America, and have even competed professionally representing Carleton University. I am exploring my interest in this game through this topic, specifically choosing to analyze one of the most important parts of the game, the draft, by looking at data from the tournament where all of the best teams in the world play against each other. I will incorporate draft simulations and experiments using this simulation, as well as compare results with real world data to answer my research question. 

I (Iman) am interested in how the topics covered in this course can be applied to video games, particularly games in which players choose characters with different abilities, strengths, and matchups. I have personally spent a lot of time playing games such as Valorant, Overwatch, and Brawl Stars, all of which involve strategic character selection and interactions between opposing choices. I plan to explore this interest by modelling the drafting process as a multi-agent decision problem and using simulations to compare different banning and champion-selection strategies.

We will code a simple draft simulation with 1 ban and 1 pick limited to exclusively Top Lane champions, and then determine win rates based on matchup terms, team proficiencies, and generic champion strengths from the data. Using this simulation, we will implement multiple strategies, including generic and opponent-targeted strategies, to find what has the best win rate. We will test in both Bo3 and Bo5 matchups, using repeated play with a depleting action set to simulate the Fearless Draft aspect. Once we have finished with the simulation, we will compare our data with the real Worlds 2026 data. 

# If applicable, define special domain terminology and describe at a high level appropriate
models you intend to use

League of Legends game: A league of legends game is a game where two teams of five players fight until one team wins. Each player has a role (Top lane, Jungle, Mid lane, Bot lane, Support) and plays one champion.. For our project, we will be focusing exclusively on Top lane, and champions that fit in that role. 
Draft: A league of legends draft starts with each team taking turns banning one champion until each team has banned 3 champions. Then, team one picks one champion, team two responds by picking two champions, then team one picks two champions, and team two picks one more champion. After this, each team gets to alternate bans for two more bans each. After all 5 bans are complete, team two will pick one champion, team one will pick their final two champions, and team two will pick their final champion to end the draft.
Counter pick: Choosing a champion after seeing the opponent’s choice to gain a favourable matchup.
Comfort pick: Choosing a champion that a player is especially proficient with
Fearless Draft: A drafting system used in matches where multiple games are played. In this system, champions that were picked in previous games within a match (usually either Bo 3 or Bo5) cannot be picked again in later games (but they can still be banned).
Simplified Draft Model: For our project, we will be analyzing a simplified draft targeted specifically at top lane, where each team gets 1 ban (1 ban out of 5 for 1 role out of 5 seems reasonable), and then one team picks a Top lane champion first and the other team gets to counter pick a Top lane champion.

Preferred Presentation Slot: No preference, any available slot should be fine.
