import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import warnings
import logging
from collections import defaultdict

# --- Initial Setup & Configuration ---

# Suppress common Streamlit warnings for a cleaner console output.
warnings.filterwarnings('ignore', category=UserWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
logging.getLogger('streamlit').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.scriptrunner.script_run_context').setLevel(logging.ERROR)

# --- App Configuration & Styling ---

# Configure the Streamlit page layout, title, and default sidebar state.
# This is called first as per Streamlit's best practices.
st.set_page_config(layout="wide", page_title="FAAB Buddy", initial_sidebar_state="expanded")

# Define a consistent color palette for all charts to ensure strategies are always
# represented by the same color, improving readability.
PALETTE = {
    "Fixed %":      "#6EC6FF",   # light-blue
    "Fractional %": "#2196F3",   # blue
    "Newsvendor":   "#F48FB1",   # pink
    "Kelly Criterion": "#EF5350" # red
}

# --- Constants & Helper Functions ---

# A static distribution of historical winning bid percentages.
# This is used by the Newsvendor model as a proxy for the distribution of market prices.
HISTORICAL_BIDS_PCT = np.array([
    0.05, 0.08, 0.12, 0.15, 0.03, 0.07, 0.20, 0.25, 0.10, 0.06, 0.09, 0.14, 0.18, 0.22, 0.11, 0.13, 0.04, 0.16, 0.19, 0.21,
    0.02, 0.17, 0.23, 0.26, 0.08, 0.12, 0.15, 0.07, 0.24, 0.28, 0.09, 0.13, 0.18, 0.05, 0.11, 0.14, 0.20, 0.27, 0.06, 0.10
])

def generate_weekly_opportunity(week, total_weeks, market_scale, points_range, rng_instance):
    """
    Generates a single player opportunity for a given week.
    - market_value is drawn from a log-normal distribution to simulate rarity of high-value players.
    - total_value represents the cumulative points the player will add for the rest of the season.
    """
    weeks_remaining = total_weeks - week + 1
    # Use a log-normal distribution for market value; cap it to prevent extreme outliers.
    market_value = min(rng_instance.lognormal(mean=np.log(market_scale), sigma=0.35), market_scale * 4)
    # Weekly points are drawn from a uniform distribution defined by the user.
    weekly_points = rng_instance.uniform(points_range[0], points_range[1])
    return {'market_value': market_value, 'total_value': weekly_points * weeks_remaining}

def simulate_opponent_bids(market_value, opponent_budgets, rng, risk_factors,
                           k_scale=1.8, pass_pr=0.10):
    """
    Simulates bids from all opponents for a single player auction.
    This model incorporates several layers of realism:
    - risk_factors: A persistent, season-long trait for each opponent (some are just more aggressive).
    - ratios: A random weekly fluctuation around their base risk factor.
    - k_scale: A global market "aggressiveness" multiplier.
    - pass_pr: The chance an opponent doesn't bid at all.
    """
    if opponent_budgets.size == 0:
        return np.array([])
    # Each opponent's bid is a function of their personal risk factor, market value, and market aggressiveness.
    ratios = risk_factors * rng.uniform(0.8, 1.2, opponent_budgets.size)
    base   = market_value * ratios * k_scale
    # Opponents cannot bid more than their remaining budget.
    bids   = np.minimum(base, opponent_budgets)
    # A certain percentage of opponents will randomly choose not to bid.
    bids  *= (rng.random(bids.size) > pass_pr)
    return bids.astype(int)

def calculate_newsvendor_bid_pct(total_value, market_value):
    """
    Calculates a bid percentage based on the Newsvendor model logic.
    This model seeks to balance the "cost of being under" (losing the player) with the
    "cost of being over" (paying too much).
    """
    if market_value <= 0: return 0
    # Use "points per dollar" as a proxy for the underage cost (value lost).
    point_value = total_value / market_value
    # The overage cost is simply the dollars spent ($1 per $1).
    # The critical ratio (CR) determines the optimal service level (in our case, bid percentile).
    CR = np.clip(point_value / (point_value + 1), 0.15, 0.9)
    # Convert the critical ratio to a bid percentage using the historical distribution.
    bid_pct = np.percentile(HISTORICAL_BIDS_PCT, CR * 100)
    return bid_pct

# --- CORE SIMULATION FUNCTION ---
def run_simulation(strategies_to_run_tuple, num_sims, start_budget, weeks, num_opp, market_scale, p_range, seed, 
                   _progress_bar=None, k_scale=1.8, beta_a=4, beta_b=4, pass_pr=0.10):
    """
    The main simulation engine. It iterates through each strategy, running a specified
    number of full-season simulations for each.
    """
    # Initialize a random number generator with a master seed for reproducibility.
    sim_rng = np.random.default_rng(seed)
    strategies_to_run = dict(strategies_to_run_tuple)
    
    # --- Data Structures for Storing Results ---
    final_stats = []  # Stores the final season stats (gain, wins, etc.) for every simulation run.
    avg_progression_all = [] # Stores the week-by-week average performance for line charts.
    win_matrix = defaultdict(lambda: np.zeros((weeks, 11))) # Tracks wins for the heatmap.
    opp_matrix = defaultdict(lambda: np.zeros((weeks, 11))) # Tracks opportunities for the heatmap.

    # --- Progress Bar Setup ---
    total_calcs = len(strategies_to_run) * num_sims
    update_step = max(1, total_calcs // 100) # Update progress bar in ~100 increments.
    current_calc = 0

    # --- Main Simulation Loop (by Strategy) ---
    for strategy_name, params in strategies_to_run.items():
        # Arrays to accumulate weekly data across all sims for a given strategy.
        season_gain_total = np.zeros(weeks)
        season_budget_total = np.zeros(weeks)
        
        # For 'Fixed %', the bid amount is calculated once and reused all season.
        fixed_dollar_bid = 0
        if strategy_name == 'Fixed %':
            fixed_dollar_bid = start_budget * params['fraction']
        
        # --- Inner Loop (by Simulation Run) ---
        for j in range(num_sims):
            # Reset season-specific variables at the start of each simulation.
            budget = start_budget
            opponent_budgets = np.full(num_opp, start_budget)
            cumulative_gain, wins, total_spent, cumulative_regret = 0, 0, 0, 0
            
            # Assign persistent "risk factors" to opponents for this entire season.
            risk_factors = sim_rng.beta(beta_a, beta_b, size=num_opp)

            # --- Weekly Loop ---
            for week in range(1, weeks + 1):
                your_bid, win_flag = 0, 0
                opportunity = generate_weekly_opportunity(week, weeks, market_scale, p_range, sim_rng)
                
                # For the heatmap, determine the current budget bucket (decile) and log an opportunity.
                budget_bucket = min(10, int((budget / start_budget) * 10))
                opp_matrix[strategy_name][week - 1, budget_bucket] += 1

                if budget > 0:
                    # Generate one set of opponent bids to be used for this week's auction.
                    opponent_bids = simulate_opponent_bids(
                        opportunity['market_value'], opponent_budgets, sim_rng, risk_factors,
                        k_scale=k_scale, pass_pr=pass_pr
                    )
                    
                    # --- Bidding Logic by Strategy ---
                    if strategy_name == 'Fixed %':
                        your_bid = min(budget, fixed_dollar_bid)
                    elif strategy_name == 'Fractional %':
                        your_bid = budget * params['fraction']
                    elif strategy_name == 'Newsvendor':
                        your_bid = budget * calculate_newsvendor_bid_pct(opportunity['total_value'], opportunity['market_value'])
                    elif strategy_name == 'Kelly Criterion':
                        # Estimate the median opponent bid from the simulated set.
                        est_m = np.median(opponent_bids[opponent_bids > 0]) if np.any(opponent_bids > 0) else 0
                        # Calculate the "edge" based on the market value vs. the estimated median.
                        edge = max(0, opportunity['market_value'] - est_m) / (opportunity['market_value'] or 1e-9)
                        
                        kelly_mult = params.get('kelly_mult', 1.0)
                        kelly_cap = params.get('kelly_cap', 0.5)
                        
                        # Calculate the bid fraction based on edge, aggressiveness, and the cap.
                        kelly_frac = 0.5 * kelly_mult * edge
                        kelly_frac = np.clip(kelly_frac, 0, kelly_cap)
                        your_bid = budget * kelly_frac
                        
                    # Ensure bid does not exceed remaining budget.
                    your_bid = round(min(budget, your_bid))
                    max_opponent_bid = np.max(opponent_bids) if len(opponent_bids) > 0 else 0

                    # --- Auction Resolution ---
                    if your_bid > max_opponent_bid:
                        win_flag = 1
                        wins += 1
                        budget -= your_bid
                        total_spent += your_bid
                        cumulative_gain += (opportunity['total_value'] - your_bid)
                    else:
                        # Calculate regret: the value missed out on by not winning.
                        missed_value = opportunity['total_value'] - max_opponent_bid
                        cumulative_regret += max(0, missed_value)
                        # Deplete the winning opponent's budget.
                        if max_opponent_bid > 0:
                            winner_indices = np.where(opponent_bids == max_opponent_bid)[0]
                            winner_idx = sim_rng.choice(winner_indices)
                            opponent_budgets[winner_idx] = max(0, opponent_budgets[winner_idx] - max_opponent_bid)
                
                # Log heatmap data and accumulate weekly progression data.
                win_matrix[strategy_name][week - 1, budget_bucket] += win_flag
                season_gain_total[week - 1] += cumulative_gain
                season_budget_total[week - 1] += budget

            # Store the final stats for this completed season.
            final_stats.append({'Policy': strategy_name, 'Total Net Gain': cumulative_gain, 'Total Wins': wins, 'Total Spent': total_spent, 'Cumulative Regret': cumulative_regret})

            # Update the progress bar.
            current_calc += 1
            if _progress_bar and (current_calc % update_step == 0):
                _progress_bar.progress(current_calc / total_calcs)
        
        # --- Post-Simulation Aggregation (per Strategy) ---
        # Calculate the average weekly progression across all simulations for this strategy.
        avg_df = pd.DataFrame({
            'Week': np.arange(1, weeks + 1),
            'Avg_Cumulative_Gain': season_gain_total / num_sims,
            'Avg_Budget': season_budget_total / num_sims,
            'Policy': strategy_name
        })
        avg_progression_all.append(avg_df)

    if _progress_bar:
        _progress_bar.progress(1.0)
    
    # Combine the average progression data from all strategies into one DataFrame.
    avg_progression_df = pd.concat(avg_progression_all, ignore_index=True)
    
    # Calculate win probabilities for the heatmap.
    win_prob = defaultdict(lambda: np.zeros((weeks, 11)))
    for policy in opp_matrix:
        win_prob[policy] = np.divide(win_matrix[policy], opp_matrix[policy], out=np.zeros_like(win_matrix[policy]), where=opp_matrix[policy]!=0)

    # Return all the aggregated results.
    return pd.DataFrame(final_stats), avg_progression_df, win_prob

def main():
    """
    Main function to build and run the Streamlit application UI.
    """
    # --- Page Title & Introduction ---
    st.title("FAAB Buddy: A FAAB Bidding Strategy Simulator")
    st.markdown("A decision support tool to compare FAAB bidding strategies using Monte Carlo simulation.")

    # --- Sidebar UI Controls ---
    master_seed = st.sidebar.slider("Master Random Seed", 0, 1000, 42, key="master_seed", help="Set a seed for reproducible results.")
    st.sidebar.caption(f"Seed = {master_seed}")
    st.sidebar.header("Select Simulation Mode")
    app_mode = st.sidebar.radio("Choose your analysis type:", ("Multi-Week Strategy Comparison", "Sensitivity Analysis"))
    st.sidebar.markdown("---")

    # --- Main App Logic: Multi-Week Strategy Comparison Mode ---
    if app_mode == "Multi-Week Strategy Comparison":
        with st.sidebar.expander("Season Parameters", expanded=True):
            initial_budget = st.slider("Starting FAAB Budget ($)", 50, 1000, 100, key="m_ib")
            num_weeks = st.slider("Weeks to Simulate", 1, 17, 16, key="m_nw")
            num_opponents = st.slider("Number of Opponents", 1, 13, 11, key="m_no")
            num_simulations = st.slider("Number of Season Simulations", 10, 1000, 500, key="m_ns")

        with st.sidebar.expander("Weekly Player & Market Model", expanded=True):
            market_scale = st.slider("Avg. Player Market Value ($)", 5, 50, 15, key="m_ms", help="Sets the average 'price' of a player on the wire.")
            points_range = st.slider("Weekly Points Added Range", 1.0, 25.0, (8.0, 18.0), key="m_pr", help="The range of weekly fantasy points a player adds.")
            st.markdown("---")
            k_scale = st.slider("Market aggressiveness k", 1.0, 2.5, 1.8, 0.1, help="Scales all opponent bids. Higher k = tougher market.")
            beta_a = st.slider("Opponent risk α (Beta)", 1, 8, 4, help="Shapes opponent risk profiles. Higher α/β = more average opponents.")
            beta_b = st.slider("Opponent risk β (Beta)", 1, 8, 4, help="Lower α/β = more extreme (passive/aggressive) opponents.")
            pass_pr = st.slider("Opponents skip bid %", 0, 40, 10, help="The chance an opponent will not bid on a player, regardless of value.") / 100.0
            st.caption("Defaults (k=1.8, α=4, β=4) mimic a balanced market. Slide left for loose, right for tight.")

        with st.sidebar.expander("Strategies to Compare", expanded=True):
            available_strategies = ["Fixed %", "Fractional %", "Newsvendor", "Kelly Criterion"]
            selected_strategies = st.multiselect("Select strategies:", available_strategies, default=available_strategies, key="m_ss")

            strategy_params = {}
            if "Fixed %" in selected_strategies:
                strategy_params['Fixed %'] = {'fraction': st.slider("Fixed Bid %", 1, 50, 15, key="m_fbp", help="Same $ every week: pct × starting bankroll.") / 100.0}
            if "Fractional %" in selected_strategies:
                strategy_params['Fractional %'] = {'fraction': st.slider("Fractional Bid %", 1, 50, 25, key="m_frp", help="pct × current bankroll each time you bid.") / 100.0}
            if "Newsvendor" in selected_strategies:
                strategy_params['Newsvendor'] = {}
            
            is_kelly_disabled = "Kelly Criterion" not in selected_strategies
            if "Kelly Criterion" in selected_strategies:
                strategy_params['Kelly Criterion'] = {
                    'kelly_mult': st.slider("Kelly aggressiveness (× half-Kelly)", 0.5, 2.0, 1.0, 0.1, disabled=is_kelly_disabled),
                    'kelly_cap': st.slider("Kelly cap (max % bankroll)", 0.30, 1.00, 0.50, 0.05, disabled=is_kelly_disabled)
                }
                st.caption("Edge is estimated as market value minus estimated median rival bid, divided by market value. ‘Aggressiveness’ multiplies the classical ½-Kelly fraction; ‘Cap’ limits the maximum bankroll % on any single player.", help="Help text for Kelly sliders")
            else:
                st.slider("Kelly aggressiveness (× half-Kelly)", 0.5, 2.0, 1.0, 0.1, disabled=is_kelly_disabled)
                st.slider("Kelly cap (max % bankroll)", 0.30, 1.00, 0.50, 0.05, disabled=is_kelly_disabled)

        if st.sidebar.button("🚀 Run Full Analysis", key="m_run"):
            st.markdown("---")
            progress_bar = st.progress(0, text="Running simulations. Please wait.")
            strategies_to_run = {name: params for name, params in strategy_params.items() if name in selected_strategies}
            strategies_tuple = tuple(sorted(strategies_to_run.items()))

            final_stats_df, avg_progression_df, win_prob = run_simulation(
                strategies_tuple, num_simulations, initial_budget, num_weeks,
                num_opponents, market_scale, points_range, master_seed,
                _progress_bar=progress_bar, k_scale=k_scale, beta_a=beta_a, beta_b=beta_b, pass_pr=pass_pr
            )
            progress_bar.empty()
            del progress_bar

            summary_df = final_stats_df.groupby('Policy').agg(
                Avg_Net_Gain=('Total Net Gain', 'mean'), Median_Net_Gain=('Total Net Gain', 'median'),
                Variance_Gain=('Total Net Gain', 'var'), Avg_Wins=('Total Wins', 'mean'),
                Avg_Total_Spent=('Total Spent', 'mean'), Avg_Regret=('Cumulative Regret', 'mean')
            ).rename(columns={'Avg_Total_Spent': 'Avg $ Spent'}).reset_index()
            
            summary_df['SE_Gain'] = np.sqrt(summary_df['Variance_Gain']) / np.sqrt(num_simulations)
            summary_df['CI95_Gain'] = 1.96 * summary_df['SE_Gain']
            
            for metric in ['Total Wins', 'Total Spent']:
                col_name = f'CI95_{metric.replace(" ", "_")}'
                m_var = final_stats_df.groupby('Policy')[metric].var().fillna(0)
                se = np.sqrt(m_var) / np.sqrt(num_simulations)
                ci_map = (1.96 * se).to_dict()
                summary_df[col_name] = summary_df['Policy'].map(ci_map)

            summary_df.rename(columns={'CI95_Total_Spent': 'CI95_$Spent'}, inplace=True)
            summary_df['Risk-Adjusted Gain'] = (summary_df['Avg_Net_Gain'] / np.sqrt(summary_df['Variance_Gain'])).fillna(0)
            summary_df['Policy'] = pd.Categorical(summary_df['Policy'], categories=available_strategies, ordered=True)
            summary_df_sorted = summary_df.sort_values('Policy')

            st.header("📊 Strategy Performance Summary")
            st.dataframe(summary_df_sorted.style.format({
                'Avg_Net_Gain': '{:.1f}', 'CI95_Gain': '±{:.1f}', 'Median_Net_Gain': '{:.1f}',
                'Variance_Gain': '{:.1f}', 'Avg_Wins': '{:.2f}', 'CI95_Total_Wins': '±{:.2f}',
                'Avg $ Spent': '${:.0f}', 'CI95_$Spent': '±${:.0f}', 'Avg_Regret': '{:.1f}',
                'Risk-Adjusted Gain': '{:.2f}'
            }).background_gradient(cmap='viridis', subset=['Avg_Net_Gain', 'Median_Net_Gain', 'Risk-Adjusted Gain']))

            st.download_button(
                label="📥 Download Full Simulation Results (CSV)",
                data=final_stats_df.to_csv(index=False).encode('utf-8'),
                file_name=f'faab_sim_results_{num_simulations}_runs.csv', mime='text/csv'
            )

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Performance Metrics", "Distributions (Box Plots)", "Distributions (Histograms)",
                "Seasonal Progression", "Win Probability", "Advanced Analysis"
            ])
            
            chart_context_title = f"(N={num_simulations} Sims, {num_opponents} Opponents)"
            final_stats_df['Policy'] = pd.Categorical(final_stats_df['Policy'], categories=available_strategies, ordered=True)
            final_stats_df_sorted = final_stats_df.sort_values('Policy')

            with tab1:
                st.subheader("Average Performance Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(summary_df_sorted, x='Policy', y='Avg_Net_Gain', color='Policy',
                                 error_y='CI95_Gain', title=f"Average Net Gain ± 95% CI {chart_context_title}",
                                 color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.bar(summary_df_sorted, x='Policy', y='Avg_Wins', color='Policy', 
                                 error_y='CI95_Total_Wins', title=f"Average Players Won ± 95% CI {chart_context_title}",
                                 color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    fig = px.bar(summary_df_sorted, x='Policy', y='Avg $ Spent', color='Policy', 
                                 error_y='CI95_$Spent', title=f"Average Total Spent ± 95% CI {chart_context_title}",
                                 color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    fig = px.bar(summary_df_sorted, x='Policy', y='Avg_Regret', color='Policy', title=f"Average Regret {chart_context_title}",
                                 color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)

            with tab2:
                st.subheader("Distribution of Outcomes (Box Plots)")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.box(final_stats_df_sorted, x='Policy', y='Total Net Gain', color='Policy', title='Distribution of Total Net Gain', color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.box(final_stats_df_sorted, x='Policy', y='Total Wins', color='Policy', title='Distribution of Total Wins', color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)

                col3, col4 = st.columns(2)
                with col3:
                    fig = px.box(final_stats_df_sorted, x='Policy', y='Total Spent', color='Policy', title='Distribution of Total Spent', color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    fig = px.box(final_stats_df_sorted, x='Policy', y='Cumulative Regret', color='Policy', title='Distribution of Cumulative Regret', color_discrete_map=PALETTE)
                    st.plotly_chart(fig, use_container_width=True)

            with tab3:
                st.subheader("Distribution of Outcomes (Histograms)")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(final_stats_df_sorted, x='Total Net Gain', color='Policy', 
                                       title='Normalized Histogram of Total Net Gain',
                                       barmode='overlay', histnorm='probability density', nbins=50,
                                       color_discrete_map=PALETTE)
                    fig.update_traces(opacity=0.55)
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.histogram(final_stats_df_sorted, x='Total Wins', color='Policy', 
                                       marginal='box', title='Histogram of Total Wins', nbins=20, 
                                       color_discrete_map=PALETTE, barmode='overlay', histnorm='probability density')
                    fig.update_traces(opacity=0.55)
                    st.plotly_chart(fig, use_container_width=True)
                
                col3, col4 = st.columns(2)
                with col3:
                    fig = px.histogram(final_stats_df_sorted, x='Total Spent', color='Policy', 
                                       marginal='box', title='Histogram of Total Spent', nbins=50, 
                                       color_discrete_map=PALETTE, barmode='overlay', histnorm='probability density')
                    fig.update_traces(opacity=0.55)
                    st.plotly_chart(fig, use_container_width=True)
                with col4:
                    fig = px.histogram(final_stats_df_sorted, x='Cumulative Regret', color='Policy', 
                                       marginal='box', title='Histogram of Cumulative Regret', nbins=50, 
                                       color_discrete_map=PALETTE, barmode='overlay', histnorm='probability density')
                    fig.update_traces(opacity=0.55)
                    st.plotly_chart(fig, use_container_width=True)

            with tab4:
                st.subheader("Seasonal Progression (Average of All Sims)")
                avg_progression_df['Policy'] = pd.Categorical(avg_progression_df['Policy'], categories=available_strategies, ordered=True)
                
                fig1 = px.line(avg_progression_df, x='Week', y='Avg_Cumulative_Gain', color='Policy', markers=True, title='Average Cumulative Net Gain Over a Season', color_discrete_map=PALETTE)
                st.plotly_chart(fig1, use_container_width=True)
                
                fig2 = px.line(avg_progression_df, x='Week', y='Avg_Budget', color='Policy', markers=True, title='Average Budget Depletion Over a Season', color_discrete_map=PALETTE)
                st.plotly_chart(fig2, use_container_width=True)

            with tab5:
                st.subheader("Win Probability Analysis (Heatmaps)")
                st.markdown("Shows the probability of winning an auction, given the week and the percentage of starting budget remaining.")
                
                policies = sorted(list(win_prob.keys()))
                for i in range(0, len(policies), 2):
                    col1, col2 = st.columns(2)
                    if i < len(policies):
                        with col1:
                            pol1 = policies[i]
                            fig_heat1 = px.imshow(win_prob[pol1], aspect='auto', origin='lower',
                                                 title=f'Win Probability – {pol1}',
                                                 labels=dict(x='Budget Remaining (%)', y='Week', color="Win Prob."),
                                                 x=[f"{b*10}%" for b in range(11)], y=[f"{w+1}" for w in range(num_weeks)],
                                                 zmin=0, zmax=win_prob[pol1].max() if win_prob[pol1].size > 0 else 1)
                            fig_heat1.update_xaxes(type="category", categoryorder="array", categoryarray=[f"{b*10}%" for b in range(11)])
                            st.plotly_chart(fig_heat1, use_container_width=True)

                    if i + 1 < len(policies):
                        with col2:
                            pol2 = policies[i+1]
                            fig_heat2 = px.imshow(win_prob[pol2], aspect='auto', origin='lower',
                                                 title=f'Win Probability – {pol2}',
                                                 labels=dict(x='Budget Remaining (%)', y='Week', color="Win Prob."),
                                                 x=[f"{b*10}%" for b in range(11)], y=[f"{w+1}" for w in range(num_weeks)],
                                                 zmin=0, zmax=win_prob[pol2].max() if win_prob[pol2].size > 0 else 1)
                            fig_heat2.update_xaxes(type="category", categoryorder="array", categoryarray=[f"{b*10}%" for b in range(11)])
                            st.plotly_chart(fig_heat2, use_container_width=True)

            with tab6:
                st.subheader("Advanced Analysis")
                st.markdown("Deeper dives into strategy performance characteristics.")

                st.subheader("Histogram: Total Net Gain (Overlay)")
                fig_hist = px.histogram(final_stats_df_sorted, x="Total Net Gain", color="Policy", 
                                        barmode="overlay", nbins=40, marginal="rug", 
                                        color_discrete_map=PALETTE, histnorm='probability density')
                fig_hist.update_traces(opacity=0.55)
                st.plotly_chart(fig_hist, use_container_width=True, key="hist_net_gain_adv")

                st.subheader("Spending vs. Net Gain")
                fig_scatter = px.scatter(final_stats_df_sorted, x="Total Spent", y="Total Net Gain", color="Policy", trendline="ols", color_discrete_map=PALETTE)
                st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_spend_gain")

                st.subheader("Risk vs. Return")
                risk_df = summary_df_sorted.assign(Risk=np.sqrt(summary_df_sorted['Variance_Gain']))
                fig_rr = px.scatter(risk_df, x='Risk', y='Avg_Net_Gain', text='Policy', color='Policy',
                                    color_discrete_map=PALETTE, title='Risk vs. Return')
                fig_rr.update_traces(textposition='bottom center', marker=dict(size=12))
                fig_rr.update_layout(showlegend=False)
                fig_rr.update_xaxes(title_text="Risk (σ)")
                fig_rr.update_yaxes(title_text="Average Net Gain")
                st.plotly_chart(fig_rr, use_container_width=True, key='risk_return')

    # --- Main App Logic: Sensitivity Analysis Mode ---
    elif app_mode == "Sensitivity Analysis":
        st.header("🔬 Sensitivity Analysis")
        st.markdown("Analyze how strategy performance changes when a key parameter is varied.")
        
        # --- SENSITIVITY ANALYSIS UI ---
        param_to_vary = st.sidebar.selectbox(
            "Parameter to Analyze:", 
            ["Number of Opponents", "Avg. Player Market Value", "Market Aggressiveness (k)", "Opponent Passivity (% skip bid)"], 
            key="sa_param_vary"
        )

        with st.sidebar.expander("Static Parameters", expanded=True):
            initial_budget_sa = st.slider("Starting Budget ($)", 50, 1000, 100, key="sa_ib")
            num_weeks_sa = 16
            num_simulations_sa = st.slider("Sims per Step", 10, 500, 100, key="sa_ns")
            
            # Conditionally display static parameters based on the chosen variable parameter
            if param_to_vary != "Number of Opponents":
                num_opponents_sa = st.slider("Number of Opponents (Static)", 1, 13, 11, key="sa_no_static")
            if param_to_vary != "Avg. Player Market Value":
                market_scale_sa = st.slider("Avg. Player Market Value ($) (Static)", 5, 50, 15, key="sa_ms_static")
            if param_to_vary != "Market Aggressiveness (k)":
                k_scale_sa = st.slider("Market Aggressiveness k (Static)", 1.0, 2.5, 1.8, 0.1, key="sa_k_static")
            if param_to_vary != "Opponent Passivity (% skip bid)":
                pass_pr_sa = st.slider("Opponents skip bid % (Static)", 0, 40, 10, key="sa_pass_static") / 100.0
            
            beta_a_sa = st.slider("Opponent risk α (Beta) (Static)", 1, 8, 4, key="sa_beta_a")
            beta_b_sa = st.slider("Opponent risk β (Beta) (Static)", 1, 8, 4, key="sa_beta_b")

        with st.sidebar.expander("Variable Parameter Range", expanded=True):
            if param_to_vary == "Number of Opponents":
                param_range = st.slider("Range of Opponents", 1, 13, (8, 12), key="sa_no_range")
                param_values = list(range(param_range[0], param_range[1] + 1))
            elif param_to_vary == "Avg. Player Market Value":
                param_range = st.slider("Range of Avg. Market Value", 5, 50, (10, 25), key="sa_ms_range")
                param_values = list(range(param_range[0], param_range[1] + 1))
            elif param_to_vary == "Market Aggressiveness (k)":
                param_range = st.slider("Range of Market Aggressiveness", 1.0, 2.5, (1.2, 2.2), 0.1, key="sa_k_range")
                param_values = np.arange(param_range[0], param_range[1] + 0.1, 0.1)
            elif param_to_vary == "Opponent Passivity (% skip bid)":
                param_range = st.slider("Range of Opponent skip bid %", 0, 40, (5, 25), 5, key="sa_pass_range")
                param_values = [v / 100.0 for v in range(param_range[0], param_range[1] + 5, 5)]

        with st.sidebar.expander("Strategies to Compare", expanded=True):
            sa_available_strategies = ["Fixed %", "Fractional %", "Newsvendor", "Kelly Criterion"]
            sa_selected_strategies = st.multiselect("Select strategies:", sa_available_strategies, default=sa_available_strategies, key="sa_ss")
            
            sa_strategy_params = {}
            if "Fixed %" in sa_selected_strategies: 
                sa_strategy_params['Fixed %'] = {'fraction': st.slider("Fixed Bid %", 1, 50, 15, key="sa_fbp") / 100.0}
            if "Fractional %" in sa_selected_strategies: 
                sa_strategy_params['Fractional %'] = {'fraction': st.slider("Fractional Bid %", 1, 50, 25, key="sa_frp") / 100.0}
            if "Newsvendor" in sa_selected_strategies: 
                sa_strategy_params['Newsvendor'] = {}
            if "Kelly Criterion" in sa_selected_strategies: 
                sa_strategy_params['Kelly Criterion'] = {}

        if st.button("📊 Run Sensitivity Analysis"):
            sa_results = []
            st.markdown("---")
            progress_bar = st.progress(0, text="Running Sensitivity Analysis...")
            
            # --- SENSITIVITY ANALYSIS LOOP ---
            for i, val in enumerate(param_values):
                # Set static and variable parameters for this iteration
                sim_params = {
                    'num_opp': num_opponents_sa if param_to_vary != "Number of Opponents" else val,
                    'market_scale': market_scale_sa if param_to_vary != "Avg. Player Market Value" else val,
                    'k_scale': k_scale_sa if param_to_vary != "Market Aggressiveness (k)" else val,
                    'pass_pr': pass_pr_sa if param_to_vary != "Opponent Passivity (% skip bid)" else val,
                    'beta_a': beta_a_sa,
                    'beta_b': beta_b_sa
                }

                strategies_to_run = {name: params for name, params in sa_strategy_params.items() if name in sa_selected_strategies}
                strategies_tuple = tuple(sorted(strategies_to_run.items()))
                
                final_stats_df, _, _ = run_simulation(
                    strategies_tuple, num_simulations_sa, initial_budget_sa,
                    num_weeks_sa, sim_params['num_opp'], sim_params['market_scale'], 
                    (8.0, 18.0), master_seed, _progress_bar=None, 
                    k_scale=sim_params['k_scale'], beta_a=sim_params['beta_a'], 
                    beta_b=sim_params['beta_b'], pass_pr=sim_params['pass_pr']
                )

                # Aggregate all KPIs for this iteration
                agg_df = final_stats_df.groupby('Policy').agg(
                    Avg_Net_Gain=('Total Net Gain', 'mean'),
                    Var_Net_Gain=('Total Net Gain', 'var'),
                    Avg_Wins=('Total Wins', 'mean'),
                    Var_Wins=('Total Wins', 'var')
                ).reset_index().fillna(0)

                agg_df['CI95_Gain'] = 1.96 * np.sqrt(agg_df['Var_Net_Gain']) / np.sqrt(num_simulations_sa)
                agg_df['CI95_Wins'] = 1.96 * np.sqrt(agg_df['Var_Wins']) / np.sqrt(num_simulations_sa)
                agg_df['Risk'] = np.sqrt(agg_df['Var_Net_Gain'])
                agg_df[param_to_vary] = val
                sa_results.append(agg_df)

                progress_bar.progress((i + 1) / len(param_values), text=f"Analyzing {param_to_vary} = {val}")

            sa_results_df = pd.concat(sa_results, ignore_index=True)
            progress_bar.empty()
            
            # --- SENSITIVITY ANALYSIS PLOTTING ---
            sa_results_df['Policy'] = pd.Categorical(sa_results_df['Policy'], categories=sa_available_strategies, ordered=True)
            sa_results_df_sorted = sa_results_df.sort_values('Policy')

            st.subheader(f"Sensitivity of Avg. Net Gain to {param_to_vary}")
            fig1 = px.line(sa_results_df_sorted, x=param_to_vary, y='Avg_Net_Gain', color='Policy', 
                           markers=True, error_y='CI95_Gain', title=f"Primary KPI: Avg. Net Gain vs. {param_to_vary}",
                           color_discrete_map=PALETTE)
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader(f"Sensitivity of Secondary KPIs to {param_to_vary}")
            col1, col2 = st.columns(2)
            with col1:
                fig2 = px.line(sa_results_df_sorted, x=param_to_vary, y='Avg_Wins', color='Policy', 
                               markers=True, error_y='CI95_Wins', title=f"Secondary KPI: Avg. Wins vs. {param_to_vary}",
                               color_discrete_map=PALETTE)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = px.line(sa_results_df_sorted, x=param_to_vary, y='Risk', color='Policy', 
                               markers=True, title=f"Secondary KPI: Risk (σ) vs. {param_to_vary}",
                               color_discrete_map=PALETTE)
                st.plotly_chart(fig3, use_container_width=True)

# --- Script Execution ---
# This block ensures that the main() function is called only when the script is run directly.
if __name__ == "__main__":
    main()
