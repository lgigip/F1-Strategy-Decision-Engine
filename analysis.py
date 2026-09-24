from pathlib import Path
import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
PAPAYA = "#FF8000"
COMPOUND_COLOURS = {"SOFT": "#FF3333", "MEDIUM": "#FFF200", "HARD": "#EBEBEB"}

# setting up FastF1 cache
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

figures_dir = Path("figures")
results_dir = Path("results")
figures_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

# loading race session
session = fastf1.get_session(2024, "Bahrain Grand Prix", "R")
session.load(telemetry=False)

# accessing lap data
laps = session.laps

print("\nSuccessfully loaded race:")
print(session.event["EventName"])
print("\nDrivers in dataset:")
print(laps["Driver"].dropna().unique())

# inspecting Norris' laps
norris_laps = laps[laps["Driver"] == "NOR"]
viewed_columns = ["LapNumber", "LapTime", "Compound", "TyreLife",
                  "FreshTyre", "Stint", "TrackStatus", "IsAccurate"]
print("\nFirst 20 laps (Norris):")
print(norris_laps[viewed_columns].head(20).to_string(index=False))



# cleaning lap data

norris_laps = norris_laps.copy()
# converting timedelta into seconds
norris_laps["LapTimeSeconds"] = (norris_laps["LapTime"].dt.total_seconds())
# only retaining laps suitable for normal pace analysis
clean_laps = norris_laps[(norris_laps["IsAccurate"] == True)
                         & (norris_laps["TrackStatus"] == "1")
                         & (norris_laps["PitInTime"].isna())
                         & (norris_laps["PitOutTime"].isna())
                         & (norris_laps["LapTimeSeconds"].notna())].copy()

print("\nClean laps (Norris):")
clean_columns = ["LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "FreshTyre", "Stint"]
print(clean_laps[clean_columns].to_string(index=False))


# summarising each stint
stint_summary = (clean_laps.groupby(["Stint", "Compound", "FreshTyre"])
                 .agg(FirstRaceLap=("LapNumber", "min"), LastRaceLap=("LapNumber", "max"),
                      FirstTyreAge=("TyreLife", "min"), LastTyreAge=("TyreLife", "max"),
                      Cleanlaps=("LapNumber", "count")).reset_index())
print("\nStint Summary:")
print(stint_summary.to_string(index=False))


# plotting clean lap time vs tyre age
fig, ax = plt.subplots(figsize=(9, 5))
for (stint, compound), stint_data in clean_laps.groupby(["Stint", "Compound"]):
    ax.scatter(stint_data["TyreLife"], stint_data["LapTimeSeconds"], color=COMPOUND_COLOURS.get(compound, "grey"), 
               edgecolor="black", linewidth=0.6, s=55, label=f"stint{int(stint)} - {compound}")
ax.set_title("Lando Norris - Bahrain 2024\n Clean Race Lap Time vs Tyre Age")
ax.set_xlabel("Tyre Age (laps)")
ax.set_ylabel("Lap Time (s)")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
plt.show()

# initial HARD tyre degradation model
hard_laps = clean_laps[clean_laps["Compound"] == "HARD"].copy()
X = np.column_stack([np.ones(len(hard_laps)), hard_laps["TyreLife"], hard_laps["LapNumber"]])
y = hard_laps["LapTimeSeconds"].to_numpy()

coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
intercept = coefficients[0]
tyre_age_coefficient = coefficients[1]
race_lap_coefficient = coefficients[2]
print("\n Initial HARD tyre model:")
print(f"Tyre age effect:" f"{tyre_age_coefficient:.3f} s/lap")
print(f"Race progression effect:" f"{race_lap_coefficient:.3f} s/race lap")


# residual = actual lap time - modelled lap time
# inspecting residuals:
hard_laps["PredictedLapTime"] = (intercept + tyre_age_coefficient*hard_laps["TyreLife"] +
                                 race_lap_coefficient*hard_laps["LapNumber"])
hard_laps["Residual"] = (hard_laps["LapTimeSeconds"] - hard_laps["PredictedLapTime"])
hard_laps["AbsoluteResidual"] = (hard_laps["Residual"].abs())
largest_residuals = (hard_laps[["LapNumber", "TyreLife", "LapTimeSeconds", "PredictedLapTime", "Residual"]].sort_values(
    "Residual", key=abs, ascending=False).head(8))
print("\nLargest model residuals:")
print(largest_residuals.to_string(index=False))

# using MAD (Median Absolute Deviation) to flag unusual residuals
median_residual = hard_laps["Residual"].median()
mad = (hard_laps["Residual"] - median_residual).abs().median()
robust_sigma = 1.4826*mad    # converting to robust estimate, comparable to std. deviation
outlier_threshold = 3*robust_sigma
hard_laps["Outlier"] = ((hard_laps["Residual"] - median_residual).abs() > outlier_threshold)
print("\nResidual filtering:")
print(f"Median residual: {median_residual:.3f} s")
print(f"MAD: {mad:.3f} s")
print(f"Outlier threshold: {outlier_threshold:.3f} s")
print("\nFlagged laps:")
print(hard_laps.loc[hard_laps["Outlier"], ["LapNumber", "TyreLife", "LapTimeSeconds", "PredictedLapTime", "Residual"]]
      .to_string(index=False))

# model without flagged laps
model_laps = hard_laps[hard_laps["Outlier"] == False].copy()
X_clean = np.column_stack([np.ones(len(model_laps)), model_laps["TyreLife"], model_laps["LapNumber"]])
y_clean = model_laps["LapTimeSeconds"].to_numpy()
clean_coefficients, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
clean_intercept = clean_coefficients[0]
clean_tyre_age_coefficient = clean_coefficients[1]
clean_race_lap_coefficient = clean_coefficients[2]
print("\nRobust HARD tyre model:")
print(f"Tyre age effect: {clean_tyre_age_coefficient:.3f} s/lap")
print(f"Race progression effect: {clean_race_lap_coefficient:.3f} s/race lap")
print(f"Laps retained: {len(model_laps)}/{len(hard_laps)}")

# preparing clean laps (for all drivers)
all_laps = laps.copy()
all_laps["LapTimeSeconds"] = (all_laps["LapTime"].dt.total_seconds())
all_clean_laps = all_laps[(all_laps["IsAccurate"] == True) & (all_laps["TrackStatus"] == "1") & (all_laps["PitInTime"].isna())
                          & (all_laps["PitOutTime"].isna()) & (all_laps["LapTimeSeconds"].notna())].copy()

# summarising HARD stints
hard_stint_summary = (all_clean_laps[(all_clean_laps["Compound"] == "HARD") & (all_clean_laps["FreshTyre"] == True)]
                      .groupby(["Driver", "Stint", "FreshTyre"])
                      .agg(FirstRaceLap=("LapNumber", "min"), LastRaceLap=("LapNumber", "max"), FirstTyreAge=("TyreLife", "min"), 
                           LastTyreAge=("TyreLife", "max"), CleanLaps=("LapNumber", "count")).reset_index())
hard_stints_usable = hard_stint_summary[hard_stint_summary["CleanLaps"] >= 8].copy()
print("\nUsable HARD stints:")
print(hard_stints_usable.to_string(index=False))

# identifying drivers with 2+ usable HARD stints
elig_drivers = (hard_stints_usable.groupby("Driver").agg(HardStints=("Stint", "nunique"), TotalCleanHardLaps=("CleanLaps", "sum")).reset_index())
elig_drivers = (elig_drivers[elig_drivers["HardStints"] >= 2].sort_values("TotalCleanHardLaps", ascending=False))
print("\nDrivers suitable for cross-driver validation:")
print(elig_drivers.to_string(index=False))

# reusable HARD tyre model
def estimate_hard_tyre_model(driver, clean_lap_data, usable_stints):
    driver_stints = usable_stints[usable_stints["Driver"] == driver]["Stint"].unique()
    driver_hard_laps = clean_lap_data[(clean_lap_data["Driver"] == driver) & (clean_lap_data["Compound"] == "HARD") 
                                      & (clean_lap_data["FreshTyre"] == True) & (clean_lap_data["Stint"].isin(driver_stints))].copy()
    if len(driver_stints) < 2:
        return None
    
    X = np.column_stack([np.ones(len(driver_hard_laps)), driver_hard_laps["TyreLife"], driver_hard_laps["LapNumber"]])
    y = driver_hard_laps["LapTimeSeconds"].to_numpy()
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    intercept = coefficients[0]
    tyre_effect = coefficients[1]
    race_effect = coefficients[2]
    
    # residuals
    driver_hard_laps["PredictedLapTime"] = (intercept + tyre_effect*driver_hard_laps["TyreLife"] + race_effect*driver_hard_laps["LapNumber"])
    driver_hard_laps["Residual"] = (driver_hard_laps["LapTimeSeconds"] - driver_hard_laps["PredictedLapTime"])
    median_residual = (driver_hard_laps["Residual"].median())
    mad = (driver_hard_laps["Residual"] - median_residual).abs().median()
    robust_sigma = 1.4826*mad
    outlier_threshold = 3*robust_sigma
    if robust_sigma > 0:
        driver_hard_laps["Outlier"] = ((driver_hard_laps["Residual"] - median_residual).abs() > outlier_threshold)
    else: driver_hard_laps["Outlier"] = False
    
    # refitting without outliers
    model_laps = driver_hard_laps[driver_hard_laps["Outlier"] == False].copy()
    X_clean = np.column_stack([np.ones(len(model_laps)), model_laps["TyreLife"], model_laps["LapNumber"]])
    y_clean = model_laps["LapTimeSeconds"].to_numpy()
    
    clean_coefficients, _, _, _ = np.linalg.lstsq(X_clean, y_clean, rcond=None)
    clean_intercept = clean_coefficients[0]
    clean_tyre_effect = clean_coefficients[1]
    clean_race_effect = clean_coefficients[2]
    
    model_laps["PredictedLapTime"] = (clean_intercept + clean_tyre_effect*model_laps["TyreLife"] + clean_race_effect*model_laps["LapNumber"])
    model_laps["Residual"] = (model_laps["LapTimeSeconds"] - model_laps["PredictedLapTime"])
    rmse = np.sqrt(np.mean(model_laps["Residual"] ** 2))
    
    return {"Driver": driver, "TyreEffect": clean_tyre_effect, "RaceEffect": clean_race_effect, "RMSE": rmse, "LapsUsed": len(model_laps), "LapsRemoved":
            (len(driver_hard_laps) - len(model_laps))}
        

# cross-driver validation
driver_model_results = []
for driver in elig_drivers["Driver"]:
    result = estimate_hard_tyre_model(driver, all_clean_laps, hard_stints_usable)
    if result is not None:
        driver_model_results.append(result)
driver_results = pd.DataFrame(driver_model_results)
driver_results = driver_results.sort_values("TyreEffect").reset_index(drop=True)
print("\nCross-driver HARD tyre model results:")
print(driver_results.to_string(index=False, formatters={"TyreEffect": "{:.3f}".format, "RaceEffect": "{:.3f}".format, "RMSE": "{:.3f}".format}))

# summarising model results
median_tyre_effect = (driver_results["TyreEffect"].median())
mean_tyre_effect = (driver_results["TyreEffect"].mean())
std_deviation = (driver_results["TyreEffect"].std())
print("\nCross-driver Summary:")
print(f"Mean Tyre Effect: {mean_tyre_effect:.3f} s/lap")
print(f"Median Tyre Effect: {median_tyre_effect:.3f} s/lap")
print(f"Standard Deviation: {std_deviation:.3f} s/lap")
 
# McLaren Racing results only
mclaren_results = driver_results[driver_results["Driver"].isin(["NOR", "PIA"])]
print("\nMcLaren Racing Results:")
print(mclaren_results.to_string(index=False, formatters={"TyreEffect": "{:.3f}".format, "RaceEffect": "{:.3f}".format, "RMSE": "{:.3f}".format}))


median_rmse = driver_results["RMSE"].median()
rmse_mad = (driver_results["RMSE"] - median_rmse).abs().median()
rmse_robust_sigma = 1.4826*rmse_mad
rmse_threshold = median_rmse + 3*rmse_robust_sigma

driver_results["PoorFit"] = driver_results["RMSE"] > rmse_threshold
print("\nModel Quality Screening:")
print(f"Median RMSE: {median_rmse:.3f} s")
print(f"RMSE MAD: {rmse_mad:.3f} s")
print(f"Poor-fit Threshold: {rmse_threshold:.3f} s")
print("\nDrivers flagged as poor fits:")
print(driver_results.loc[driver_results["PoorFit"], ["Driver", "TyreEffect", "RaceEffect", "RMSE", "LapsUsed"]].to_string(
    index=False, formatters={"TyreEffect": "{:.3f}".format, "RaceEffect": "{:.3f}".format, "RMSE": "{:.3f}".format}))


validated_results = driver_results[driver_results["PoorFit"] == False].copy()
validated_mean = validated_results["TyreEffect"].mean()
validated_median = validated_results["TyreEffect"].median()
validated_std = validated_results["TyreEffect"].std()
validated_results.to_csv(results_dir / "hard_tyre_model_results.csv", index=False)
print("\nValidated Cross-driver Summary:")
print(f"Drivers retained: {len(validated_results)}/{len(driver_results)}")
print(f"Mean Tyre Effect: {validated_mean:.3f} s/lap")
print(f"Median Tyre Effect: {validated_median:.3f} s/lap")
print(f"Standard Deviation: {validated_std:.3f} s/lap")

plot_results = validated_results.sort_values("TyreEffect").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6))
for index, row in plot_results.iterrows():
    if row["Driver"] in ["NOR", "PIA"]:
        ax.scatter(row["TyreEffect"], index, s=100,
                   color=PAPAYA, edgecolor="black", linewidth=0.7, zorder=3) 
    else:
        ax.scatter(row["TyreEffect"], index, s=55, color="lightgrey",
                   edgecolor="grey", linewidth=0.5, zorder=2)
      
    
ax.axvline(validated_median, color="black", linestyle="--", linewidth=1.5, label=f"Field median ({validated_median:.3f} s/lap)")
ax.set_yticks(range(len(plot_results)))
ax.set_yticklabels(plot_results["Driver"])
ax.set_xlabel("Estimated HARD Tyre Age Effect (s/lap)")
ax.set_ylabel("Driver")
ax.set_title("Bahrain 2024 - Cross-Driver HARD Tyre Degradation\n")
ax.grid(axis="x", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "hard_tyre_validation.png", dpi=300, bbox_inches="tight")
plt.show()

        
# strategy model
race_length = 57
decision_lap = 20
current_tyre_age = 7
pit_loss = 22.0

mclaren_strategy_data = validated_results[validated_results["Driver"].isin(["NOR", "PIA"])]
strategy_tyre_effect = mclaren_strategy_data["TyreEffect"].mean()
strategy_race_effect = mclaren_strategy_data["RaceEffect"].mean()
current_lap_time = clean_laps.loc[clean_laps["LapNumber"] == decision_lap, "LapTimeSeconds"].iloc[0]
base_pace = (current_lap_time - strategy_tyre_effect*current_tyre_age - strategy_race_effect*decision_lap)

def predict_lap_time(lap_number, tyre_age):
    return(base_pace + strategy_tyre_effect*tyre_age + strategy_race_effect*lap_number)

def simulate_remaining_race(pit_lap=None):
    total_time = 0
    for lap_number in range(decision_lap + 1, race_length + 1):
        if pit_lap is None or lap_number <= pit_lap:
            tyre_age = current_tyre_age + (lap_number - decision_lap)
        else:
            tyre_age = lap_number - pit_lap
            
        total_time += predict_lap_time(lap_number, tyre_age)
    if pit_lap is not None:
        total_time += pit_loss
    return total_time

strategy_results = []
for pit_lap in range(25, 41):
    remaining_time = simulate_remaining_race(pit_lap)
    strategy_results.append({"PitLap": pit_lap, "RemainingRaceTime": remaining_time})
strategy_results = pd.DataFrame(strategy_results)

no_stop_time = simulate_remaining_race()
best_index = strategy_results["RemainingRaceTime"].idxmin()
best_pit_lap = int(strategy_results.loc[best_index, "PitLap"])
best_time = strategy_results.loc[best_index, "RemainingRaceTime"]
strategy_results["DeltaToBest"] = (strategy_results["RemainingRaceTime"] - best_time)
strategy_results["GainVsNoStop"] = (no_stop_time - strategy_results["RemainingRaceTime"])
strategy_results.to_csv(results_dir / "nominal_strategy_results.csv", index=False)
print("\nNominal Pit-stop Strategy:")
print(f"Best pit lap: {best_pit_lap}")
print(f"Predicted gain vs hypothetical no-stop baseline:" f"{no_stop_time - best_time:.2f} s")

# plotting pit-stop strategy performance
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(strategy_results["PitLap"], strategy_results["DeltaToBest"], marker="o", color=PAPAYA, linewidth=2)
ax.axvline(best_pit_lap, color="black", linestyle="--", linewidth=1.5, label=f"Best pit lap: {best_pit_lap}")
ax.set_xlabel("Pit Lap")
ax.set_ylabel("Predicted Time Loss vs Best Strategy (s)")
ax.set_title("Nominal Second-stop Strategy \nNorris, Bahrain 2024")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "nominal_pit_strategy.png", dpi=300, bbox_inches="tight")
plt.show()


rng = np.random.default_rng(42)
n_sims = 5000
preferred_pit_laps = []

for _ in range(n_sims):
    current_deg = rng.normal(strategy_tyre_effect, validated_std)
    new_deg = rng.normal(strategy_tyre_effect, validated_std)
    sim_base_pace = (current_lap_time - current_deg*current_tyre_age - strategy_race_effect*decision_lap)
    simulated_times = []
    for pit_lap in range(25, 41):
        total_time = 0
        for lap_number in range(decision_lap + 1, race_length + 1):
            if lap_number <= pit_lap:
                tyre_age = current_tyre_age + (lap_number - decision_lap)
                tyre_effect = current_deg
            else:
                tyre_age = lap_number - pit_lap
                tyre_effect = new_deg
                
            lap_time = (sim_base_pace + tyre_effect*tyre_age + strategy_race_effect*lap_number)
            total_time += lap_time
            
        total_time += pit_loss
        simulated_times.append(total_time)
        
    preferred_pit_laps.append(range(25, 41)[np.argmin(simulated_times)])
    
pit_lap_counts = pd.Series(preferred_pit_laps).value_counts().sort_index()
pit_lap_probability = (pit_lap_counts / n_sims*100)
robustness_results = pd.DataFrame({"Pitlap": pit_lap_probability.values})
robustness_results.to_csv(results_dir / "strategy_robustness_results.csv", index=False)
most_robust_pit_lap = int(pit_lap_probability.idxmax())
lower_window = int(np.percentile(preferred_pit_laps, 10))
upper_window = int(np.percentile(preferred_pit_laps, 90))
print("\nStrategy Robustness:")
print(f"Most frequently preferred pit lap: {most_robust_pit_lap}")
print(f"80% preferred pit-lap window: " f"{lower_window}-{upper_window}")
    
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(pit_lap_probability.index, pit_lap_probability.values, color=PAPAYA)
ax.axvline(best_pit_lap, color="black", linestyle="--", linewidth=1.5, label=f"Nominal optimum: lap {best_pit_lap}")
ax.set_xlabel("Preferred Pit Lap")
ax.set_ylabel("Share of Simulations (%)")
ax.set_title("Pit-stop Strategy Robustness\n" "5000 Degradation Scenarios")
ax.grid(axis="y", alpha=0.25)
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "strategy_robustness.png", dpi=300, bbox_inches="tight")
plt.show()
    
    
    
    
    
    
    
    