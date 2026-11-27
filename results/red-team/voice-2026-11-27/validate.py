"""Check this frozen red-team result for internal consistency; no API calls. The Elo and risk need the per-goal scores,
which are held out with the goals, so they are checked for shape here, not refitted."""

import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    data = json.loads((Path(__file__).with_name("results.json")).read_text())
    agents, categories = data["agents"], data["categories"]
    require(sum(c["goals"] for c in categories) == data["goals"] == 32, "categories must cover the 32 goals")
    require(len(categories) == data["categoryCount"], "category count")
    for a in agents:
        i, name = a["id"], a["short"]
        for budget in ("oneAttack", "fiveAttacks"):
            require(a["broken"][budget] + a["partial"][budget] <= data["goals"], f"{name}: more goals than exist")
        require(a["broken"]["oneAttack"] <= a["broken"]["fiveAttacks"], f"{name}: broken at 1 above broken at 5")
        require(sum(c["broken"][i] for c in categories) == a["broken"]["fiveAttacks"], f"{name}: categories vs broken")
        require(sum(c["partial"][i] for c in categories) == a["partial"]["fiveAttacks"], f"{name}: categories vs partial")
        require(sum(a["breakProof"].values()) == a["broken"]["fiveAttacks"], f"{name}: proof kinds vs broken")
        risk = a["riskByAttacks"]
        require(len(risk) == data["attacksPerGoal"] and risk == sorted(risk), f"{name}: risk must not fall with attacks")
        for budget in ("oneAttack", "fiveAttacks"):
            e = a["elo"][budget]
            require(e["low"] <= e["value"] <= e["high"], f"{name}: Elo outside its interval")
        require(0 <= a["controlsPassed"] <= data["controls"], f"{name}: controls")
        print(f"{name:20} Elo {a['elo']['fiveAttacks']['value']:5}  broken {a['broken']['fiveAttacks']:2}/32"
              f" (+{a['partial']['fiveAttacks']} partial)  risk {risk[-1]:3}%  controls {a['controlsPassed']}/5")
    print(f"ok: {len(agents)} agents")


if __name__ == "__main__":
    main()
