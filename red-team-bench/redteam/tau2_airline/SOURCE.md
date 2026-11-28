# tau2 airline agent

The agent under attack is tau2-bench's airline customer-service agent, written by Sierra, not by this team: its
policy, 14 tools and a database of 500 users, 2,000 reservations and 300 flights. It is not copied into this
repository. `scripts/fetch_tau2_airline.py` puts it here from
[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) at commit
`033bf082f48e4a3c6a1256208e93cf2ce9c3bb6f`, under the MIT license it comes with (`LICENSE`), and checks each file
against the one the published results ran on.

| File | What it is |
|---|---|
| `policy.md`, `db.json` | tau2's airline policy and database, unchanged |
| `system-prompt.txt` | The policy inside tau2's agent instructions, word for word as its LLM agent builds it |
| `first-message.txt` | tau2's default first message from the agent |
| `tool-definitions.json` | The schemas tau2's toolkit produces for its 14 tools, in the order `tools.py` defines them |
| `LICENSE` | tau2-bench's MIT license |

`redteam/airline_agent.py` ports the tools' logic (`src/tau2/domains/airline/tools.py`) to plain dicts over a fresh
copy of the database for each conversation: the same checks, results and error messages. An error reaches the agent
as "Error: <message>", as tau2's environment returns it.
