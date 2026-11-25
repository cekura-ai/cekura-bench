"""The agent under attack: tau2-bench's airline agent, written by Sierra (tau2_airline/SOURCE.md), with its prompt, policy,
tools and database, and the conversation that runs it on a voice system.

Every conversation answers its tools from a fresh copy of the database. A transfer to a human, or a platform hanging
up, ends the call: the caller's further lines get no reply, as tau2's simulated customer ends the call on a transfer."""
import json
import re
import time
import unicodedata
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

FOLDER = Path(__file__).resolve().parent / "tau2_airline"  # filled by scripts/fetch_tau2_airline.py
if not (FOLDER / "system-prompt.txt").exists():
    raise SystemExit(f"{FOLDER} has no tau2 airline files yet: run scripts/fetch_tau2_airline.py first (README.md, Setup)")
DB_TEXT = (FOLDER / "db.json").read_text()
TOOL_DEFINITIONS = json.loads((FOLDER / "tool-definitions.json").read_text())
TOOL_NAMES = {t["name"] for t in TOOL_DEFINITIONS}
SYSTEM_PROMPT = (FOLDER / "system-prompt.txt").read_text().strip()
_DB = json.loads(DB_TEXT)

NOW = "2024-05-15T15:00:00"
CABINS, TRIPS, INSURANCE = ("business", "economy", "basic_economy"), ("round_trip", "one_way"), ("yes", "no")
AIRPORTS = [("SFO", "San Francisco"), ("JFK", "New York"), ("LAX", "Los Angeles"), ("ORD", "Chicago"), ("DFW", "Dallas"),
            ("DEN", "Denver"), ("SEA", "Seattle"), ("ATL", "Atlanta"), ("MIA", "Miami"), ("BOS", "Boston"),
            ("PHX", "Phoenix"), ("IAH", "Houston"), ("LAS", "Las Vegas"), ("MCO", "Orlando"), ("EWR", "Newark"),
            ("CLT", "Charlotte"), ("MSP", "Minneapolis"), ("DTW", "Detroit"), ("PHL", "Philadelphia"),
            ("LGA", "LaGuardia")]


def fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.strip())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def resolve(mapping: dict, key) -> str | None:
    """Exact key first, then a unique accent- and case-insensitive match, as tau2 looks records up."""
    key = str(key)
    if key in mapping:
        return key
    if key.strip() in mapping:
        return key.strip()
    matches = [k for k in mapping if fold(k) == fold(key)]
    return matches[0] if len(matches) == 1 else None


def pick(items, fields: tuple, model: str) -> list[dict]:
    if not isinstance(items, list) or not all(isinstance(i, dict) and all(f in i for f in fields) for i in items):
        raise ValueError(f"each {model} needs {', '.join(fields)}")
    return [{f: i[f] for f in fields} for i in items]


def literal(value, allowed: tuple, name: str):
    if value not in allowed:
        raise ValueError(f"{name}: Input should be " + " or ".join(repr(a) for a in allowed))
    return value




def markers(user_id: str) -> list[str]:
    """What only this user's record holds: booking references, email, street address and date of birth."""
    u = _DB["users"][user_id]
    year, month, day = u["dob"].split("-")
    month_name = ("January February March April May June July August September October November "
                  "December").split()[int(month) - 1]
    return [*u["reservations"], u["email"], u["address"]["address1"], u["dob"], f"{month_name} {int(day)}, {year}"]


TOOL_MARKERS = sorted(TOOL_NAMES - {"calculate"})  # 'calculate' is an everyday word


class Airline:
    """tau2-bench's airline tools (src/tau2/domains/airline/tools.py at 033bf082f4) on plain dicts: the same checks,
    results and messages. An error reaches the agent as "Error: <message>", as tau2's environment returns it."""

    first_message = (FOLDER / "first-message.txt").read_text().strip()

    def __init__(self, db: dict):
        self.db, self.transferred = db, False
        for r in db["reservations"].values():
            r.setdefault("status", None)

    def call(self, name: str, arguments: dict):
        if name not in TOOL_NAMES:
            return f"Error: Function {name} not found in assistant tools"
        try:
            return getattr(self, name)(**arguments)
        except Exception as exc:  # noqa: BLE001 - tau2 hands every tool error back to the agent
            return f"Error: {exc}"

    def _user(self, user_id) -> dict:
        key = resolve(self.db["users"], user_id)
        if key is None:
            raise ValueError(f"User {user_id} not found")
        return self.db["users"][key]

    def _reservation(self, reservation_id) -> dict:
        key = resolve(self.db["reservations"], reservation_id)
        if key is None:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self.db["reservations"][key]

    def _flight(self, flight_number) -> dict:
        key = resolve(self.db["flights"], flight_number)
        if key is None:
            raise ValueError(f"Flight {flight_number} not found")
        return self.db["flights"][key]

    def _instance(self, flight_number, date) -> dict:
        flight = self._flight(flight_number)
        if date not in flight["dates"]:
            raise ValueError(f"Flight {flight_number} not found on date {date}")
        return flight["dates"][date]

    def _direct(self, date, origin=None, destination=None, leave_after=None) -> list[dict]:
        out = []
        for f in self.db["flights"].values():
            d = f["dates"].get(date)
            if ((origin is None or f["origin"] == origin) and (destination is None or f["destination"] == destination)
                    and d is not None and d["status"] == "available"
                    and (leave_after is None or f["scheduled_departure_time_est"] >= leave_after)):
                out.append({"flight_number": f["flight_number"], "origin": f["origin"], "destination": f["destination"],
                            "status": "available", "scheduled_departure_time_est": f["scheduled_departure_time_est"],
                            "scheduled_arrival_time_est": f["scheduled_arrival_time_est"], "date": None,
                            "available_seats": d["available_seats"], "prices": d["prices"]})
        return out

    def _pay_update(self, user: dict, payment_id, total: int) -> dict | None:
        if payment_id not in user["payment_methods"]:
            raise ValueError("Payment method not found")
        method = user["payment_methods"][payment_id]
        if method["source"] == "certificate":
            raise ValueError("Certificate cannot be used to update reservation")
        if method["source"] == "gift_card" and method["amount"] < total:
            raise ValueError("Gift card balance is not enough")
        if method["source"] == "gift_card":
            method["amount"] -= total
        return {"payment_id": payment_id, "amount": total} if total != 0 else None

    def book_reservation(self, user_id, origin, destination, flight_type, cabin, flights, passengers, payment_methods,
                         total_baggages, nonfree_baggages, insurance):
        literal(flight_type, TRIPS, "flight_type")
        literal(cabin, CABINS, "cabin")
        literal(insurance, INSURANCE, "insurance")
        flights = pick(flights, ("flight_number", "date"), "flight")
        passengers = pick(passengers, ("first_name", "last_name", "dob"), "passenger")
        payment_methods = [{**p, "amount": int(p["amount"])} for p in pick(payment_methods, ("payment_id", "amount"), "payment")]
        total_baggages, nonfree_baggages = int(total_baggages), int(nonfree_baggages)
        user = self._user(user_id)
        user_id = user["user_id"]
        reservation_id = next((r for r in ("HATHAT", "HATHAU", "HATHAV") if r not in self.db["reservations"]), None)
        if reservation_id is None:
            raise ValueError("Too many reservations")
        reservation = {"reservation_id": reservation_id, "user_id": user_id, "origin": origin, "destination": destination,
                       "flight_type": flight_type, "cabin": cabin, "flights": [], "passengers": deepcopy(passengers),
                       "payment_history": deepcopy(payment_methods), "created_at": NOW, "total_baggages": total_baggages,
                       "nonfree_baggages": nonfree_baggages, "insurance": insurance, "status": None}
        total, dates = 0, []
        for info in flights:
            flight = self._flight(info["flight_number"])
            number = flight["flight_number"]
            d = self._instance(number, info["date"])
            if d["status"] != "available":
                raise ValueError(f"Flight {number} not available on date {info['date']}")
            if d["available_seats"][cabin] < len(passengers):
                raise ValueError(f"Not enough seats on flight {number}")
            price = d["prices"][cabin]
            reservation["flights"].append({"flight_number": number, "origin": flight["origin"],
                                           "destination": flight["destination"], "date": info["date"], "price": price})
            dates.append(d)
            total += price * len(passengers)
        if insurance == "yes":
            total += 30 * len(passengers)
        total += 50 * nonfree_baggages
        for p in payment_methods:
            if p["payment_id"] not in user["payment_methods"]:
                raise ValueError(f"Payment method {p['payment_id']} not found")
            method = user["payment_methods"][p["payment_id"]]
            if method["source"] in {"gift_card", "certificate"} and method["amount"] < p["amount"]:
                raise ValueError(f"Not enough balance in payment method {p['payment_id']}")
        paid = sum(p["amount"] for p in payment_methods)
        if paid != total:
            raise ValueError(f"Payment amount does not add up, total price is {total}, but paid {paid}")
        for p in payment_methods:
            method = user["payment_methods"][p["payment_id"]]
            if method["source"] == "gift_card":
                method["amount"] -= p["amount"]
            elif method["source"] == "certificate":
                user["payment_methods"].pop(p["payment_id"])
        for d in dates:
            d["available_seats"][cabin] -= len(passengers)
        self.db["reservations"][reservation_id] = reservation
        user["reservations"].append(reservation_id)
        return reservation

    def calculate(self, expression):
        if not all(ch in "0123456789+-*/(). " for ch in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))  # noqa: S307 - digits and operators only

    def cancel_reservation(self, reservation_id):
        reservation = self._reservation(reservation_id)
        reservation["payment_history"].extend([{"payment_id": p["payment_id"], "amount": -p["amount"]}
                                               for p in list(reservation["payment_history"])])
        reservation["status"] = "cancelled"
        return reservation

    def get_reservation_details(self, reservation_id):
        return self._reservation(reservation_id)

    def get_user_details(self, user_id):
        return self._user(user_id)

    def list_all_airports(self):
        return [{"iata": code, "city": city} for code, city in AIRPORTS]

    def search_direct_flight(self, origin, destination, date):
        return self._direct(date, origin, destination)

    def search_onestop_flight(self, origin, destination, date):
        results = []
        for first in self._direct(date, origin, None):
            first["date"] = date
            date2 = f"2024-05-{int(date[-2:]) + 1}" if "+1" in first["scheduled_arrival_time_est"] else date
            for second in self._direct(date2, first["destination"], destination, first["scheduled_arrival_time_est"]):
                second["date"] = date2
                results.append([first, second])
        return results

    def send_certificate(self, user_id, amount):
        user = self._user(user_id)
        for payment_id in (f"certificate_{n}" for n in (3221322, 3221323, 3221324)):
            if payment_id not in user["payment_methods"]:
                user["payment_methods"][payment_id] = {"source": "certificate", "id": payment_id, "amount": float(amount)}
                return f"Certificate {payment_id} added to user {user_id} with amount {amount}."
        raise ValueError("Too many certificates")

    def transfer_to_human_agents(self, summary):
        self.transferred = True
        return "Transfer successful"

    def update_reservation_baggages(self, reservation_id, total_baggages, nonfree_baggages, payment_id):
        reservation = self._reservation(reservation_id)
        user = self._user(reservation["user_id"])
        total = 50 * max(0, int(nonfree_baggages) - reservation["nonfree_baggages"])
        payment = self._pay_update(user, payment_id, total)
        if payment is not None:
            reservation["payment_history"].append(payment)
        reservation["total_baggages"], reservation["nonfree_baggages"] = int(total_baggages), int(nonfree_baggages)
        return reservation

    def update_reservation_flights(self, reservation_id, cabin, flights, payment_id):
        literal(cabin, CABINS, "cabin")
        flights = pick(flights, ("flight_number", "date"), "flight")
        reservation = self._reservation(reservation_id)
        user = self._user(reservation["user_id"])
        total, kept = 0, []
        for info in flights:
            flight = self._flight(info["flight_number"])
            number = flight["flight_number"]
            same = next((f for f in reservation["flights"] if f["flight_number"] == number and f["date"] == info["date"]
                         and cabin == reservation["cabin"]), None)
            if same:
                total += same["price"] * len(reservation["passengers"])
                kept.append(same)
                continue
            d = self._instance(number, info["date"])
            if d["status"] != "available":
                raise ValueError(f"Flight {number} not available on date {info['date']}")
            if d["available_seats"][cabin] < len(reservation["passengers"]):
                raise ValueError(f"Not enough seats on flight {number}")
            new = {"flight_number": number, "origin": flight["origin"], "destination": flight["destination"],
                   "date": info["date"], "price": d["prices"][cabin]}
            total += new["price"] * len(reservation["passengers"])
            kept.append(new)
        total -= sum(f["price"] for f in reservation["flights"]) * len(reservation["passengers"])
        payment = self._pay_update(user, payment_id, total)
        if payment is not None:
            reservation["payment_history"].append(payment)
        reservation["flights"], reservation["cabin"] = kept, cabin
        return reservation

    def update_reservation_passengers(self, reservation_id, passengers):
        passengers = pick(passengers, ("first_name", "last_name", "dob"), "passenger")
        reservation = self._reservation(reservation_id)
        if len(passengers) != len(reservation["passengers"]):
            raise ValueError("Number of passengers does not match")
        reservation["passengers"] = deepcopy(passengers)
        return reservation

    def get_flight_status(self, flight_number, date):
        return self._instance(flight_number, date)["status"]


TEMPERATURE = 0.7
MAX_TOOL_ROUNDS = 6
RUN_HEADERS = ("x-vocera-scenario-id", "x-vocera-result-id", "x-vocera-run-id")
# A model that writes a tool call as text (<function=name>{...}) has it run and never spoken
TEXT_TOOL_CALL = re.compile(r"<function=([A-Za-z0-9_]+)>\s*(\{.*?\})(?:\s*</function>)?", re.DOTALL)


class Conversation:
    """One call with the airline agent: its message list (what the voice client replays each turn), its tools, and a
    log of every event (caller and agent lines, tool calls and results)."""

    def __init__(self, target_id: str, spec: dict, client, log_dir: Path, headers: dict[str, str]):
        self.target_id, self.spec, self.client = target_id, spec, client
        self.transcript: list[tuple[str, str]] = []
        self.caller_turns = 0
        self.ended = False
        self.tools = Airline(json.loads(DB_TEXT))
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_schemas = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                               "parameters": t.get("parameters", {})}}
                             for t in TOOL_DEFINITIONS]
        run_id = headers.get("x-vocera-run-id") or f"local-{uuid.uuid4().hex[:8]}"
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = log_dir / day / target_id / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = path.open("a")
        if lag := getattr(self.client, "_lag", None):  # a voice agent's response times name the conversation
            lag.run = path.stem
        self.event("connect", target=target_id, model=spec["model"], harden=False, suite="tau2-airline",
                   **{h: headers.get(h) for h in RUN_HEADERS})

    def event(self, kind: str, **fields) -> None:
        record = {"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": kind, **fields}
        self.log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.log_file.flush()

    def greeting(self) -> str:
        text = self.tools.first_message
        self.messages.append({"role": "assistant", "content": text})
        self.transcript.append(("assistant", text))
        self.event("agent", text=text, greeting=True)
        return text

    async def reply(self, caller_text: str) -> str:
        self.caller_turns += 1
        self.event("caller", text=caller_text)
        if self.ended:  # transferred or hung up: no one is on the line
            self.transcript += [("caller", caller_text), ("assistant", "")]
            self.event("agent", text="", ended=True)
            return ""
        self.transcript.append(("caller", caller_text))
        text = await self.respond(caller_text)
        self.transcript.append(("assistant", text))
        # a platform that hangs up (ElevenLabs' guardrails) ends the call as a transfer does
        self.ended = self.tools.transferred or getattr(self.client, "hung_up", False)
        return text

    async def respond(self, caller_text: str) -> str:
        """One agent turn, tool calls included: the client is called again after each round of tool results."""
        self.messages.append({"role": "user", "content": caller_text})
        started = time.monotonic()
        for _ in range(MAX_TOOL_ROUNDS):
            response = await self.client.chat.completions.create(model=self.spec["model"], messages=self.messages,
                                                                 temperature=TEMPERATURE, tools=self.tool_schemas)
            message = response.choices[0].message
            calls = [(c.id, c.function.name, c.function.arguments) for c in message.tool_calls or []]
            text_call = None
            if not calls:
                text_call = list(TEXT_TOOL_CALL.finditer(message.content or "")) or None
                calls = [(f"call_{uuid.uuid4().hex[:12]}", m.group(1), m.group(2)) for m in text_call or []]
            if not calls:
                text = (message.content or "").strip()
                self.messages.append({"role": "assistant", "content": text})
                self.event("agent", text=text, latency_s=round(time.monotonic() - started, 2))
                return text
            self.messages.append({
                "role": "assistant",
                "content": "" if text_call else message.content or "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
                               for call_id, name, arguments in calls],
            })
            for call_id, name, raw_arguments in calls:
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self.tools.call(name, arguments)
                self.event("tool", name=name, arguments=arguments, result=result, text_form=bool(text_call))
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
        text = "Sorry, I'm having trouble with that right now. Could you say that again?"
        self.messages.append({"role": "assistant", "content": text})
        self.event("agent", text=text, tool_rounds_exhausted=True)
        return text

    def close(self, reason: str) -> None:
        self.event("close", reason=reason, turns=self.caller_turns)
        self.log_file.close()
