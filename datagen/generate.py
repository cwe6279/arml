"""
Generate ARML training data: varied tools and requests with known right answers, a teacher
model that answers them in ARML, and a check on every session before it is kept.

    python datagen/generate.py --show 5                          # look at scenarios, run nothing
    python datagen/generate.py --teacher oracle --n 2000         # templated answers, no model
    python datagen/generate.py --url http://localhost:8002/v1 --model qwen3.8-flash-next --n 500

Each scenario is built by code, not by a model: a toolset drawn from a catalog (names and
parameter names vary), a request, the calls that request needs and what the answer must say.
The tools are fakes with fixed data. So a session can be checked: the right calls with the
right arguments, no call the request did not ask for, no invented tool, an answer that says
the right thing. Every turn is logged in the shared format with that outcome
(`turns.jsonl`), and the turns that are clean under `strict` and belong to a session that
passed become chat-format training examples (`train.jsonl`).

The teacher is any OpenAI-compatible server. `--teacher oracle` needs no model: it writes
the known-right calls and a templated answer. That teaches the format but not natural
wording; mix it with a real teacher's sessions.

Nothing here uses the benchmark's tools or requests (bench/native_vs_arml.py), so a model
trained on this data can be scored on the benchmark. See datagen/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arml import format_rules, lint, parse, tools_notice                 # noqa: E402
from arml.record import TurnRecord, append, read, training_example       # noqa: E402

# The benchmark's tools. Nothing generated here may use these names.
HELD_OUT_TOOLS = {"weather", "calculate", "clock", "calendar", "contacts", "send_message", "set_timer"}

# ── the catalog ─────────────────────────────────────────────────────────────
# A lookup answers a question about one thing. data: thing -> (result text, the token an
# answer must carry). numeric: the token is a number, so two things can be compared.
LOOKUPS: List[dict] = [
    dict(key="stock", names=["stock_price", "get_quote", "share_price"], params=["ticker", "symbol"],
         desc="Latest share price for a ticker symbol.", noun="share price", numeric=True,
         data={"ACME": ("142.15 USD", "142.15"), "GLOBX": ("58.35 USD", "58.35"), "INIT": ("211.85 USD", "211.85"),
               "ZENO": ("17.42 USD", "17.42"), "HOLI": ("96.05 USD", "96.05")}, unknown=["QQXZ", "BRVT"],
         asks=["What's {e} trading at?", "How much is one share of {e} right now?", "Check the price of {e} for me.",
               "Give me the latest quote for {e}."], vague="What's the share price?", question="Which ticker?"),
    dict(key="flight", names=["flight_status", "get_flight", "track_flight"], params=["flight", "flight_number"],
         desc="Status and gate of a flight.", noun="flight status",
         data={"UA 212": ("delayed 40 minutes, gate B14", "B14"), "LH 455": ("on time, gate A7", "A7"),
               "BA 98": ("boarding, gate C22", "C22"), "DL 31": ("cancelled, rebooking desk D3", "cancelled")},
         unknown=["ZZ 900", "QF 7711"],
         asks=["Is flight {e} on time?", "What gate is {e} leaving from?", "Any news on flight {e}?",
               "Check the status of {e}."], vague="Is my flight on time?", question="Which flight number?"),
    dict(key="order", names=["order_status", "track_order", "get_order"], params=["order_id", "order"],
         desc="Where a shop order is.", noun="order status",
         data={"A-1047": ("shipped, arriving Thursday", "Thursday"), "A-2210": ("being packed", "packed"),
               "B-0931": ("delivered to the front desk", "front desk"), "C-5518": ("refunded", "refunded")},
         unknown=["X-0000", "K-7777"],
         asks=["Where is order {e}?", "Has order {e} shipped yet?", "What's happening with my order {e}?",
               "Track order {e}."], vague="Where's my order?", question="What is the order number?"),
    dict(key="balance", names=["account_balance", "get_balance", "balance"], params=["account", "account_name"],
         desc="Current balance of one of the user's accounts.", noun="balance", numeric=True,
         data={"checking": ("1840.55 EUR", "1840.55"), "savings": ("12300.75 EUR", "12300.75"),
               "travel": ("412.25 EUR", "412.25"), "business": ("7655.45 EUR", "7655.45")}, unknown=["pension", "kids"],
         asks=["How much is in my {e} account?", "What's the balance of {e}?", "Check my {e} balance.",
               "Tell me what I have in {e}."], vague="What's my balance?", question="Which account?"),
    dict(key="bus", names=["next_bus", "bus_arrival", "bus_times"], params=["stop", "stop_name"],
         desc="Minutes until the next bus at a stop.", noun="next bus", numeric=True,
         data={"Harbor Street": ("7 minutes, line 12", "7"), "Mill Road": ("23 minutes, line 4", "23"),
               "Central Park": ("11 minutes, line 30", "11"), "Old Town": ("36 minutes, line 4", "36")},
         unknown=["Moon Avenue", "Pier 99"],
         asks=["When's the next bus at {e}?", "How long until a bus comes to {e}?", "Next bus from {e}?",
               "Is there a bus soon at {e}?"], vague="When's the next bus?", question="Which stop?"),
    dict(key="battery", names=["battery_level", "device_battery", "get_battery"], params=["device", "device_name"],
         desc="Battery charge of one of the user's devices, in percent.", noun="battery level", numeric=True,
         data={"phone": ("64 percent", "64"), "laptop": ("31 percent", "31"), "watch": ("88 percent", "88"),
               "headphones": ("12 percent", "12"), "tablet": ("47 percent", "47")}, unknown=["drone", "camera"],
         asks=["How charged is my {e}?", "What's the battery on the {e}?", "Does my {e} need charging?",
               "Battery level of the {e}?"], vague="How's the battery?", question="Which device?"),
    dict(key="room", names=["room_temperature", "get_temperature", "read_thermometer"], params=["room", "zone"],
         desc="Temperature measured in a room of the house, in Celsius.", noun="temperature", numeric=True,
         data={"kitchen": ("21.5 C", "21.5"), "bedroom": ("18.2 C", "18.2"), "office": ("23.4 C", "23.4"),
               "garage": ("9.6 C", "9.6"), "nursery": ("20.3 C", "20.3")}, unknown=["attic", "sauna"],
         asks=["How warm is the {e}?", "What's the temperature in the {e}?", "Is it cold in the {e}?",
               "Read the thermometer in the {e}."], vague="How warm is it inside?", question="Which room?"),
    dict(key="stockroom", names=["inventory_count", "stock_level", "items_in_stock"], params=["item", "product"],
         desc="How many units of a product are in the warehouse.", noun="stock level", numeric=True,
         data={"blue mugs": ("240 units", "240"), "desk lamps": ("18 units", "18"), "notebooks": ("1375 units", "1375"),
               "umbrellas": ("0 units, restock due Monday", "0"), "backpacks": ("62 units", "62")},
         unknown=["snow globes", "kazoos"],
         asks=["How many {e} do we have in stock?", "Check the stock of {e}.", "Are there any {e} left?",
               "What's the inventory for {e}?"], vague="How much stock do we have?", question="Which product?"),
    dict(key="hours", names=["opening_hours", "store_hours", "get_hours"], params=["place", "location"],
         desc="Today's opening hours of a place.", noun="opening hours",
         data={"the library": ("open 9:00 to 20:00", "20:00"), "the pharmacy": ("open 8:00 to 18:30", "18:30"),
               "the pool": ("closed today for cleaning", "closed"), "the bakery": ("open 6:30 to 14:00", "14:00")},
         unknown=["the planetarium", "the ice rink"],
         asks=["When does {e} close today?", "Is {e} open today?", "What are the hours for {e}?",
               "How late is {e} open?"], vague="What time does it close?", question="Which place?"),
    dict(key="queue", names=["wait_time", "queue_length", "current_wait"], params=["desk", "service"],
         desc="Current waiting time at a service desk, in minutes.", noun="wait", numeric=True,
         data={"passport office": ("95 minutes", "95"), "returns desk": ("8 minutes", "8"),
               "help desk": ("26 minutes", "26"), "pharmacy counter": ("14 minutes", "14")},
         unknown=["visa desk", "lost and found"],
         asks=["How long is the wait at the {e}?", "Is there a queue at the {e}?", "What's the wait for the {e} right now?",
               "Check the line at the {e}."], vague="How long is the wait?", question="Which desk?"),
    dict(key="server", names=["server_status", "host_health", "check_host"], params=["host", "hostname"],
         desc="Health of a server.", noun="server status",
         data={"web-01": ("healthy, load 0.42", "0.42"), "db-02": ("degraded, disk 97 percent full", "97"),
               "cache-01": ("healthy, load 0.11", "0.11"), "mail-03": ("down since 03:15", "03:15")},
         unknown=["gpu-09", "backup-07"],
         asks=["Is {e} up?", "How is {e} doing?", "Check on {e} for me.", "What's the status of {e}?"],
         vague="Is the server up?", question="Which host?"),
    dict(key="build", names=["build_status", "ci_status", "last_build"], params=["project", "repo"],
         desc="Result of the latest CI build of a project.", noun="build status",
         data={"atlas": ("passed, 412 tests", "412"), "beacon": ("failed, 3 tests failing", "3"),
               "comet": ("running, 60 percent done", "60"), "delta": ("passed, 88 tests", "88")},
         unknown=["zephyr", "quasar"],
         asks=["Did the {e} build pass?", "What's the build status of {e}?", "Is CI green on {e}?",
               "How did the last {e} build go?"], vague="Did the build pass?", question="Which project?"),
    dict(key="ticket", names=["ticket_status", "get_ticket", "issue_status"], params=["ticket_id", "ticket"],
         desc="State and owner of a support ticket.", noun="ticket status",
         data={"T-301": ("open, assigned to Noor", "Noor"), "T-322": ("waiting on customer", "customer"),
               "T-350": ("closed as duplicate", "duplicate"), "T-377": ("open, unassigned", "unassigned")},
         unknown=["T-999", "T-004"],
         asks=["Who has ticket {e}?", "What's the state of {e}?", "Is {e} still open?", "Look up ticket {e}."],
         vague="What's the status of the ticket?", question="Which ticket number?"),
    dict(key="showtimes", names=["movie_showtimes", "showtimes", "film_times"], params=["film", "title"],
         desc="Tonight's screening times of a film at the local cinema.", noun="showtimes",
         data={"Paper Moons": ("18:10 and 21:00", "21:00"), "The Long Tide": ("19:45", "19:45"),
               "Copper Sky": ("17:30, 20:15 and 22:40", "20:15"), "Small Engines": ("sold out tonight", "sold out")},
         unknown=["Glass Harbor", "Nine Winters"],
         asks=["When is {e} playing tonight?", "What time can I see {e}?", "Showtimes for {e}?",
               "Is {e} on tonight?"], vague="What's playing when?", question="Which film?"),
    dict(key="parking", names=["parking_spaces", "garage_availability", "free_spaces"], params=["garage", "car_park"],
         desc="Free spaces in a parking garage right now.", noun="free parking", numeric=True,
         data={"North Garage": ("132 free spaces", "132"), "Station Garage": ("4 free spaces", "4"),
               "Market Garage": ("57 free spaces", "57"), "Riverside Garage": ("0 free spaces", "0")},
         unknown=["Airport Garage", "Castle Garage"],
         asks=["How many spaces are free at {e}?", "Can I park at {e} right now?", "Is {e} full?",
               "Check parking at {e}."], vague="Is there parking?", question="Which garage?"),
    dict(key="air", names=["air_quality", "aqi", "pollution_index"], params=["district", "area"],
         desc="Air quality index measured in a city district.", noun="air quality", numeric=True,
         data={"Northside": ("AQI 41, good", "41"), "Docklands": ("AQI 118, unhealthy for some", "118"),
               "Old Quarter": ("AQI 67, moderate", "67"), "Hillcrest": ("AQI 29, good", "29")},
         unknown=["Lakeshore", "Eastgate"],
         asks=["How's the air in {e}?", "What's the air quality in {e} right now?", "Is the air clean in {e} today?",
               "AQI for {e}?"], vague="How's the air quality?", question="Which district?"),
    dict(key="exchange", names=["exchange_rate", "fx_rate", "currency_rate"], params=["pair", "currency_pair"],
         desc="Current exchange rate for a currency pair.", noun="exchange rate", numeric=True,
         data={"EUR/USD": ("1.0842", "1.0842"), "GBP/USD": ("1.2715", "1.2715"), "USD/JPY": ("151.37", "151.37"),
               "AUD/USD": ("0.6581", "0.6581")}, unknown=["XYZ/ABC", "EUR/QQQ"],
         asks=["What's the {e} rate?", "Where is {e} trading?", "Give me the current {e} exchange rate.",
               "How much is {e} right now?"], vague="What's the exchange rate?", question="Which currency pair?"),
    dict(key="package", names=["package_tracking", "track_parcel", "parcel_status"], params=["tracking_number", "parcel_id"],
         desc="Where a parcel is, by tracking number.", noun="parcel status",
         data={"1Z993": ("out for delivery, due by 17:00", "17:00"), "RX4410": ("held at the depot", "depot"),
               "PK2077": ("delivered, signed by Okafor", "Okafor"), "JD5512": ("in customs", "customs")},
         unknown=["ZZ0001", "QQ9999"],
         asks=["Where's parcel {e}?", "Track {e} for me.", "Has {e} been delivered?", "What's the status of package {e}?"],
         vague="Where's my parcel?", question="What is the tracking number?"),
    dict(key="pollen", names=["pollen_count", "pollen_level", "allergy_forecast"], params=["city", "town"],
         desc="Today's pollen count in a town.", noun="pollen count", numeric=True,
         data={"Bramford": ("pollen 83, high", "83"), "Kelso": ("pollen 22, low", "22"), "Whitby": ("pollen 51, moderate", "51"),
               "Ashton": ("pollen 117, very high", "117")}, unknown=["Atlantis", "Erewhon"],
         asks=["What's the pollen like in {e}?", "Is the pollen bad in {e} today?", "Pollen count for {e}?",
               "How high is the pollen in {e}?"], vague="How's the pollen?", question="Which town?"),
    dict(key="library", names=["book_availability", "library_search", "find_book"], params=["title", "book_title"],
         desc="Whether the library has a book on the shelf.", noun="availability",
         data={"The Salt Road": ("2 copies on the shelf, aisle 7", "aisle 7"), "Quiet Machines": ("all copies out, due back 14 April", "14 April"),
               "A Year of Bread": ("1 copy on the shelf, aisle 3", "aisle 3"), "North of Nowhere": ("on order", "on order")},
         unknown=["The Unwritten Map", "Seven Pale Doors"],
         asks=["Does the library have {e}?", "Is {e} available to borrow?", "Can I get {e} from the library?",
               "Check if {e} is on the shelf."], vague="Is my book in?", question="Which book?"),
    dict(key="price", names=["product_price", "get_price", "price_check"], params=["product", "item"],
         desc="Current shop price of a product.", noun="price", numeric=True,
         data={"desk chair": ("189.00 EUR", "189.00"), "kettle": ("34.50 EUR", "34.50"), "bike lock": ("22.95 EUR", "22.95"),
               "headlamp": ("41.20 EUR", "41.20"), "yoga mat": ("27.80 EUR", "27.80")}, unknown=["jetpack", "time machine"],
         asks=["How much is the {e}?", "What does the {e} cost?", "Price of the {e}?", "Check what the {e} sells for."],
         vague="How much is it?", question="Which product?"),
    dict(key="traffic", names=["travel_time", "commute_time", "route_duration"], params=["destination", "to"],
         desc="Driving time from home to a destination in current traffic, in minutes.", noun="travel time", numeric=True,
         data={"the airport": ("48 minutes, heavy traffic", "48"), "the office": ("22 minutes", "22"),
               "the hospital": ("15 minutes", "15"), "the stadium": ("63 minutes, road works", "63")},
         unknown=["the moon", "Narnia"],
         asks=["How long to drive to {e}?", "What's the drive to {e} like right now?", "How many minutes to {e} in traffic?",
               "Travel time to {e}?"], vague="How long will the drive take?", question="Where to?"),
    dict(key="plant", names=["soil_moisture", "plant_sensor", "moisture_level"], params=["plant", "plant_name"],
         desc="Soil moisture of a plant, in percent.", noun="soil moisture", numeric=True,
         data={"fern": ("moisture 72 percent", "72"), "basil": ("moisture 19 percent", "19"), "monstera": ("moisture 44 percent", "44"),
               "lemon tree": ("moisture 33 percent", "33")}, unknown=["cactus", "bonsai"],
         asks=["Does the {e} need water?", "How wet is the soil of the {e}?", "Check the {e}'s moisture.", "Is the {e} dry?"],
         vague="Do the plants need water?", question="Which plant?"),
    dict(key="loyalty", names=["loyalty_points", "rewards_balance", "points_balance"], params=["program", "programme"],
         desc="Points balance in one of the user's loyalty programmes.", noun="points balance", numeric=True,
         data={"Aerolux airline": ("18450 points", "18450"), "BeanClub coffee": ("320 points", "320"), "StayWell hotel": ("7600 points", "7600"),
               "FreshCart grocery": ("1275 points", "1275")}, unknown=["LuckyStar casino", "OceanBlue cruise"],
         asks=["How many {e} points do I have?", "What's my {e} rewards balance?", "Check my {e} points.",
               "Points left on {e}?"], vague="How many points do I have?", question="Which programme?"),
    dict(key="water", names=["water_temperature", "sea_temperature", "lake_temperature"], params=["beach", "spot"],
         desc="Water temperature at a swimming spot, in Celsius.", noun="water temperature", numeric=True,
         data={"Shell Bay": ("17.8 C", "17.8"), "Long Strand": ("15.1 C", "15.1"), "Mirror Lake": ("21.6 C", "21.6"),
               "North Cove": ("13.4 C", "13.4")}, unknown=["Coral Lagoon", "Ice Point"],
         asks=["How warm is the water at {e}?", "Is {e} warm enough to swim?", "Water temperature at {e}?",
               "What's the sea like at {e} today?"], vague="How's the water?", question="Which beach?"),
    dict(key="vaccine", names=["pet_record", "vet_record", "pet_vaccinations"], params=["pet", "pet_name"],
         desc="When a pet's next vaccination is due.", noun="next vaccination",
         data={"Biscuit": ("rabies booster due 3 June", "3 June"), "Mochi": ("all up to date until next year", "up to date"),
               "Rex": ("overdue since February", "overdue"), "Pepper": ("annual jab due 19 August", "19 August")},
         unknown=["Ghost", "Sprocket"],
         asks=["When is {e} due at the vet?", "Is {e} up to date on jabs?", "Check {e}'s vaccinations.",
               "When's {e}'s next vaccination?"], vague="When is the pet due at the vet?", question="Which pet?"),
    dict(key="elevator", names=["lift_status", "elevator_status", "check_lift"], params=["building", "block"],
         desc="Whether the lifts in a building are working.", noun="lift status",
         data={"Block A": ("both lifts working", "working"), "Block B": ("lift 2 out of service", "out of service"),
               "Block C": ("under maintenance until 14:00", "14:00"), "Tower": ("all four lifts working", "four")},
         unknown=["Block Z", "Annex"],
         asks=["Are the lifts working in {e}?", "Is the elevator in {e} running?", "Check the lifts at {e}.",
               "Lift status for {e}?"], vague="Is the lift working?", question="Which building?"),
    dict(key="meter", names=["energy_usage", "power_usage", "meter_reading"], params=["meter", "circuit"],
         desc="Energy used today on a circuit, in kilowatt hours.", noun="energy use", numeric=True,
         data={"heating": ("14.6 kWh today", "14.6"), "kitchen": ("5.3 kWh today", "5.3"), "car charger": ("21.9 kWh today", "21.9"),
               "office": ("2.8 kWh today", "2.8")}, unknown=["hot tub", "workshop"],
         asks=["How much power has the {e} used today?", "Energy use on {e} so far?", "Check the {e} meter.",
               "What has {e} drawn today?"], vague="How much energy have we used?", question="Which circuit?"),
    dict(key="recipe", names=["recipe_lookup", "get_recipe", "find_recipe"], params=["dish", "recipe"],
         desc="Cooking time and oven temperature for a saved recipe.", noun="recipe",
         data={"banana bread": ("60 minutes at 175 C", "175"), "roast chicken": ("90 minutes at 200 C", "200"),
               "focaccia": ("25 minutes at 220 C", "220"), "flapjacks": ("30 minutes at 160 C", "160")},
         unknown=["dragon stew", "moon cake deluxe"],
         asks=["What temperature for the {e}?", "How long does the {e} bake?", "Pull up my {e} recipe.",
               "Oven settings for {e}?"], vague="How long does it bake?", question="Which recipe?"),
    dict(key="shift", names=["shift_schedule", "rota_lookup", "who_is_on"], params=["team", "shift"],
         desc="Who is on duty today for a team.", noun="rota",
         data={"night desk": ("Amara and Lukas", "Amara"), "support": ("Wen until 15:00, then Diego", "Diego"),
               "kitchen": ("Sofia", "Sofia"), "security": ("Tomasz", "Tomasz")}, unknown=["marketing", "legal"],
         asks=["Who's on the {e} today?", "Who is working {e} today?", "Check the {e} rota.", "Who covers {e} today?"],
         vague="Who's on duty?", question="Which team?"),

]

# An action changes something. Each parameter has a pool the request draws from. free: the
# parameter is free text, so a chain can put a looked-up fact in it.
ACTIONS: List[dict] = [
    dict(key="thermostat", names=["set_thermostat", "set_target_temperature", "heating_control"],
         desc="Set a room's target temperature in degrees Celsius.",
         params=[dict(key="room", names=["room", "zone"], type="string", pool=["hallway", "bedroom", "office", "living room"]),
                 dict(key="degrees", names=["degrees", "celsius", "target"], type="number", pool=[17, 18, 19, 20, 21, 22, 23])],
         asks=["Set the {room} to {degrees} degrees.", "Make it {degrees} in the {room}.",
               "Change the {room} thermostat to {degrees}.", "I'd like the {room} at {degrees} degrees, please."],
         result="{room} set to {degrees} C", vague="Change the thermostat.", question="Which room, and to what temperature?"),
    dict(key="shopping", names=["add_to_list", "shopping_list_add", "list_add"],
         desc="Add an item to one of the user's lists.",
         params=[dict(key="list", names=["list", "list_name"], type="string", pool=["groceries", "hardware", "packing", "gifts"]),
                 dict(key="item", names=["item", "entry"], type="string", pool=["oat milk", "wood glue", "sunscreen", "candles", "batteries", "rice"])],
         asks=["Add {item} to my {list} list.", "Put {item} on the {list} list.", "I need {item}. Add it to {list}.",
               "Can you add {item} to {list}?"],
         result="added {item} to {list}", vague="Add something to my list.", question="What should I add, and to which list?"),
    dict(key="reminder", names=["create_reminder", "remind_me", "add_reminder"],
         desc="Create a reminder for a time of day.",
         params=[dict(key="text", names=["text", "about"], type="string", free=True,
                      pool=["call the plumber", "water the plants", "renew the passport", "pick up the dry cleaning"]),
                 dict(key="time", names=["time", "at"], type="string", pool=["07:30", "12:15", "16:45", "19:00"])],
         asks=["Remind me to {text} at {time}.", "Set a reminder at {time}: {text}.", "At {time}, remind me to {text}.",
               "I need a reminder to {text} at {time}."],
         result="reminder set for {time}", vague="Set a reminder.", question="What should I remind you of, and when?"),
    dict(key="music", names=["play_music", "start_playback", "play"],
         desc="Play an artist or playlist on a speaker.",
         params=[dict(key="what", names=["query", "artist_or_playlist"], type="string", pool=["Nina Simone", "focus mix", "rainy day jazz", "The Kinks"]),
                 dict(key="speaker", names=["speaker", "room"], type="string", pool=["kitchen", "study", "patio"])],
         asks=["Play {what} in the {speaker}.", "Put on {what} on the {speaker} speaker.", "I want to hear {what} in the {speaker}.",
               "Start {what} on the {speaker} speaker."],
         result="playing {what} on {speaker}", vague="Play some music.", question="What would you like to hear, and on which speaker?"),
    dict(key="volume", names=["set_volume", "volume", "adjust_volume"],
         desc="Set a speaker's volume from 0 to 100.",
         params=[dict(key="speaker", names=["speaker", "device"], type="string", pool=["kitchen", "study", "patio", "bedroom"]),
                 dict(key="level", names=["level", "percent"], type="number", pool=[10, 25, 40, 55, 70, 85])],
         asks=["Set the {speaker} speaker to {level}.", "Volume {level} on the {speaker} speaker.",
               "Turn the {speaker} speaker to {level} percent.", "Make the {speaker} speaker volume {level}."],
         result="{speaker} volume {level}", vague="Change the volume.", question="Which speaker, and how loud?"),
    dict(key="door", names=["lock_door", "set_lock", "door_lock"],
         desc="Lock or unlock a door.",
         params=[dict(key="door", names=["door", "entrance"], type="string", pool=["front door", "back door", "garage door", "side gate"]),
                 dict(key="state", names=["state", "action"], type="string", pool=["locked", "unlocked"])],
         asks=["Make sure the {door} is {state}.", "I want the {door} {state}.", "Set the {door} to {state}.",
               "Please get the {door} {state}."],
         result="{door} is now {state}", vague="Do the lock.", question="Which door, and should it be locked or unlocked?"),
    dict(key="table", names=["book_table", "reserve_table", "make_reservation"],
         desc="Reserve a restaurant table.",
         params=[dict(key="restaurant", names=["restaurant", "venue"], type="string", pool=["Luma", "The Copper Pot", "Sakura House", "Fig & Olive"]),
                 dict(key="people", names=["people", "party_size"], type="number", pool=[2, 3, 4, 5, 6, 8]),
                 dict(key="time", names=["time", "at"], type="string", pool=["18:30", "19:00", "20:15", "21:00"])],
         asks=["Book a table for {people} at {restaurant} at {time}.", "Reserve {restaurant} for {people} people at {time}.",
               "Get us a table at {restaurant}, {people} people, {time}.", "I need a reservation at {restaurant} for {people} at {time}."],
         result="table for {people} at {restaurant}, {time}", vague="Book a table.", question="Which restaurant, for how many people, and at what time?"),
    dict(key="transfer", names=["transfer_money", "send_money", "make_transfer"],
         desc="Move money from the user's checking account to one of their other accounts.",
         params=[dict(key="to", names=["to_account", "destination"], type="string", pool=["savings", "travel", "business"]),
                 dict(key="amount", names=["amount", "eur"], type="number", pool=[25, 60, 150, 320, 475])],
         asks=["Move {amount} euros to my {to} account.", "Transfer {amount} to {to}.", "Put {amount} euros into {to}.",
               "Send {amount} from checking to {to}."],
         result="moved {amount} EUR to {to}", vague="Make a transfer.", question="How much, and to which account?"),
    dict(key="vacuum", names=["start_vacuum", "robot_clean", "clean_room"],
         desc="Send the robot vacuum to clean a room.",
         params=[dict(key="room", names=["room", "area"], type="string", pool=["hallway", "kitchen", "living room", "study"])],
         asks=["Vacuum the {room}.", "Send the robot to clean the {room}.", "The {room} needs vacuuming.",
               "Have the robot do the {room}."],
         result="cleaning the {room}", vague="Run the vacuum.", question="Which room?"),
    dict(key="ticket_new", names=["open_ticket", "create_issue", "file_ticket"],
         desc="Open a support ticket.",
         params=[dict(key="title", names=["title", "summary"], type="string", free=True,
                      pool=["printer on floor 2 is jammed", "VPN drops every hour", "badge reader not working", "projector has no sound"]),
                 dict(key="priority", names=["priority", "severity"], type="string", pool=["low", "normal", "high"])],
         asks=["Open a {priority} priority ticket: {title}.", "File a ticket, {priority} priority, saying {title}.",
               "Create a {priority} ticket for this: {title}.", "Log a {priority} priority issue: {title}."],
         result="ticket opened with {priority} priority", vague="Open a ticket.", question="What is the problem, and how urgent is it?"),
    dict(key="note", names=["create_note", "save_note", "write_note"],
         desc="Save a note.",
         params=[dict(key="text", names=["text", "body"], type="string", free=True,
                      pool=["the spare key is with the neighbours", "boiler serviced in March", "wifi guest code is on the fridge"])],
         asks=["Save a note: {text}.", "Write this down: {text}.", "Make a note that {text}.", "Note for later: {text}."],
         result="note saved", vague="Save a note.", question="What should the note say?"),
    dict(key="post", names=["post_update", "post_to_channel", "channel_message"],
         desc="Post a short update to a team channel.",
         params=[dict(key="channel", names=["channel", "room"], type="string", pool=["ops", "support", "general", "releases"]),
                 dict(key="text", names=["text", "message"], type="string", free=True,
                      pool=["deploy is done", "standup moved to 10", "office closed Friday", "lunch order goes in at 11"])],
         asks=["Post to {channel}: {text}.", "Tell the {channel} channel that {text}.", "Put this in {channel}: {text}.",
               "Send an update to {channel} saying {text}."],
         result="posted to {channel}", vague="Post an update.", question="Which channel, and what should it say?"),
    dict(key="lights", names=["set_lights", "light_control", "dim_lights"],
         desc="Set the brightness of the lights in a room, in percent.",
         params=[dict(key="room", names=["room", "zone"], type="string", pool=["kitchen", "hallway", "study", "bedroom"]),
                 dict(key="level", names=["brightness", "percent"], type="number", pool=[0, 20, 35, 50, 75, 100])],
         asks=["Set the {room} lights to {level} percent.", "Dim the {room} to {level}.", "Lights in the {room} at {level} please.",
               "Make the {room} lights {level} percent."],
         result="{room} lights at {level}%", vague="Change the lights.", question="Which room, and how bright?"),
    dict(key="email", names=["send_email", "compose_email", "email_send"],
         desc="Send an email.",
         params=[dict(key="to", names=["to", "recipient"], type="string", pool=["noor@example.org", "billing@example.org", "team@example.org"]),
                 dict(key="body", names=["body", "text"], type="string", free=True,
                      pool=["the invoice is attached", "I'll be ten minutes late", "the meeting room is booked"])],
         asks=["Email {to}: {body}.", "Send an email to {to} saying {body}.", "Write to {to}: {body}.",
               "Drop {to} an email that {body}."],
         result="email sent to {to}", vague="Send an email.", question="Who to, and what should it say?"),
    dict(key="alarm", names=["set_alarm", "create_alarm", "alarm_set"],
         desc="Set a wake-up alarm for a time of day.",
         params=[dict(key="time", names=["time", "at"], type="string", pool=["05:45", "06:30", "07:15", "08:00"])],
         asks=["Wake me at {time}.", "Set an alarm for {time}.", "I need an alarm at {time}.", "Alarm at {time} please."],
         result="alarm set for {time}", vague="Set an alarm.", question="For what time?"),
    dict(key="feeder", names=["feed_pet", "dispense_food", "pet_feeder"],
         desc="Dispense food from the automatic pet feeder, in grams.",
         params=[dict(key="grams", names=["grams", "amount"], type="number", pool=[20, 35, 50, 65])],
         asks=["Give the cat {grams} grams of food.", "Feed the dog {grams} grams.", "Dispense {grams} grams from the feeder.",
               "Put {grams} grams in the bowl."],
         result="dispensed {grams} g", vague="Feed the pet.", question="How much food?"),
    dict(key="blinds", names=["set_blinds", "blinds_control", "move_blinds"],
         desc="Open or close the blinds in a room, as a percentage open.",
         params=[dict(key="room", names=["room", "window"], type="string", pool=["bedroom", "living room", "study"]),
                 dict(key="open", names=["percent_open", "position"], type="number", pool=[0, 25, 50, 100])],
         asks=["Set the {room} blinds to {open} percent open.", "Blinds in the {room} to {open}.",
               "Open the {room} blinds to {open} percent.", "Put the {room} blinds at {open} percent."],
         result="{room} blinds {open}% open", vague="Adjust the blinds.", question="Which room, and how far open?"),
    dict(key="calendar_hold", names=["block_time", "hold_time", "create_event"],
         desc="Put a block of time in the user's diary this week. The day is a weekday name.",
         params=[dict(key="title", names=["title", "label"], type="string", free=True,
                      pool=["deep work", "school run", "dentist", "quarterly review prep"]),
                 dict(key="day", names=["day", "weekday"], type="string", pool=["Monday", "Wednesday", "Friday"]),
                 dict(key="start", names=["start", "from"], type="string", pool=["09:00", "13:30", "16:00"])],
         asks=["Block {day} at {start} for {title}.", "Put {title} in my diary on {day} at {start}.",
               "Hold {start} on {day} for {title}.", "Add {title} on {day} from {start}."],
         result="{title} on {day} at {start}", vague="Block some time.", question="What for, which day, and what time?"),
    dict(key="taxi", names=["book_ride", "call_taxi", "request_ride"],
         desc="Book a taxi to an address.",
         params=[dict(key="to", names=["destination", "drop_off"], type="string", pool=["Central Station", "the airport", "14 Elm Road", "City Hospital"]),
                 dict(key="time", names=["pickup_time", "at"], type="string", pool=["now", "08:15", "17:40", "22:30"])],
         asks=["Get me a taxi to {to} at {time}.", "Book a ride to {to}, pickup {time}.", "I need a cab to {to} at {time}.",
               "Order a taxi for {time} to {to}."],
         result="taxi to {to} at {time}", vague="Book me a taxi.", question="Where to, and when?"),
    dict(key="pause", names=["pause_subscription", "suspend_plan", "hold_subscription"],
         desc="Pause one of the user's subscriptions for some weeks.",
         params=[dict(key="service", names=["service", "subscription"], type="string", pool=["FreshBox meal kit", "PulseFit gym", "Streamio", "Field Notes magazine"]),
                 dict(key="weeks", names=["weeks", "duration_weeks"], type="number", pool=[1, 2, 4, 6])],
         asks=["Pause my {service} subscription for {weeks} weeks.", "Put the {service} plan on hold for {weeks} weeks.",
               "Suspend {service} for {weeks} weeks.", "Stop {service} for {weeks} weeks please."],
         result="{service} paused for {weeks} weeks", vague="Pause my subscription.", question="Which subscription, and for how long?"),
    dict(key="sprinkler", names=["water_garden", "run_sprinklers", "irrigation"],
         desc="Run the garden sprinklers in a zone for some minutes.",
         params=[dict(key="zone", names=["zone", "area"], type="string", pool=["front lawn", "vegetable beds", "back lawn"]),
                 dict(key="minutes", names=["minutes", "duration"], type="number", pool=[5, 10, 15, 20, 30])],
         asks=["Water the {zone} for {minutes} minutes.", "Run the sprinklers on the {zone} for {minutes} minutes.",
               "Give the {zone} {minutes} minutes of water.", "Sprinklers on the {zone}, {minutes} minutes."],
         result="watering {zone} for {minutes} min", vague="Water the garden.", question="Which zone, and for how long?"),
    dict(key="kudos", names=["send_kudos", "thank_colleague", "give_shoutout"],
         desc="Send a colleague a thank-you on the team board.",
         params=[dict(key="person", names=["colleague", "person"], type="string", pool=["Noor", "Diego", "Wen", "Amara"]),
                 dict(key="text", names=["message", "reason"], type="string", free=True,
                      pool=["covering my shift", "fixing the build", "the great demo"])],
         asks=["Send {person} kudos for {text}.", "Thank {person} on the board for {text}.",
               "Give {person} a shout-out for {text}.", "Post a thank-you to {person} for {text}."],
         result="kudos sent to {person}", vague="Send someone kudos.", question="Who to, and what for?"),
    dict(key="print", names=["print_document", "send_to_printer", "print_file"],
         desc="Print a document from the user's files.",
         params=[dict(key="file", names=["file", "document"], type="string", pool=["boarding-pass.pdf", "lease.pdf", "recipe-card.pdf", "report-q3.pdf"]),
                 dict(key="copies", names=["copies", "count"], type="number", pool=[1, 2, 3, 5])],
         asks=["Print {copies} copies of {file}.", "Send {file} to the printer, {copies} copies.", "I need {copies} printouts of {file}.",
               "Print {file}, {copies} copies please."],
         result="printing {copies} of {file}", vague="Print the document.", question="Which file, and how many copies?"),

]
# How to ask which of two things has more, per comparable lookup.
COMPARE_ASKS = {
    "stock": "Which is trading higher right now, {a} or {b}?",
    "balance": "Which account has more money in it, {a} or {b}?",
    "bus": "Which stop has the longer wait for a bus, {a} or {b}?",
    "battery": "Which is more charged, my {a} or my {b}?",
    "room": "Which is warmer right now, the {a} or the {b}?",
    "stockroom": "Do we have more {a} or more {b} in stock?",
    "queue": "Where is the wait longer right now, the {a} or the {b}?",
    "parking": "Which garage has more free spaces, {a} or {b}?",
    "air": "Which district has the higher air quality index, {a} or {b}?",
    "exchange": "Which rate is higher right now, {a} or {b}?",
    "pollen": "Where is the pollen count higher today, {a} or {b}?",
    "price": "Which costs more, the {a} or the {b}?",
    "traffic": "Which is the longer drive right now, {a} or {b}?",
    "plant": "Which has the wetter soil, the {a} or the {b}?",
    "loyalty": "Where do I have more points, {a} or {b}?",
    "water": "Where is the water warmer, {a} or {b}?",
    "meter": "Which circuit has used more energy today, {a} or {b}?",
}
CHAIN_ASKS = ["Look up the {noun} for {e} and {sink}.", "Check the {noun} of {e}, then {sink}.",
              "Get the current {noun} for {e} and {sink}.", "Find the {noun} for {e} and {sink} so I have it."]
SINK_PHRASE = {"note": "save it in a note", "post": "post it to the {channel} channel",
               "reminder": "put it in a reminder for {time}", "ticket_new": "open a {priority} priority ticket with it as the title",
               "email": "email it to {to}", "calendar_hold": "put it in my diary on {day} at {start}",
               "kudos": "thank {person} on the board with it"}

# Requests that need no tool: (request, a regex the answer must match or None, the oracle's answer)
NO_TOOL: List[Tuple[str, Optional[str], str]] = [
    ("What's the capital of Japan?", r"tokyo", "Tokyo."),
    ("How do you say 'thank you' in French?", r"merci", "Merci."),
    ("What's the chemical symbol for gold?", r"\bAu\b", "Au."),
    ("How many days are in a leap year?", r"366", "366."),
    ("What colour do you get from mixing blue and yellow?", r"green", "Green."),
    ("Which planet is known as the Red Planet?", r"mars", "Mars."),
    ("What's the opposite of 'generous'?", r"stingy|selfish|mean|miserly|greedy|tight", "Stingy."),
    ("How many sides does a hexagon have?", r"\b6\b|six", "Six."),
    ("Who wrote 'Pride and Prejudice'?", r"austen", "Jane Austen."),
    ("What's the plural of 'mouse'?", r"mice", "Mice."),
    ("What is 12 plus 30?", r"42", "42."),
    ("Spell 'necessary' for me.", r"necessary|n.?e.?c.?e.?s.?s.?a.?r.?y", "N-E-C-E-S-S-A-R-Y."),
    ("What's the boiling point of water at sea level in Celsius?", r"100", "100 degrees Celsius."),
    ("Which ocean is the largest?", r"pacific", "The Pacific."),
    ("Give me a synonym for 'quick'.", r"\w", "Fast."),
    ("Good morning!", None, "Good morning. How can I help?"),
    ("Thanks, that's everything.", None, "You're welcome."),
    ("Never mind, forget it.", None, "No problem."),
    ("Tell me a one-line joke.", r"\w", "I told my plants a joke. They rooted for me."),
    ("Suggest a name for a sailing boat.", r"\w", "Second Wind."),
    ("What's a good word to describe a very quiet room?", r"\w", "Hushed."),
    ("In one sentence, what does a compiler do?", r"\w", "It translates source code into a program a computer can run."),
    ("Is a tomato a fruit or a vegetable?", r"fruit", "Botanically it's a fruit."),
    ("How many minutes are in two hours?", r"120", "120."),
    ("What's the square root of 81?", r"\b9\b|nine", "9."),
    ("Which is bigger, a kilometre or a mile?", r"mile", "A mile."),
    ("How many legs does a spider have?", r"\b8\b|eight", "Eight."),
    ("What's the past tense of 'swim'?", r"swam", "Swam."),
    ("Name a primary colour.", r"red|blue|yellow", "Red."),
    ("What's the freezing point of water in Fahrenheit?", r"32", "32 degrees Fahrenheit."),
    ("Who painted the Mona Lisa?", r"leonardo|da vinci", "Leonardo da Vinci."),
    ("What does 'bonjour' mean?", r"hello|good (morning|day)", "Hello, or good day."),
    ("How many continents are there?", r"\b7\b|seven", "Seven."),
    ("What's the largest mammal?", r"whale", "The blue whale."),
    ("Rewrite 'pls send asap' politely.", r"\w", "Could you please send it as soon as you can?"),
    ("What's a word that rhymes with 'light'?", r"\w", "Bright."),
    ("Hi there.", None, "Hi. What can I do for you?"),
    ("That's perfect, thank you.", None, "Glad it helped."),
    ("Okay, cancel that, I'll do it myself.", None, "All right."),
    ("What's 15 percent of 200?", r"\b30\b", "30."),
    ("Which gas do plants take in?", r"carbon dioxide|co2|co₂", "Carbon dioxide."),
    ("In which country is Machu Picchu?", r"peru", "Peru."),

]
IDENTITIES = [
    "You are a careful assistant. Use a tool when the request needs one. When something you need is missing, ask one short question.",
    "You are a home assistant. Act through the tools you have. If a request leaves out something you need, ask for it briefly.",
    "You are an office helper for a small team. Use tools for anything you cannot know yourself, and ask one short question when details are missing.",
    "You are a concise assistant. Call a tool when the task needs it. Do not guess missing details: ask.",
    "You help one person with everyday tasks. Use the tools provided, never invent one, and ask when you are unsure what they mean.",
    "You are a reliable assistant. Look things up with tools instead of guessing, and ask a brief question if the request is incomplete.",
    "You run errands for a busy household. Use the tools for anything that changes or that you cannot know, and ask before guessing.",
    "You are a front-desk assistant. Look things up with tools, act only through tools, and keep answers short.",
    "You are a practical assistant for a small shop. Use tools when you need facts or must act; otherwise just answer.",
    "You are an assistant that never pretends. If a tool can do it, use the tool. If none can, say so. If details are missing, ask.",

]
KINDS = [("lookup", 22), ("action", 18), ("chain", 14), ("no tool", 14), ("compare", 10),
         ("missing", 10), ("impossible", 6), ("unknown", 6),
         # the tools change during the conversation, or one stops answering
         ("removed", 5), ("renamed", 5), ("added", 5), ("tool error", 5)]
# Kinds where a tool result that is an error is the expected outcome, not a mistake.
ERROR_EXPECTED = {"unknown", "tool error"}
ASKS_RE = r"\?|\b(provide|specify|tell me|let me know|which|who|whom|what|where|when|need)\b"


def _norm(value: Any) -> str:
    """Lowercase, a leading "the" dropped, only letters, digits and dots: "The Airport" and
    "airport" are the same place to a tool and to the checker."""
    text = re.sub(r"^\s*the\s+", "", str(value).lower())
    return re.sub(r"[^a-z0-9.]+", "", text)


def _find(table: dict, value: Any) -> Optional[str]:
    """A fake tool's lookup, as lenient as the checker: an exact key, else the one key that
    contains the value as whole words ("FreshCart" finds "FreshCart grocery", "C" finds
    "Block C"), else the one key the value contains ("the returns desk please")."""
    if _norm(value) in table:
        return table[_norm(value)][0]
    words = re.findall(r"[a-z0-9.]+", re.sub(r"^\s*the\s+", "", str(value).lower()))
    hits = [k for k in table if words and _contains_words(table[k][1], words)]
    if len(hits) != 1:
        hits = [k for k in table if k in _norm(value)]
    return table[hits[0]][0] if len(hits) == 1 else None


def _contains_words(key_words: List[str], words: List[str]) -> bool:
    n = len(words)
    return any(key_words[i:i + n] == words for i in range(len(key_words) - n + 1))


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(w.capitalize() for w in rest)


# ── building one scenario ───────────────────────────────────────────────────
def _instantiate(rng: random.Random, concept: dict, kind: str) -> dict:
    """A concept from the catalog as one concrete tool: a name, parameter names, a fake."""
    style = rng.choice([str, str, _camel])
    name = style(rng.choice(concept["names"]))
    if kind == "lookup":
        pname = style(rng.choice(concept["params"]))
        table = {_norm(k): (v[0], re.findall(r"[a-z0-9.]+", k.lower())) for k, v in concept["data"].items()}

        def run(p, table=table, pname=pname, noun=concept["noun"]):
            if pname not in p:
                return f"error: missing parameter {pname}"
            return _find(table, p[pname]) or f"error: no {noun} data for {p[pname]!r}"
        params = {pname: {"type": "string", "required": True}}
        return dict(concept=concept, kind=kind, name=name, desc=concept["desc"], params=params, run=run, pname=pname)
    pnames = {p["key"]: style(rng.choice(p["names"])) for p in concept["params"]}

    def run(p, concept=concept, pnames=pnames):
        vals = {}
        for spec in concept["params"]:
            if pnames[spec["key"]] not in p:
                return f"error: missing parameter {pnames[spec['key']]}"
            vals[spec["key"]] = p[pnames[spec["key"]]]
            if spec["type"] == "number":
                try:
                    vals[spec["key"]] = f"{float(vals[spec['key']]):g}"
                except (TypeError, ValueError):
                    return f"error: {pnames[spec['key']]} must be a number"
        return concept["result"].format(**vals)
    params = {pnames[p["key"]]: {"type": p["type"], "required": True} for p in concept["params"]}
    return dict(concept=concept, kind=kind, name=name, desc=concept["desc"], params=params, run=run, pnames=pnames)


def scenario(seed: int, index: int) -> dict:
    """Scenario number `index` for this seed. The same numbers always give the same scenario."""
    rng = random.Random(f"{seed}:{index}")
    kind = rng.choices([k for k, _ in KINDS], [w for _, w in KINDS])[0]
    lookups, actions = rng.sample(LOOKUPS, len(LOOKUPS)), rng.sample(ACTIONS, len(ACTIONS))
    sinks = [a for a in actions if a["key"] in SINK_PHRASE]
    used_l: List[dict] = []
    used_a: List[dict] = []
    absent: Optional[dict] = None
    if kind in ("lookup", "unknown"):
        used_l = lookups[:1]
    elif kind == "compare":
        used_l = [next(c for c in lookups if c.get("numeric"))]
    elif kind == "action":
        used_a = actions[:1]
    elif kind == "chain":
        used_l, used_a = lookups[:1], sinks[:1]
    elif kind == "missing":
        used_l, used_a = (lookups[:1], []) if rng.random() < 0.4 else ([], actions[:1])
    elif kind == "impossible":
        absent = rng.choice(lookups + actions)
    elif kind in ("removed", "renamed", "tool error"):
        used_l = lookups[:1]
    elif kind == "added":
        used_l, absent = lookups[1:2], lookups[0]       # [1] is used earlier; [0] arrives with the notice
    # distractors: other concepts, so the model has to choose
    pool = [c for c in lookups + actions if c not in used_l and c not in used_a and c is not absent]
    extra = rng.sample(pool, rng.randint(2, 6))
    tools = [_instantiate(rng, c, "lookup" if c in LOOKUPS else "action") for c in used_l + used_a + extra]
    rng.shuffle(tools)
    by_key = {t["concept"]["key"]: t for t in tools}
    need: List[Tuple[str, Dict[str, Any]]] = []
    forbid: set = set()
    s = dict(id=f"gen-{seed}-{index}", kind=kind, identity=rng.choice(IDENTITIES), tools=tools,
             answer_has=[], answer_not=[], answer_re=None, must_ask=False, any_answer=False, steps=[],
             history=[], notice="", live={t["name"]: t["run"] for t in tools}, removed=set(), max_calls={})

    def earlier(concept, entity):
        """An exchange already in the conversation: a lookup that worked, and its answer."""
        t = by_key[concept["key"]]
        args = {t["pname"]: entity}
        s["history"] += [{"role": "user", "content": rng.choice(concept["asks"]).format(e=entity)},
                         {"role": "assistant", "content": f'<tool name="{t["name"]}">{json.dumps(args)}</tool>'},
                         {"role": "user", "content": f"Result of {t['name']}: {t['run'](args)}"},
                         {"role": "assistant", "content": f"<output>The {concept['noun']} for {entity}: "
                                                          f"{concept['data'][entity][0]}.</output>"}]

    def look(concept, entity):
        t = by_key[concept["key"]]
        need.append((t["name"], {t["pname"]: entity}))
        s["steps"].append((t["name"], {t["pname"]: entity}))
        return t

    def act(concept, values):
        t = by_key[concept["key"]]
        free = {p["key"] for p in concept["params"] if p.get("free")}
        args = {t["pnames"][k]: (("words", v) if k in free else v) for k, v in values.items()}
        need.append((t["name"], args))
        s["steps"].append((t["name"], {t["pnames"][k]: v for k, v in values.items()}))   # what the oracle writes
        return t

    if kind == "lookup":
        c = used_l[0]
        e = rng.choice(list(c["data"]))
        look(c, e)
        s.update(ask=rng.choice(c["asks"]).format(e=e), answer_has=[c["data"][e][1]],
                 gold=f"The {c['noun']} for {e}: {c['data'][e][0]}.")
    elif kind == "unknown":
        c = used_l[0]
        e = rng.choice(c["unknown"])
        look(c, e)
        s.update(ask=rng.choice(c["asks"]).format(e=e), any_answer=True,
                 answer_not=[v[1] for v in c["data"].values() if len(v[1]) > 2],
                 gold=f"I couldn't find any {c['noun']} data for {e}.")
    elif kind == "compare":
        c = used_l[0]
        a, b = rng.sample(list(c["data"]), 2)
        look(c, a)
        look(c, b)
        hi, lo = (a, b) if float(c["data"][a][1]) > float(c["data"][b][1]) else (b, a)
        s.update(ask=COMPARE_ASKS[c["key"]].format(a=a, b=b) + " Answer with just the name.",
                 answer_has=[hi], answer_not=[lo], gold=f"{hi}.")
    elif kind == "action":
        c = used_a[0]
        values = {p["key"]: rng.choice(p["pool"]) for p in c["params"]}
        act(c, values)
        s.update(ask=rng.choice(c["asks"]).format(**values), any_answer=True, gold="Done: " + c["result"].format(**values) + ".")
    elif kind == "chain":
        c, sink = used_l[0], used_a[0]
        e = rng.choice(list(c["data"]))
        look(c, e)
        values = {p["key"]: rng.choice(p["pool"]) for p in sink["params"]}
        free = next(p["key"] for p in sink["params"] if p.get("free"))
        values[free] = f"{c['noun']} for {e}: {c['data'][e][0]}"
        act(sink, values)
        need[-1] = (need[-1][0], {**need[-1][1], by_key[sink["key"]]["pnames"][free]: ("has", c["data"][e][1])})
        phrase = SINK_PHRASE[sink["key"]].format(**values)
        s.update(ask=rng.choice(CHAIN_ASKS).format(noun=c["noun"], e=e, sink=phrase),
                 any_answer=True, gold="Done.")
    elif kind == "no tool":
        ask, regex, gold = rng.choice(NO_TOOL)
        forbid = {t["name"] for t in tools}
        s.update(ask=ask, answer_re=regex, any_answer=regex is None, gold=gold)
    elif kind == "missing":
        c = (used_l or used_a)[0]
        forbid = {by_key[c["key"]]["name"]}
        s.update(ask=c["vague"], must_ask=True, gold=c["question"])
    elif kind == "impossible":
        forbid = {t["name"] for t in tools}
        if absent in LOOKUPS:
            ask = rng.choice(absent["asks"]).format(e=rng.choice(list(absent["data"])))
        else:
            ask = rng.choice(absent["asks"]).format(**{p["key"]: rng.choice(p["pool"]) for p in absent["params"]})
        s.update(ask=ask, any_answer=True, gold="I can't do that here: I don't have a tool for it.")
    elif kind in ("removed", "renamed"):
        c = used_l[0]
        e1, e2 = rng.sample(list(c["data"]), 2)
        earlier(c, e1)
        old = by_key[c["key"]]
        ask2 = rng.choice(["And {e}?", "Now check {e}.", "What about {e}?", "Same for {e}, please."]).format(e=e2)
        del s["live"][old["name"]]
        s["removed"] = {old["name"]}
        if kind == "removed":
            s["notice"] = tools_notice(removed=[old["name"]])
            forbid = {t["name"] for t in tools}
            s.update(ask=ask2, any_answer=False, answer_not=[c["data"][e2][1]] if len(c["data"][e2][1]) > 2 else [],
                     gold=f"I can't check that any more: the {c['noun']} tool has been removed.")
        else:
            taken = {t["name"] for t in tools} | HELD_OUT_TOOLS
            new = next((style(n) for n in rng.sample(c["names"], len(c["names"])) for style in (str, _camel)
                        if style(n) not in taken), None) or old["name"] + "_v2"
            s["live"][new] = old["run"]
            s["notice"] = tools_notice(renamed={old["name"]: new})
            need.append((new, {old["pname"]: e2}))
            s["steps"].append((new, {old["pname"]: e2}))
            s.update(ask=ask2, answer_has=[c["data"][e2][1]], gold=f"The {c['noun']} for {e2}: {c['data'][e2][0]}.")
    elif kind == "added":
        first, c = used_l[0], absent
        earlier(first, rng.choice(list(first["data"])))
        newt = _instantiate(rng, c, "lookup")
        if newt["name"] in {t["name"] for t in tools}:
            newt["name"] += "_new"
        s["live"][newt["name"]] = newt["run"]
        s["notice"] = tools_notice(added=[{"name": newt["name"], "description": newt["desc"], "parameters": newt["params"]}])
        e = rng.choice(list(c["data"]))
        need.append((newt["name"], {newt["pname"]: e}))
        s["steps"].append((newt["name"], {newt["pname"]: e}))
        s.update(ask=rng.choice(c["asks"]).format(e=e), answer_has=[c["data"][e][1]],
                 gold=f"The {c['noun']} for {e}: {c['data'][e][0]}.")
        s["added_tool"] = newt
    elif kind == "tool error":
        c = used_l[0]
        e = rng.choice(list(c["data"]))
        broken = by_key[c["key"]]
        s["live"][broken["name"]] = lambda p, noun=c["noun"]: f"error: the {noun} service is not responding; try again later"
        look(c, e)
        s["max_calls"] = {broken["name"]: 2}             # one retry is fine; more is a loop
        s.update(ask=rng.choice(c["asks"]).format(e=e),
                 answer_not=[v[1] for v in c["data"].values() if len(v[1]) > 2],
                 gold=f"The {c['noun']} service isn't responding right now, so I can't check {e}. Try again in a little while.")
    s.update(need=need, forbid=forbid,
             actions={t["name"] for t in tools if t["kind"] == "action"} & set(s["live"]),
             names=set(s["live"]))
    assert not ((s["names"] | {t["name"] for t in tools}) & HELD_OUT_TOOLS)
    return s


def tool_specs(s: dict) -> List[dict]:
    return [{"name": t["name"], "description": t["desc"], "parameters": t["params"]} for t in s["tools"]]


def system_prompt(s: dict) -> str:
    return s["identity"] + "\n\n" + format_rules(tool_specs(s))


# ── checking a session ──────────────────────────────────────────────────────
_STOP = {"the", "a", "an", "to", "for", "of", "on", "in", "at", "is", "it", "my", "and", "with"}


def _words(text: Any) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower()) if w not in _STOP]


def _arg_ok(expected: Any, actual: Any) -> bool:
    """Whether a call's argument is right. Names match when one contains the other, so
    "garage" is the garage door and "unlock" is unlocked, but "back door" is not the front
    door. Free text is right when it keeps most of the words asked for."""
    if isinstance(expected, tuple) and expected[0] == "has":   # free text that must carry a fact
        return str(expected[1]).lower() in str(actual).lower()
    if isinstance(expected, tuple) and expected[0] == "words":
        want, got = _words(expected[1]), set(_words(actual))
        return bool(want) and sum(w in got for w in want) / len(want) >= 0.6
    if isinstance(expected, (int, float)):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False
    e, a = _norm(expected), _norm(actual)
    return bool(e and a) and (e in a or (len(a) >= 3 and a in e))


def check(s: dict, calls: List[Tuple[str, dict]], final: str) -> Tuple[bool, str]:
    """(passed, why not)"""
    left = list(calls)
    for name, args in s["need"]:
        hit = next((i for i, (n, p) in enumerate(left)
                    if n == name and all(k in p and _arg_ok(v, p[k]) for k, v in args.items())), None)
        if hit is None:
            return False, f"no correct {name} call"
        left.pop(hit)
    made = [n for n, _ in calls]
    gone = [n for n in made if n in s.get("removed", ())]
    if gone:
        return False, f"called {gone[0]}, which the runtime had removed"
    for tool, most in s.get("max_calls", {}).items():
        if made.count(tool) > most:
            return False, f"called {tool} {made.count(tool)} times after it failed"
    unknown = [n for n in made if n not in s["names"]]
    if unknown:
        return False, f"invented a tool: {unknown[0]}"
    bad = [n for n in made if n in s["forbid"]]
    if bad:
        return False, f"called {bad[0]}, which this request did not call for"
    for tool in s["actions"]:                           # an action runs as often as asked, no more
        wanted = sum(1 for n, _ in s["need"] if n == tool)
        if made.count(tool) != wanted:
            return False, f"{tool} called {made.count(tool)} times, expected {wanted}"
    text = (final or "").strip()
    low = text.lower()
    if s["must_ask"] and not re.search(ASKS_RE, text, re.I):
        return False, "did not ask for what is missing"
    if s["answer_re"] and not re.search(s["answer_re"], text, re.I):
        return False, "the answer does not say it" if text else "no answer"
    for token in s["answer_has"]:
        if str(token).lower() not in low:
            return False, "the answer does not say it" if text else "no answer"
    for token in s["answer_not"]:
        if re.search(rf"(?<!\w){re.escape(str(token).lower())}(?!\w|\.\d)", low):
            return False, f"the answer says {token!r}, which it should not"
    if not text and not (s["any_answer"] and s["need"]):
        return False, "no answer"
    return True, ""


# ── the teacher ─────────────────────────────────────────────────────────────
def make_chat(url: str, model: str, key: str = "", reasoning_effort: str = "", timeout: int = 300,
              temperature: float = 0.3) -> Callable[[List[dict]], str]:
    def chat(messages):
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False,
                                "temperature": temperature, "max_tokens": 2048}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"].get("content") or ""
    return chat


def oracle(s: dict, rng: random.Random) -> Callable[[List[dict]], str]:
    """The known-right session, written as ARML. No model: the calls come from the scenario and
    the answer is a template."""
    steps = list(s["steps"])
    before = sum(1 for m in s.get("history", []) if m["role"] == "assistant")

    def chat(messages):
        done = sum(1 for m in messages if m["role"] == "assistant") - before
        think = "<think>{}</think>\n" if rng.random() < 0.5 else "{0:.0}"
        if done < len(steps):
            name, args = steps[done]
            return think.format(f"This needs {name}.") + f'<tool name="{name}">{json.dumps(args)}</tool>'
        return think.format("I have what I need to answer." if steps else "No tool is needed for this.") \
            + f"<output>{s['gold']}</output>"
    return chat


def run_session(s: dict, chat, model: str, max_steps: int = 6) -> dict:
    system = system_prompt(s)                    # the tools as they were at the start, never rewritten
    ask = (s["notice"] + "\n\n" + s["ask"]) if s.get("notice") else s["ask"]
    messages = [{"role": "system", "content": system}, *s.get("history", []), {"role": "user", "content": ask}]
    fakes = s["live"]
    calls, records, codes, final = [], [], [], ""
    for step in range(1, max_steps + 1):
        raw = chat(messages)
        codes += [i.code for i in lint(raw) if i.level in ("error", "warn")]
        rec = TurnRecord.from_response(raw, model=model, harness="arml-datagen/1", session=s["id"], turn=step,
                                       context=messages[1:], system=system, keep_system=True, tags=[s["kind"]],
                                       redact=None)     # synthetic data: redacting would teach "<email>"
        records.append(rec)
        messages.append({"role": "assistant", "content": raw})
        parsed = parse(raw)
        tools = [] if parsed.is_halted() else parsed.tools()
        if not tools:
            final = parsed.visible_text().strip()
            break
        tool = tools[0]
        try:
            params = tool.params()
        except Exception:
            params = {}
        params = params if isinstance(params, dict) else {}
        calls.append((tool.name, params))
        result = fakes[tool.name](params) if tool.name in fakes else f"error: no tool named {tool.name}"
        ok = not result.startswith("error:")
        rec.set_outcome(tool_ok=ok, **({} if ok else {"tool_error": result[:200]}))
        messages.append({"role": "user", "content": f"Result of {tool.name}: {result}"})
    else:
        codes.append("MAX_STEPS")
    passed, why = check(s, calls, final)
    if "MAX_STEPS" in codes:
        passed, why = False, "still calling tools after the step limit"
    # A lookup that fails because the thing does not exist is the right call in an "unknown" scenario.
    for rec in records:
        if s["kind"] in ERROR_EXPECTED and rec.outcome.get("tool_ok") is False and passed:
            rec.outcome["tool_ok"] = True
            rec.outcome.pop("tool_error", None)
        rec.set_outcome(task_ok=passed, **({} if passed else {"note": why}))
    return {"records": records, "passed": passed, "why": why, "codes": codes, "calls": calls, "final": final}


# ── running ─────────────────────────────────────────────────────────────────
def describe(s: dict) -> str:
    lines = [f"{s['id']}  [{s['kind']}]  {s['ask']}",
             "  tools: " + ", ".join(f"{t['name']}({', '.join(t['params'])})" for t in s["tools"])]
    if s.get("history"):
        lines.append(f"  earlier: {s['history'][0]['content']}  ->  {s['history'][1]['content']}")
    if s.get("notice"):
        lines.append("  notice: " + " | ".join(l for l in s["notice"].splitlines()[1:] if l.strip()))
    for name, args in s["need"]:
        lines.append(f"  needs: {name} {json.dumps({k: (v[1] if isinstance(v, tuple) else v) for k, v in args.items()})}")
    if s["forbid"]:
        lines.append("  must not call: " + (", ".join(sorted(s["forbid"])) if len(s["forbid"]) < len(s["names"]) else "any tool"))
    lines.append(f"  right answer, for example: {s['gold']}")
    return "\n".join(lines)


def generate(n: int, seed: int, teacher: Callable[[dict], Callable], model: str, out_dir: str,
             max_steps: int = 6, quiet: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    turns_path, train_path = os.path.join(out_dir, "turns.jsonl"), os.path.join(out_dir, "train.jsonl")
    done = {r.session for r in read(turns_path)} if os.path.exists(turns_path) else set()
    stats: Dict[str, Any] = {"scenarios": 0, "passed": 0, "skipped": 0, "errors": 0, "by_kind": {}, "why": {}, "codes": {}}
    for i in range(n):
        s = scenario(seed, i)
        if s["id"] in done:
            stats["skipped"] += 1
            continue
        try:
            r = run_session(s, teacher(s), model, max_steps)
        except Exception as e:                          # a server error: leave this one for the next run
            stats["errors"] += 1
            if not quiet:
                print(f"  {s['id']:14} {s['kind']:10} ERROR {type(e).__name__}: {e}", flush=True)
            continue
        for rec in r["records"]:
            append(turns_path, rec)
        stats["scenarios"] += 1
        stats["passed"] += r["passed"]
        k = stats["by_kind"].setdefault(s["kind"], [0, 0])
        k[0] += r["passed"]
        k[1] += 1
        if not r["passed"]:
            stats["why"][r["why"].split(":")[0]] = stats["why"].get(r["why"].split(":")[0], 0) + 1
        for c in r["codes"]:
            stats["codes"][c] = stats["codes"].get(c, 0) + 1
        if not quiet:
            flags = f"  [{', '.join(sorted(set(r['codes'])))}]" if r["codes"] else ""
            print(f"  {s['id']:14} {s['kind']:10} {'ok  ' if r['passed'] else 'FAIL'} {r['why']}{flags}", flush=True)
    # the training file is rebuilt from the whole log each time
    records = list(read(turns_path)) if os.path.exists(turns_path) else []
    examples = [ex for rec in records if (ex := training_example(rec, strict=True))]
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats.update(turns_logged=len(records), examples=len(examples), turns=turns_path, train=train_path,
                 tool_names=len({b.split(":", 1)[1] for r in records for b in r.blocks if b.startswith("tool:")}))
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200, help="scenarios to run")
    ap.add_argument("--seed", type=int, default=1, help="a different seed gives different scenarios")
    ap.add_argument("--teacher", default="model", choices=["model", "oracle"])
    ap.add_argument("--url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="", help="the teacher model's name on the server")
    ap.add_argument("--key", default=os.environ.get("OPENAI_API_KEY", ""), help="if your server wants one")
    ap.add_argument("--reasoning-effort", default="", help="sent as reasoning_effort when set")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--show", type=int, default=0, metavar="N", help="print N scenarios and stop")
    a = ap.parse_args()
    if a.show:
        for i in range(a.show):
            print(describe(scenario(a.seed, i)) + "\n")
        return 0
    if a.teacher == "oracle":
        name = "oracle"
        teacher = lambda s: oracle(s, random.Random(s["id"]))             # noqa: E731
    else:
        if not a.model:
            ap.error("--model is required with a model teacher")
        name = a.model
        chat = make_chat(a.url, a.model, a.key, a.reasoning_effort, temperature=a.temperature)
        teacher = lambda s: chat                                          # noqa: E731
    t0 = time.time()
    print(f"Teacher: {name}. {a.n} scenarios, seed {a.seed}. Output in {a.out}")
    st = generate(a.n, a.seed, teacher, name, a.out, a.max_steps)
    print(f"\nRan {st['scenarios']} scenarios in {time.time() - t0:.0f} s"
          + (f" ({st['skipped']} already in the log)" if st["skipped"] else "")
          + (f"; {st['errors']} server errors" if st["errors"] else "") + ".")
    if st["scenarios"]:
        print(f"Sessions that passed the check: {st['passed']} of {st['scenarios']}")
        for kind, (ok, total) in sorted(st["by_kind"].items()):
            print(f"  {kind:11} {ok} of {total}")
        if st["why"]:
            print("Why sessions failed: " + ", ".join(f"{k} x{v}" for k, v in sorted(st["why"].items(), key=lambda kv: -kv[1])))
        if st["codes"]:
            print("Lint errors and warnings: " + ", ".join(f"{k} x{v}" for k, v in sorted(st["codes"].items())))
    print(f"Log: {st['turns_logged']} turns in {st['turns']}")
    print(f"Training examples (strict, from passed sessions): {st['examples']} in {st['train']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
