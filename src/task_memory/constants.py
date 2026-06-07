from __future__ import annotations

LOCATIONS = ["start", "key_area", "door", "object_area", "target_area"]
OBJECT_FAMILIES = ["document", "vessel", "tool", "decor", "toy", "device", "parcel"]
TARGET_FAMILIES = ["storage", "disposal", "serving", "workspace"]
HANDLING_ACTIONS = ["none", "inspect_object", "use_cart", "secure_lid"]

ACTION_NAMES = [
    "move_to_key",
    "pickup_key",
    "move_to_door",
    "open_door",
    "clear_obstacle",
    "move_to_object",
    "inspect_object",
    "use_cart",
    "secure_lid",
    "pick_object",
    "move_to_target",
    "place_object",
]
ACTION_TO_ID = {name: index for index, name in enumerate(ACTION_NAMES)}
ID_TO_ACTION = {index: name for name, index in ACTION_TO_ID.items()}

ACTION_COST = {
    "move_to_key": 3,
    "pickup_key": 1,
    "move_to_door": 2,
    "open_door": 1,
    "clear_obstacle": 4,
    "move_to_object": 3,
    "inspect_object": 2,
    "use_cart": 3,
    "secure_lid": 2,
    "pick_object": 1,
    "move_to_target": 3,
    "place_object": 1,
}

OBJECT_CATALOG = [
    ("book", "document"),
    ("folder", "document"),
    ("cup", "vessel"),
    ("bottle", "vessel"),
    ("toolbox", "tool"),
    ("hammer", "tool"),
    ("vase", "decor"),
    ("plant_pot", "decor"),
    ("toy_car", "toy"),
    ("blocks", "toy"),
    ("tablet", "device"),
    ("camera", "device"),
    ("package", "parcel"),
    ("storage_crate", "parcel"),
]

TARGET_CATALOG = [
    ("bookshelf", "storage"),
    ("drawer", "storage"),
    ("cabinet", "storage"),
    ("trash_bin", "disposal"),
    ("recycle_bin", "disposal"),
    ("serving_tray", "serving"),
    ("dish_rack", "serving"),
    ("desk_zone", "workspace"),
    ("charging_zone", "workspace"),
]
