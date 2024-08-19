import numpy as np
import random
import copy
from collections import defaultdict, deque
from itertools import chain
from enum import IntEnum

# Enums and Constants

class BuildingType(IntEnum):
    Settlement = 0
    City = 1

class PlayerId(IntEnum):
    White = 1
    Blue = 2
    Orange = 3
    Red = 4

class Terrain(IntEnum):
    Desert = 0
    Hills = 1
    Forest = 2
    Mountains = 3
    Pastures = 4
    Fields = 5

class Resource(IntEnum):
    Empty = 0
    Brick = 1
    Wood = 2
    Ore = 3
    Sheep = 4
    Wheat = 5

class DevelopmentCard(IntEnum):
    Knight = 0
    VictoryPoint = 1
    YearOfPlenty = 2
    RoadBuilding = 3
    Monopoly = 4

class ActionTypes(IntEnum):
    PlaceSettlement = 0
    PlaceRoad = 1
    UpgradeToCity = 2
    BuyDevelopmentCard = 3
    PlayDevelopmentCard = 4
    ExchangeResource = 5
    ProposeTrade = 6
    RespondToOffer = 7
    MoveRobber = 8
    RollDice = 9
    EndTurn = 10
    StealResource = 11
    DiscardResource = 12

TILE_ADJACENCY_INDS = [
    [[1, "R"], [3, "BL"], [4, "BR"]],
    [[0, "L"], [2, "R"], [4, "BL"], [5, "BR"]],
    [[1, "L"], [5, "BL"], [6, "BR"]],
    [[0, "TR"], [4, "R"], [7, "BL"], [8, "BR"]],
    [[0, "TL"], [1, "TR"], [3, "L"], [5, "R"], [8, "BL"], [9, "BR"]],
    [[1, "TL"], [2, "TR"], [4, "L"], [6, "R"], [9, "BL"], [10, "BR"]],
    [[2, "TL"], [5, "L"], [10, "BL"], [11, "BR"]],
    [[3, "TR"], [8, "R"], [12, "BR"]],
    [[3, "TL"], [4, "TR"], [7, "L"], [9, "R"], [12, "BL"], [13, "BR"]],
    [[4, "TL"], [5, "TR"], [8, "L"], [10, "R"], [13, "BL"], [14, "BR"]],
    [[5, "TL"], [6, "TR"], [9, "L"], [11, "R"], [14, "BL"], [15, "BR"]],
    [[6, "TL"], [10, "L"], [15, "BL"]],
    [[7, "TL"], [8, "TR"], [13, "R"], [16, "BR"]],
    [[8, "TL"], [9, "TR"], [12, "L"], [14, "R"], [16, "BL"], [17, "BR"]],
    [[9, "TL"], [10, "TR"], [13, "L"], [15, "R"], [17, "BL"], [18, "BR"]],
    [[10, "TL"], [11, "TR"], [14, "L"], [18, "BL"]],
    [[12, "TL"], [13, "TR"], [17, "R"]],
    [[13, "TL"], [14, "TR"], [16, "L"], [18, "R"]],
    [[14, "TL"], [15, "TR"], [17, "L"]]
]

TILE_NEIGHBOURS = []
for inds in TILE_ADJACENCY_INDS:
    tile_dict = {}
    for ind_lab in inds:
        tile_dict[ind_lab[1]] = ind_lab[0]
    TILE_NEIGHBOURS.append(copy.copy(tile_dict))

PREV_CORNER_LOOKUP = {
    "T": [["TR", "BL"], ["TL", "BR"]],
    "TR": [["TR", "B"]],
    "TL": [["TL", "B"], ["L", "TR"]],
    "BL": [["L", "BR"]],
    "BR": [[None, None]],
    "B": [[None, None]]
}
PREV_EDGE_LOOKUP = {
    "L": ["L", "R"],
    "TL": ["TL", "BR"],
    "TR": ["TR", "BL"],
    "R": [],
    "BR": [],
    "BL": []
}
CORNER_NEIGHBOURS_IN_TILE = {
    "T": {"TR": "TR", "TL": "TL"},
    "TL": {"BL": "L", "T": "TL"},
    "BL": {"TL": "L", "B": "BL"},
    "B": {"BL": "BL", "BR": "BR"},
    "BR": {"B": "BR", "TR": "R"},
    "TR": {"BR": "R", "T": "TR"}
}

HARBOUR_CORNER_AND_EDGES = {
    0: [0, "TL", "T", "TL"],
    1: [1, "T", "TR", "TR"],
    2: [6, "T", "TR", "TR"],
    3: [11, "TR", "BR", "R"],
    4: [15, "BR", "B", "BR"],
    5: [17, "BR", "B", "BR"],
    6: [16, "B", "BL", "BL"],
    7: [12, "TL", "BL", "L"],
    8: [3, "TL", "BL", "L"]
}

# Utility Functions

def DFS(G, v, seen=None, path=None):
    if seen is None: seen = []
    if path is None: path = [v]

    seen.append(v)

    paths = []
    for t in G[v]:
        if t not in seen:
            t_path = path + [t]
            paths.append(tuple(t_path))
            paths.extend(DFS(G, t, seen[:], t_path))
    return paths

def check_resource_conservation(env, res_tot=39):
    conserved = True
    for res in [Resource.Wood, Resource.Brick, Resource.Wheat, Resource.Sheep, Resource.Ore]:
        sum = env.game.resource_bank[res]
        for player in [PlayerId.Blue, PlayerId.Red, PlayerId.White, PlayerId.Orange]:
            sum += env.game.players[player].resources[res]
        if sum != res_tot:
            conserved = False
            break
    return conserved

# Game Components

class Building(object):
    def __init__(self, type: BuildingType, owner: PlayerId, corner: None):
        self.type = type
        self.owner = owner
        self.corner = corner

class Corner(object):
    def __init__(self, id):
        self.id = id
        self.neighbours_placed = 0
        self.adjacent_tiles_placed = 0
        self.corner_neighbours = [[None, None], [None, None], [None, None]]
        self.adjacent_tiles = [None, None, None]
        self.harbour = None
        self.building = None

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

    def insert_building(self, building):
        self.building = building

    def can_place_settlement(self, player, initial_placement=False):
        player_roads = 0
        if self.building is not None:
            return False
        for corner in self.corner_neighbours:
            if corner[0] is not None:
                if corner[0].building is not None:
                    return False
                if corner[1].road is not None and corner[1].road == player:
                    player_roads += 1
        if initial_placement:
            return True
        if player_roads > 0:
            return True
        else:
            return False

    def insert_neighbour(self, corner, edge):
        self.corner_neighbours[self.neighbours_placed][0] = corner
        self.corner_neighbours[self.neighbours_placed][1] = edge
        self.neighbours_placed += 1

    def insert_adjacent_tile(self, tile):
        self.adjacent_tiles[self.adjacent_tiles_placed] = tile
        self.adjacent_tiles_placed += 1

class Edge(object):
    def __init__(self, id):
        self.id = id
        self.corner_1 = None
        self.corner_2 = None
        self.road = None
        self.harbour = None

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

    def insert_corners(self, corner_1: Corner, corner_2: Corner):
        self.corner_1 = corner_1
        self.corner_2 = corner_2

    def can_place_road(self, player: PlayerId, after_second_settlement=False, second_settlement=None):
        if self.road is not None:
            return False
        else:
            if after_second_settlement:
                if self.corner_1.id == second_settlement or self.corner_2.id == second_settlement:
                    return True
                else:
                    return False
            if (self.corner_1.building is not None and self.corner_1.building.owner == player) or \
                    (self.corner_2.building is not None and self.corner_2.building.owner == player):
                return True
            else:
                for corner in [self.corner_1, self.corner_2]:
                    for next_corner in corner.corner_neighbours:
                        if next_corner[1] is not None:
                            if next_corner[1].road == player:
                                if corner.building is None:
                                    return True
        return False

    def insert_road(self, player: PlayerId):
        self.road = player

class Harbour(object):
    def __init__(self, resource: Resource = None, exchange_value=3, id=None):
        self.resource = resource
        self.exchange_value = exchange_value
        self.corners = []
        self.edge = None
        self.id = id

class Tile(object):
    def __init__(self, terrain: Terrain, value: int, id: int = None):
        self.terrain = terrain
        self.resource = Resource(terrain)
        self.value = value
        if value == 7:
            self.likelihood = None
        elif value == 6 or value == 8:
            self.likelihood = 5
        elif value == 5 or value == 9:
            self.likelihood = 4
        elif value == 4 or value == 10:
            self.likelihood = 3
        elif value == 3 or value == 11:
            self.likelihood = 2
        else:
            self.likelihood = 1
        self.id = id
        self.contains_robber = False

        self.corners = {
            "T": None,
            "TL": None,
            "BL": None,
            "B": None,
            "BR": None,
            "TR": None
        }
        self.edges = {
            "BL": None,
            "BR": None,
            "L": None,
            "R": None,
            "TL": None,
            "TR": None
        }

    def __eq__(self, other):
        if self.id == other.id:
            return True
        else:
            return False

class Board(object):
    def __init__(self, randomise_number_placement=True, fixed_terrain_placements=None, fixed_number_order=None):
        self.DEFAULT_NUMBER_ORDER = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
        self.NUMBER_PLACEMENT_INDS = [0, 3, 7, 12, 16, 17, 18, 15, 11, 6, 2, 1, 4, 8, 13, 14, 10, 5, 9]
        self.TERRAIN_TO_PLACE = [Terrain.Desert] + [Terrain.Hills] * 3 + [Terrain.Fields] * 4 + \
                                [Terrain.Forest] * 4 + [Terrain.Mountains] * 3 + [Terrain.Pastures] * 4
        self.HARBOURS_TO_PLACE = [Harbour(Resource.Ore, exchange_value=2, id=0), Harbour(Resource.Sheep, exchange_value=2, id=1),
                                  Harbour(Resource.Wheat, exchange_value=2, id=2), Harbour(Resource.Wood, exchange_value=2, id=3),
                                  Harbour(Resource.Brick, exchange_value=2, id=4), Harbour(None, exchange_value=3, id=5),
                                  Harbour(None, exchange_value=3, id=6), Harbour(None, exchange_value=3, id=7),
                                  Harbour(None, exchange_value=3, id=8)]

        self.randomise_number_placement = randomise_number_placement
        self.fixed_terrain_placements = fixed_terrain_placements
        self.fixed_number_order = fixed_number_order

        if fixed_terrain_placements is not None:
            assert np.array_equal([fixed_terrain_placements.count(terrain) == self.TERRAIN_TO_PLACE.count(terrain)
                for terrain in [Terrain.Hills, Terrain.Forest, Terrain.Fields, Terrain.Pastures, Terrain.Mountains]])

        if fixed_number_order is not None:
            assert np.array_equal([fixed_number_order.count(n) == self.DEFAULT_NUMBER_ORDER.count(n) for n in
                                   range(2, 13)])

        self.reset()
        self.build_adjacency_matrices()

    def validate_number_order(self, number_order, terrain_order):
        tile_vals = {}
        n_ind = 0
        for i in range(19):
            if terrain_order[self.NUMBER_PLACEMENT_INDS[i]] == Terrain.Desert:
                tile_vals[self.NUMBER_PLACEMENT_INDS[i]] = 7
            else:
                tile_vals[self.NUMBER_PLACEMENT_INDS[i]] = number_order[n_ind]
                n_ind += 1
        for i in range(19):
            if tile_vals[i] == 6 or tile_vals[i] == 8:
                for key in TILE_NEIGHBOURS[i]:
                    neighbour_ind = TILE_NEIGHBOURS[i][key]
                    if tile_vals[neighbour_ind] == 6 or tile_vals[neighbour_ind] == 8:
                        return False
        return True

    def reset(self):
        if self.fixed_terrain_placements is not None:
            terrain_order = copy.copy(self.fixed_terrain_placements)
        else:
            terrain_order = copy.copy(self.TERRAIN_TO_PLACE)
            np.random.shuffle(terrain_order)

        if self.fixed_number_order is not None:
            number_order = copy.copy(self.fixed_number_order)
        else:
            number_order = copy.copy(self.DEFAULT_NUMBER_ORDER)
            if self.randomise_number_placement:
                np.random.shuffle(number_order)
                while self.validate_number_order(number_order, terrain_order) == False:
                    np.random.shuffle(number_order)

        self.harbours = copy.copy(self.HARBOURS_TO_PLACE)
        np.random.shuffle(self.harbours)
        for harbour in self.harbours:
            harbour.corners = []

        self.tiles = tuple([Tile(terrain_order[i], -1, i) for i in range(19)])
        self.value_to_tiles = {}
        num_ind = 0
        for i in range(19):
            if self.tiles[self.NUMBER_PLACEMENT_INDS[i]].terrain == Terrain.Desert:
                self.tiles[self.NUMBER_PLACEMENT_INDS[i]].value = 7
                self.tiles[self.NUMBER_PLACEMENT_INDS[i]].contains_robber = True
                self.robber_tile = self.tiles[self.NUMBER_PLACEMENT_INDS[i]]
            else:
                self.tiles[self.NUMBER_PLACEMENT_INDS[i]].value = number_order[num_ind]
                self.value_to_tiles[number_order[num_ind]] = self.value_to_tiles.get(number_order[num_ind], []) + \
                    [self.tiles[self.NUMBER_PLACEMENT_INDS[i]]]
                num_ind += 1
        self.corners = tuple([Corner(id=i) for i in range(54)])
        self.edges = tuple([Edge(id=i) for i in range(72)])
        corner_ind = 0
        edge_ind = 0

        for tile_ind in range(19):
            for corner_location in self.tiles[tile_ind].corners.keys():
                prev_info = PREV_CORNER_LOOKUP[corner_location]
                prev_tile_ind = None
                prev_corner_loc = None
                if len(prev_info) > 0:
                    for info in prev_info:
                        ind = TILE_NEIGHBOURS[tile_ind].get(info[0], None)
                        if ind is not None:
                            prev_tile_ind = ind
                            prev_corner_loc = info[1]
                            break
                if prev_tile_ind is None:
                    self.tiles[tile_ind].corners[corner_location] = self.corners[corner_ind]
                    corner_ind += 1
                else:
                    self.tiles[tile_ind].corners[corner_location] = self.tiles[prev_tile_ind].corners[prev_corner_loc]

            for edge_location in self.tiles[tile_ind].edges.keys():
                prev_info = PREV_EDGE_LOOKUP[edge_location]
                prev_tile_ind = None
                prev_edge_loc = None
                if len(prev_info) > 0:
                    ind = TILE_NEIGHBOURS[tile_ind].get(prev_info[0], None)
                    if ind is not None:
                        prev_tile_ind = ind
                        prev_edge_loc = prev_info[1]
                if prev_tile_ind is None:
                    self.tiles[tile_ind].edges[edge_location] = self.edges[edge_ind]
                    edge_ind += 1
                else:
                    self.tiles[tile_ind].edges[edge_location] = self.tiles[prev_tile_ind].edges[prev_edge_loc]

        for tile_ind in range(19):
            for corner_loc, corner in self.tiles[tile_ind].corners.items():
                for n_corner_loc in CORNER_NEIGHBOURS_IN_TILE[corner_loc].keys():
                    edge_loc = CORNER_NEIGHBOURS_IN_TILE[corner_loc][n_corner_loc]
                    edge = self.tiles[tile_ind].edges[edge_loc]
                    n_corner = self.tiles[tile_ind].corners[n_corner_loc]
                    corner_included = False
                    for z in range(corner.neighbours_placed):
                        if n_corner == corner.corner_neighbours[z][0]:
                            corner_included = True
                    if corner_included == False:
                        edge.corner_1 = corner
                        edge.corner_2 = n_corner
                        corner.insert_neighbour(n_corner, edge)
                corner.insert_adjacent_tile(self.tiles[tile_ind])

        for i, harbour in enumerate(self.harbours):
            h_info = HARBOUR_CORNER_AND_EDGES[i]
            tile = self.tiles[h_info[0]]
            corner_1 = tile.corners[h_info[1]]
            corner_2 = tile.corners[h_info[2]]
            edge = tile.edges[h_info[3]]

            corner_1.harbour = harbour
            corner_2.harbour = harbour
            edge.harbour = harbour

            harbour.corners.append(corner_1)
            harbour.corners.append(corner_2)
            harbour.edge = edge

    def build_adjacency_matrices(self):
        self.corner_adjacency_matrix = np.zeros((54, 54))
        self.corner_egde_identification_map = np.zeros((54, 54))
        for corner in self.corners:
            for n_corner in corner.corner_neighbours:
                if n_corner[0] is not None:
                    self.corner_adjacency_matrix[corner.id, n_corner[0].id] = 1.0
                    self.corner_egde_identification_map[corner.id, n_corner[0].id] = n_corner[1].id

    def insert_settlement(self, player, corner, initial_placement=False):
        if corner.can_place_settlement(player.id, initial_placement=initial_placement):
            building = Building(BuildingType.Settlement, player.id, corner)
            corner.insert_building(building)
            if corner.harbour is not None:
                player.harbours[corner.harbour.resource] = corner.harbour
            return building
        else:
            raise ValueError("Cannot place settlement here.")

    def insert_city(self, player, corner):
        if corner.building is not None and \
                (corner.building.type == BuildingType.Settlement and corner.building.owner == player):
            building = Building(BuildingType.City, player, corner)
            corner.insert_building(building)
            return building
        else:
            raise ValueError("Cannot place city here!")

    def insert_road(self, player, edge):
        if edge.can_place_road(player):
            edge.insert_road(player)
        else:
            raise ValueError("Cannot place road here!")

    def get_available_settlement_locations(self, player, initial_round=False):
        available_locations = np.zeros((len(self.corners),), dtype=np.int)
        if initial_round == False:
            if player.resources[Resource.Wood] > 0 and player.resources[Resource.Sheep] > 0 and \
                player.resources[Resource.Brick] > 0 and player.resources[Resource.Wheat] > 0:
                pass
            else:
                return available_locations
        for i, corner in enumerate(self.corners):
            if corner.can_place_settlement(player.id, initial_placement=initial_round):
                available_locations[i] = 1
        return available_locations

    def get_available_city_locations(self, player):
        available_locations = np.zeros((len(self.corners),), dtype=np.int)
        if player.resources[Resource.Ore] >= 3 and player.resources[Resource.Wheat] >= 2:
            pass
        else:
            return available_locations
        for i, corner in enumerate(self.corners):
            if corner.building is not None and (corner.building.type == BuildingType.Settlement \
                and corner.building.owner == player.id):
                available_locations[i] = 1
        return available_locations

    def get_available_road_locations(self, player, initial_round=False):
        available_locations = np.zeros((len(self.edges),), dtype=np.int)
        if initial_round == False:
            if player.resources[Resource.Wood] > 0 and player.resources[Resource.Brick] > 0:
                pass
            else:
                return available_locations
        for i, edge in enumerate(self.edges):
            if edge.can_place_road(player.id):
                available_locations[i] = 1
        return available_locations

    def move_robber(self, tile):
        self.robber_tile.contains_robber = False
        tile.contains_robber = True
        self.robber_tile = tile

class Player(object):
    def __init__(self, id: PlayerId):
        self.id = id

    def reset(self, player_order):
        self.player_order = player_order
        self.player_lookup = {}
        self.inverse_player_lookup = {}
        for i in range(len(player_order)):
            if player_order[i] == self.id:
                p_ind = i
        for i, label in enumerate(["next", "next_next", "next_next_next"]):
            ind = (p_ind + 1 + i) % 4
            self.player_lookup[self.player_order[ind]] = label
            self.inverse_player_lookup[label] = self.player_order[ind]

        self.buildings = {}
        self.roads = []
        self.resources = {
            Resource.Brick: 0,
            Resource.Wood: 0,
            Resource.Wheat: 0,
            Resource.Ore: 0,
            Resource.Sheep: 0
        }
        self.visible_resources = {
            Resource.Brick: self.resources[Resource.Brick],
            Resource.Wood: self.resources[Resource.Wood],
            Resource.Wheat: self.resources[Resource.Wheat],
            Resource.Sheep: self.resources[Resource.Sheep],
            Resource.Ore: self.resources[Resource.Ore]
        }
        self.opponent_max_res = {
            "next": copy.deepcopy(self.visible_resources),
            "next_next": copy.deepcopy(self.visible_resources),
            "next_next_next": copy.deepcopy(self.visible_resources)
        }
        self.opponent_min_res = copy.deepcopy(self.opponent_max_res)
        self.harbours = {}
        self.longest_road = 0
        self.hidden_cards = []
        self.visible_cards = []
        self.victory_points = 0

class Game(object):
    def __init__(self, board_config={}, interactive=False, debug_mode=False, policies=None):
        self.board = Board(**board_config)
        self.players = {
            PlayerId.Blue: Player(PlayerId.Blue),
            PlayerId.Red: Player(PlayerId.Red),
            PlayerId.Orange: Player(PlayerId.Orange),
            PlayerId.White: Player(PlayerId.White)
        }
        self.reset()
        self.interactive = interactive
        self.debug_mode = debug_mode
        self.policies = policies

    def reset(self):
        self.board.reset()
        self.player_order = [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]
        np.random.shuffle(self.player_order)
        for player_id in self.players:
            self.players[player_id].reset(self.player_order)

        self.players_go = self.player_order[0]
        self.player_order_id = 0
        self.resource_bank = {
            Resource.Sheep: 19,
            Resource.Wheat: 19,
            Resource.Brick: 19,
            Resource.Ore: 19,
            Resource.Wood: 19
        }
        self.building_bank = {
            "settlements": {
                PlayerId.Blue: 5,
                PlayerId.White: 5,
                PlayerId.Orange: 5,
                PlayerId.Red: 5
            },
            "cities": {
                PlayerId.Blue: 4,
                PlayerId.White: 4,
                PlayerId.Orange: 4,
                PlayerId.Red: 4
            }
        }
        self.road_bank = {
            PlayerId.Blue: 15,
            PlayerId.Red: 15,
            PlayerId.Orange: 15,
            PlayerId.White: 15
        }
        self.development_cards = [DevelopmentCard.Knight] * 14 + [DevelopmentCard.VictoryPoint] * 5 + \
            [DevelopmentCard.YearOfPlenty] * 2 + [DevelopmentCard.RoadBuilding] * 2 + [DevelopmentCard.Monopoly] * 2
        np.random.shuffle(self.development_cards)
        self.development_cards_pile = deque(self.development_cards)
        self.longest_road = None
        self.largest_army = None

        self.max_trade_resources = 4

        self.initial_placement_phase = True
        self.initial_settlements_placed = {
            PlayerId.Blue: 0,
            PlayerId.Red: 0,
            PlayerId.Orange: 0,
            PlayerId.White: 0
        }
        self.initial_roads_placed = {
            PlayerId.Blue: 0,
            PlayerId.Red: 0,
            PlayerId.Orange: 0,
            PlayerId.White: 0
        }
        self.initial_second_settlement_corners = {
            PlayerId.Blue: None,
            PlayerId.Red: None,
            PlayerId.Orange: None,
            PlayerId.White: None
        }
        self.dice_rolled_this_turn = False
        self.played_development_card_this_turn = False
        self.must_use_development_card_ability = False
        self.must_respond_to_trade = False
        self.proposed_trade = None
        self.road_building_active = [False, 0]  # active, num roads placed
        self.can_move_robber = False
        self.just_moved_robber = False
        self.players_need_to_discard = False
        self.players_to_discard = []

        self.die_1 = None
        self.die_2 = None
        self.trades_proposed_this_turn = 0
        self.actions_this_turn = 0
        self.turn = 0
        self.development_cards_bought_this_turn = []
        self.longest_road = None
        self.largest_army = None
        self.current_longest_path = defaultdict(lambda: 0)
        self.current_army_size = defaultdict(lambda: 0)
        self.colours = {
            PlayerId.White: (255, 255, 255),
            PlayerId.Red: (255, 0, 0),
            PlayerId.Blue: (0, 0, 255),
            PlayerId.Orange: (255, 153, 51)
        }
        self.resource_text = {
            Resource.Wood: "wood",
            Resource.Brick: "brick",
            Resource.Wheat: "wheat",
            Resource.Sheep: "sheep",
            Resource.Ore: "ore"
        }

    def roll_dice(self):
        self.die_1 = np.random.randint(1, 7)
        self.die_2 = np.random.randint(1, 7)

        roll_value = int(self.die_1 + self.die_2)

        if roll_value == 7:
            for p_id in self.player_order:
                if sum(self.players[p_id].resources.values()) > 7:
                    self.players_need_to_discard = True
                    self.players_to_discard.append(p_id)
            return roll_value

        tiles_hit = self.board.value_to_tiles[roll_value]

        resources_allocated = {
            resource: defaultdict(lambda: 0) for resource in [Resource.Wood, Resource.Ore, Resource.Brick, Resource.Wheat, Resource.Sheep]
        }

        for tile in tiles_hit:
            if tile.contains_robber:
                continue
            for corner_key, corner in tile.corners.items():
                if corner.building is not None:
                    if corner.building.type == BuildingType.Settlement:
                        increment = 1
                    elif corner.building.type == BuildingType.City:
                        increment = 2
                    resources_allocated[tile.resource][corner.building.owner] += increment
                    resources_allocated[tile.resource]["total"] += increment

        for resource in resources_allocated.keys():
            if resources_allocated[resource]["total"] <= self.resource_bank[resource]:
                for player in [PlayerId.Blue, PlayerId.Orange, PlayerId.White, PlayerId.Red]:
                    self.players[player].resources[resource] += resources_allocated[resource][player]
                    self.resource_bank[resource] -= resources_allocated[resource][player]

        return roll_value

    def can_buy_settlement(self, player):
        if self.initial_placement_phase:
            return True
        if self.building_bank["settlements"][player.id] > 0:
            if player.resources[Resource.Wheat] > 0 and player.resources[Resource.Wood] > 0 and \
                    player.resources[Resource.Brick] > 0 and player.resources[Resource.Sheep] > 0:
                return True
        return False

    def build_settlement(self, player, corner):
        if self.initial_placement_phase == False:
            player.resources[Resource.Wheat] -= 1
            player.resources[Resource.Sheep] -= 1
            player.resources[Resource.Wood] -= 1
            player.resources[Resource.Brick] -= 1
            self.resource_bank[Resource.Wheat] += 1
            self.resource_bank[Resource.Sheep] += 1
            self.resource_bank[Resource.Wood] += 1
            self.resource_bank[Resource.Brick] += 1
        self.board.insert_settlement(player, corner, initial_placement=self.initial_placement_phase)
        self.building_bank["settlements"][player.id] -= 1
        player.buildings[corner.id] = BuildingType.Settlement
        player.victory_points += 1

    def can_buy_road(self, player):
        if self.initial_placement_phase:
            return True
        if player.resources[Resource.Wood] > 0 and player.resources[Resource.Brick] > 0:
            return True
        else:
            return False

    def build_road(self, player, edge, road_building=False):
        if self.initial_placement_phase == False:
            if road_building == False:
                player.resources[Resource.Wood] -= 1
                player.resources[Resource.Brick] -= 1
                self.resource_bank[Resource.Wood] += 1
                self.resource_bank[Resource.Brick] += 1
        self.board.insert_road(player.id, edge)
        player.roads.append(edge.id)

    def can_buy_city(self, player):
        if self.building_bank["cities"][player.id] > 0:
            if player.resources[Resource.Wheat] > 1 and player.resources[Resource.Ore] > 2:
                return True
        return False

    def build_city(self, player, corner):
        player.resources[Resource.Wheat] -= 2
        player.resources[Resource.Ore] -= 3
        self.resource_bank[Resource.Wheat] += 2
        self.resource_bank[Resource.Ore] += 3
        self.board.insert_city(player.id, corner)
        player.victory_points += 1
        self.building_bank["cities"][player.id] -= 1
        self.building_bank["settlements"][player.id] += 1
        player.buildings[corner.id] = BuildingType.City

    def update_players_go(self, left=False):
        if left:
            self.player_order_id -= 1
            if self.player_order_id < 0:
                self.player_order_id = 3
        else:
            self.player_order_id += 1
            if self.player_order_id > 3:
                self.player_order_id = 0
        self.players_go = self.player_order[self.player_order_id]

    def validate_action(self, action):
        player = self.players[self.players_go]
        if action["type"] == ActionTypes.PlaceSettlement:
            if self.can_buy_settlement(player):
                corner = action["corner"]
                if self.board.corners[corner].can_place_settlement(player.id, initial_placement=self.initial_placement_phase):
                    if self.initial_placement_phase:
                        if self.initial_settlements_placed[self.players_go] == 0 or \
                                (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 1):
                            return True, None
                        return False, "You cannot place a settlement here!"
                    return True, None
            return False, "You cannot afford a settlement!"
        elif action["type"] == ActionTypes.PlaceRoad:
            if self.can_buy_road(player):
                edge = action["edge"]
                if self.board.edges[edge].can_place_road(player.id):
                    if self.initial_placement_phase:
                        if (self.initial_settlements_placed[self.players_go] == 1 and self.initial_roads_placed[self.players_go] == 0):
                            return True, None
                        elif (self.initial_settlements_placed[self.players_go] == 2 and self.initial_roads_placed[self.players_go] == 1):
                            if self.board.edges[edge].can_place_road(player.id, after_second_settlement=True,
                                                                    second_settlement=self.initial_second_settlement_corners[player.id]):
                                return True, None
                            else:
                                return False, "Must place second road next to second settlement."
                        return False, "You cannot place a road here!"
                    return True, None
                return False, "You cannot place a road here!"
            else:
                return False, "You cannot afford a road!"
        elif action["type"] == ActionTypes.UpgradeToCity:
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.can_buy_city(player):
                corner = self.board.corners[action["corner"]]
                if corner.building is not None and corner.building.type == BuildingType.Settlement:
                    if corner.building.owner == player.id:
                        return True, None
                else:
                    return False, "This cannot be upgraded to a city!"
            return False, "You cannot afford to upgrade to a city!"
        elif action["type"] == ActionTypes.RollDice:
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn:
                return False, "You have already rolled the dice this turn!"
            return True, None
        elif action["type"] == ActionTypes.EndTurn:
            if self.initial_placement_phase:
                return False, "Still in initial placement phase!"
            elif self.dice_rolled_this_turn == False:
                return False, "You cannot end your turn before rolling the dice!"
            return True, None

    def apply_action(self, action):
        player = self.players[self.players_go]
        if action["type"] == ActionTypes.PlaceSettlement:
            corner = self.board.corners[action["corner"]]
            self.build_settlement(player, corner)
            if self.initial_placement_phase:
                self.initial_settlements_placed[player.id] += 1
                if self.initial_settlements_placed[player.id] == 2:
                    tile_res = defaultdict(lambda: 0)
                    for tile in corner.adjacent_tiles:
                        if tile is None or tile.resource == Resource.Empty:
                            continue
                        player.resources[tile.resource] += 1
                        self.resource_bank[tile.resource] -= 1
                    self.initial_second_settlement_corners[player.id] = action["corner"]
        elif action["type"] == ActionTypes.PlaceRoad:
            edge = self.board.edges[action["edge"]]
            self.build_road(player, edge)
            if self.initial_placement_phase:
                self.initial_roads_placed[player.id] += 1
                first_settlements_placed = 0
                second_settlements_placed = 0
                for player_id, count in self.initial_settlements_placed.items():
                    if count == 1:
                        first_settlements_placed += 1
                    elif count == 2:
                        first_settlements_placed += 1
                        second_settlements_placed += 1
                if first_settlements_placed < 4:
                    self.update_players_go()
                elif first_settlements_placed == 4 and second_settlements_placed == 0:
                    pass
                elif first_settlements_placed == 4 and second_settlements_placed < 4:
                    self.update_players_go(left=True)
                else:
                    self.initial_placement_phase = False
        elif action["type"] == ActionTypes.UpgradeToCity:
            self.build_city(player, self.board.corners[action["corner"]])
        elif action["type"] == ActionTypes.RollDice:
            roll_value = self.roll_dice()
            self.dice_rolled_this_turn = True
            if roll_value == 7:
                self.can_move_robber = True
        elif action["type"] == ActionTypes.EndTurn:
            self.can_move_robber = False
            self.dice_rolled_this_turn = False
            self.update_players_go()
            self.turn += 1
            self.development_cards_bought_this_turn = []
            self.trades_proposed_this_turn = 0
            self.actions_this_turn = 0

    def update_longest_road(self, player_id):
        max_path_len = self.get_longest_path(player_id)

        self.current_longest_path[player_id] = max_path_len

        longest_road_update = {
            "player": player_id,
            "count": max_path_len
        }

        if self.longest_road is None:
            if max_path_len >= 5:
                self.longest_road = longest_road_update
                self.players[player_id].victory_points += 2
            return

        if self.longest_road["player"] == player_id:
            if self.longest_road["count"] > max_path_len:
                max_p_len = max_path_len
                player = player_id
                tied_longest_road = False
                for other_pid in [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]:
                    if other_pid == player_id:
                        continue
                    p_len = self.get_longest_path(other_pid)
                    if p_len == max_p_len:
                        tied_longest_road = True
                    elif p_len > max_p_len:
                        max_p_len = p_len
                        tied_longest_road = False
                        player = other_pid
                if max_p_len >= 5:
                    if tied_longest_road:
                        if player == player_id:
                            self.longest_road = longest_road_update
                        else:
                            self.longest_road = None
                            self.players[player_id].victory_points -= 2
                    else:
                        self.longest_road = {
                            "player": player,
                            "count": max_p_len
                        }
                        self.players[player].victory_points += 2
                        self.players[player_id].victory_points -= 2
                else:
                    self.longest_road = None
                    self.players[player_id].victory_points -= 2
            else:
                self.longest_road = longest_road_update
        else:
            if max_path_len > self.longest_road["count"]:
                self.players[self.longest_road["player"]].victory_points -= 2
                self.players[player_id].victory_points += 2
                self.longest_road = longest_road_update

    def get_longest_path(self, player_id):
        player_edges = []
        for edge in self.board.edges:
            if edge.road is not None and edge.road == player_id:
                player_edges.append([edge.corner_1.id, edge.corner_2.id])

        G = defaultdict(list)
        for (s, t) in player_edges:
            if self.board.corners[s].building is not None and self.board.corners[s].building.owner != player_id:
                pass
            else:
                G[s].append(t)
            if self.board.corners[t].building is not None and self.board.corners[t].building.owner != player_id:
                pass
            else:
                G[t].append(s)

        all_paths = list(chain.from_iterable(DFS(G, n) for n in set(G)))
        max_path_len = max(len(p) - 1 for p in all_paths)
        return max_path_len

    def save_current_state(self):
        state = {}
        state["players_need_to_discard"] = self.players_need_to_discard
        state["players_to_discard"] = copy.copy(self.players_to_discard)
        state["tile_info"] = []
        for tile in self.board.tiles:
            state["tile_info"].append(
                (tile.terrain, tile.resource, tile.value, tile.likelihood, tile.contains_robber)
            )
        state["edge_occupancy"] = [edge.road for edge in self.board.edges]
        state["corner_buildings"] = []
        for corner in self.board.corners:
            if corner.building is not None:
                state["corner_buildings"].append((corner.building.type, corner.building.owner))
            else:
                state["corner_buildings"].append((None, None))
        state["harbour_order"] = [harbour.id for harbour in self.board.harbours]
        state["players"] = {}
        for player_key, player in self.players.items():
            state["players"][player_key] = {"id": player.id,
                "player_order": copy.copy(player.player_order),
                "player_lookup": copy.copy(player.player_lookup),
                "inverse_player_lookup": copy.copy(player.inverse_player_lookup),
                "buildings": copy.copy(player.buildings),
                "roads": copy.copy(player.roads),
                "resources": copy.copy(player.resources),
                "visible_resources": copy.copy(player.visible_resources),
                "opponent_max_res": copy.copy(player.opponent_max_res),
                "opponent_min_res": copy.copy(player.opponent_min_res),
                "harbour_info": [(key, val.id) for key, val in player.harbours.items()],
                "longest_road": player.longest_road,
                "hidden_cards": copy.copy(player.hidden_cards),
                "visible_cards": copy.copy(player.visible_cards),
                "victory_points": player.victory_points
            }
        state["players_go"] = self.players_go
        state["player_order"] = copy.deepcopy(self.player_order)
        state["player_order_id"] = self.player_order_id
        state["resource_bank"] = copy.copy(self.resource_bank)
        state["building_bank"] = copy.copy(self.building_bank)
        state["road_bank"] = copy.copy(self.road_bank)
        state["development_cards"] = copy.deepcopy(self.development_cards)
        state["development_card_pile"] = copy.deepcopy(self.development_cards_pile)
        state["largest_army"] = self.largest_army
        state["longest_road"] = self.longest_road
        state["initial_placement_phase"] = self.initial_placement_phase
        state["initial_settlements_placed"] = copy.copy(self.initial_settlements_placed)
        state["initial_roads_placed"] = copy.copy(self.initial_roads_placed)
        state["initial_second_settlement_corners"] = copy.copy(self.initial_second_settlement_corners)
        state["dice_rolled_this_turn"] = self.dice_rolled_this_turn
        state["played_development_card_this_turn"] = self.played_development_card_this_turn
        state["must_use_development_card_ability"] = self.must_use_development_card_ability
        state["must_respond_to_trade"] = self.must_respond_to_trade
        state["proposed_trade"] = copy.deepcopy(self.proposed_trade)
        state["road_building_active"] = self.road_building_active
        state["can_move_robber"] = self.can_move_robber
        state["just_moved_robber"] = self.just_moved_robber
        state["trades_proposed_this_turn"] = self.trades_proposed_this_turn
        state["actions_this_turn"] = self.actions_this_turn
        state["turn"] = self.turn
        state["development_cards_bought_this_turn"] = self.development_cards_bought_this_turn
        state["current_longest_path"] = copy.copy(self.current_longest_path)
        state["current_army_size"] = copy.copy(self.current_army_size)
        state["die_1"] = self.die_1
        state["die_2"] = self.die_2
        return state

    def restore_state(self, state):
        state = copy.deepcopy(state)
        self.players_to_discard = state["players_to_discard"]
        self.players_need_to_discard = state["players_need_to_discard"]
        self.board.value_to_tiles = {}
        for i, info in enumerate(state["tile_info"]):
            terrain, resource, value, likelihood, contains_robber = info[0], info[1], info[2], info[3], info[4]
            self.board.tiles[i].terrain = terrain
            self.board.tiles[i].resource = resource
            self.board.tiles[i].value = value
            self.board.tiles[i].likelihood = likelihood
            self.board.tiles[i].contains_robber = contains_robber
            if value != 7:
                if value in self.board.value_to_tiles:
                    self.board.value_to_tiles[value].append(self.board.tiles[i])
                else:
                    self.board.value_to_tiles[value] = [self.board.tiles[i]]
            if contains_robber:
                self.board.robber_tile = self.board.tiles[i]
        for i, road in enumerate(state["edge_occupancy"]):
            self.board.edges[i].road = road
        for i, entry in enumerate(state["corner_buildings"]):
            building, player = entry[0], entry[1]
            if building is not None:
                self.board.corners[i].building = Building(building, player, self.board.corners[i])
            else:
                self.board.corners[i].building = None
        self.board.harbours = copy.copy([self.board.HARBOURS_TO_PLACE[i] for i in state["harbour_order"]])
        for corner in self.board.corners:
            corner.harbour = None
        for edge in self.board.edges:
            edge.harbour = None
        for i, harbour in enumerate(self.board.harbours):
            h_info = HARBOUR_CORNER_AND_EDGES[i]
            tile = self.board.tiles[h_info[0]]
            corner_1 = tile.corners[h_info[1]]
            corner_2 = tile.corners[h_info[2]]
            edge = tile.edges[h_info[3]]
            corner_1.harbour = harbour
            corner_2.harbour = harbour
            edge.harbour = harbour
            harbour.corners.append(corner_1)
            harbour.corners.append(corner_2)
            harbour.edge = edge
        for key, player_state in state["players"].items():
            self.players[key].id = player_state["id"]
            self.players[key].player_order = player_state["player_order"]
            self.players[key].player_lookup = player_state["player_lookup"]
            self.players[key].inverse_player_lookup = player_state["inverse_player_lookup"]
            self.players[key].buildings = player_state["buildings"]
            self.players[key].roads = player_state["roads"]
            self.players[key].resources = player_state["resources"]
            self.players[key].visible_resources = player_state["visible_resources"]
            self.players[key].opponent_max_res = player_state["opponent_max_res"]
            self.players[key].opponent_min_res = player_state["opponent_min_res"]
            self.players[key].longest_road = player_state["longest_road"]
            self.players[key].hidden_cards = player_state["hidden_cards"]
            self.players[key].visible_cards = player_state["visible_cards"]
            self.players[key].victory_points = player_state["victory_points"]
            for info in player_state["harbour_info"]:
                key_res = info[0]; id = info[1]
                for harbour in self.board.harbours:
                    if id == harbour.id:
                        self.players[key].harbours[key_res] = harbour
        self.players_go = state["players_go"]
        self.player_order = state["player_order"]
        self.player_order_id = state["player_order_id"]
        self.resource_bank = state["resource_bank"]
        self.building_bank = state["building_bank"]
        self.road_bank = state["road_bank"]
        self.development_cards = state["development_cards"]
        self.development_cards_pile = state["development_card_pile"]
        self.largest_army = state["largest_army"]
        self.longest_road = state["longest_road"]
        self.initial_placement_phase = state["initial_placement_phase"]
        self.initial_settlements_placed = state["initial_settlements_placed"]
        self.initial_roads_placed = state["initial_roads_placed"]
        self.initial_second_settlement_corners = state["initial_second_settlement_corners"]
        self.dice_rolled_this_turn = state["dice_rolled_this_turn"]
        self.played_development_card_this_turn = state["played_development_card_this_turn"]
        self.must_use_development_card_ability = state["must_use_development_card_ability"]
        self.must_respond_to_trade = state["must_respond_to_trade"]
        self.proposed_trade = state["proposed_trade"]
        self.road_building_active = state["road_building_active"]
        self.can_move_robber = state["can_move_robber"]
        self.just_moved_robber = state["just_moved_robber"]
        self.trades_proposed_this_turn = state["trades_proposed_this_turn"]
        self.actions_this_turn = state["actions_this_turn"]
        self.turn = state["turn"]
        self.development_cards_bought_this_turn = state["development_cards_bought_this_turn"]
        self.current_longest_path = state["current_longest_path"]
        self.current_army_size = state["current_army_size"]
        self.die_1 = state["die_1"]
        self.die_2 = state["die_2"]

# To play the game
game = Game()
