"""Contains the highway network class."""

from flow.networks.base import Network
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
import numpy as np

ADDITIONAL_NET_PARAMS = {
    "length": 100,
    "num_vehicles":6,
    "initial_gaps": [20, 20, 20, 20, 20],
    "speed_limit":1
}


class PlatoonHighwayNetwork(Network):
    """Highway network class.

    This network consists of `num_edges` different straight highway sections
    with a total characteristic length and number of lanes.

    Requires from net_params:

    * **length** : length of the highway
    * **lanes** : number of lanes in the highway
    * **speed_limit** : max speed limit of the highway
    * **num_edges** : number of edges to divide the highway into
    * **use_ghost_edge** : whether to include a ghost edge. This edge is
      provided a different speed limit.
    * **ghost_speed_limit** : speed limit for the ghost edge
    * **boundary_cell_length** : length of the downstream ghost edge with the
      reduced speed limit

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.networks import HighwayNetwork
    >>>
    >>> network = HighwayNetwork(
    >>>     name='highway',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'length': 230,
    >>>             'lanes': 1,
    >>>             'speed_limit': 30,
    >>>             'num_edges': 1
    >>>         },
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Initialize a highway network."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]

        nodes = []
        
        nodes.append({
            "id": "edge_{}".format(0),
            "x": 0,
            "y": 0
        })

        nodes.append({
                "id": "edge_{}".format(1),
                "x": length,
                "y": 0
            })

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        length = net_params.additional_params["length"]

        edges = [{
                "id": "highway_{}".format(0),
                "type": "highwayType",
                "from": "edge_{}".format(0),
                "to": "edge_{}".format(1),
                "length": length
            }]

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        speed_limit = net_params.additional_params["speed_limit"]

        types = [{
            "id": "highwayType",
            "speed": speed_limit
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {"highway_{}".format(0): ["highway_{}".format(0)]}
            
        return rts

    def specify_edge_starts(self):
        """See parent class."""
        length = self.net_params.additional_params["length"]

        # Add the main edges.
        edge_starts = [("highway_{}".format(0), 0)]

        return edge_starts

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):
        """Generate a user defined set of starting positions.

        This method is just used for testing.
        """
        initial_gaps = net_params.additional_params["initial_gaps"]


        start_positions, start_lanes = [], []
        position = 40 #########quick fix

        for i in range(num_vehicles - 1):
            start_lanes.append(0)
            start_positions.append(("highway_0", position))
            position += initial_gaps[i]
        
        start_lanes.append(0)
        start_positions.append(("highway_0", position))


        return start_positions, start_lanes
