import numpy as np
import pandas as pd
import typing as t

from waymo_agent.data_classes import EnvConfig, RequestDF, ObservationDict, VehicleDF, SupplyDemandDF
from waymo_agent.graph_env.ENV import RideShareEnv


class PricingAgent:
    base_ratio: float
    sd_slope: float

    def __init__(self, config: EnvConfig):
        self.config = config
        self.base_ratio = 1.0
        self.sd_slope = 1.0

    def price(self, obs: ObservationDict):
        """Heuristic pricing based on supply-demand ratio."""
        sd_ratio = SupplyDemandDF.from_obs_numpy(obs["supply_demand_ratio"])
        sd_current = sd_ratio.sd_current[0]
        price_multiplier: pd.Series = self.base_ratio + self.sd_slope * (1 - sd_current)

        req = t.cast(RequestDF, RequestDF.from_obs_numpy(obs["pending_requests"]))
        prices = req["est_cost"] * price_multiplier
        return prices.to_numpy().clip(min=0.0).astype(np.float32)


class DispatchAgent:
    distance_threshold: float

    def __init__(self, config: EnvConfig):
        self.config = config
        self.distance_threshold = 0.5  # normalized distance

    def dispatch(self, obs: ObservationDict):
        """Heuristic dispatch based on distance threshold."""
        vehicles = t.cast(VehicleDF, VehicleDF.from_obs_numpy(obs["vehicles"]))

        # Recreate vehicle IDs to match original indexing
        idx_offset = self.config.no_action_id + 1
        vehicles["vehicle_id"] = np.arange(len(vehicles)) + idx_offset
        veh_available = obs["dispatch_mask"]

        requests = RequestDF(RequestDF.from_obs_numpy(obs["pending_requests"]))
        f_req_dispatchable = requests.f_need_dispatch
        print(f"Dispatchable requests: {np.sum(f_req_dispatchable)} out of {len(requests)}")

        f_taken = np.zeros(len(vehicles), dtype=bool)
        dispatch_actions = np.full(requests.shape[0], self.config.no_action_id, dtype=int)

        for idx_, req in requests.iterrows():
            req_idx = int(idx_)  # type: ignore
            if not f_req_dispatchable[req_idx]:
                continue  # skip requests that do not need pricing decision

            veh_avail = vehicles[~f_taken & veh_available]
            # print(veh_avail)
            # print("\n\n")
            x_norm = req.pickup_x_norm
            y_norm = req.pickup_y_norm
            distances = np.array(np.sqrt((veh_avail.loc_x_norm - x_norm) ** 2 + (veh_avail.loc_y_norm - y_norm) ** 2))
            # print(distances)
            # print(type(distances), distances.shape)
            # print("\n\n")

            min_dist_idx = np.argmin(distances)
            # print(f"min_dist_idx: {min_dist_idx}")
            min_dist = distances[min_dist_idx]
            veh_idx = veh_avail.vehicle_id.to_numpy()[min_dist_idx]
            print(f"Request {req_idx} closest vehicle {veh_idx} at distance {min_dist} min_dist_idx {min_dist_idx}")

            if min_dist <= self.distance_threshold:
                print(f"Assigning vehicle {veh_idx} to request {req_idx}")
                dispatch_actions[req_idx] = veh_idx
                f_taken[veh_idx] = True

        return dispatch_actions


class RepositionAgent:
    def __init__(self, env: RideShareEnv):
        self.env = env

    def reposition(self, obs: ObservationDict):
        """Heuristic repositioning to center of map."""
        vehicles = t.cast(VehicleDF, VehicleDF.from_obs_numpy(obs["vehicles"]))

        num_samples = min(len(vehicles), len(self.env.node_df))
        out = self.env.node_df.sample(num_samples, weights="degree_centrality", replace=False)

        return out[["x_norm", "y_norm"]].to_numpy(np.float32)
