from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from waymo_agent.data_classes import ActionDict, ObservationDict, RequestDF, price_acceptance_probability
from waymo_agent.data_classes.active_rides import ActiveRideDF
from waymo_agent.data_classes.enriched_df_base import validate_typed_df_keys
from waymo_agent.data_classes.requests import RequestStatusEnum
from waymo_agent.data_classes.vehicles import VehicleDF, VehicleStatusEnum
from waymo_agent.graph_env.cost_reward import compute_amortized_reward
from waymo_agent.osmnx.euclidean_L2_embed import interpolate_position_on_edge
from waymo_agent.osmnx.traverse_graph import step_along_route
from waymo_agent.simulation.generate_obs_state import get_sd_ratio, init_active_ride_df
from waymo_agent.graph_env.df_utils import masked_assign

from .interface import GymEnvInterface


class RewardMixin(GymEnvInterface):
    """
    Mixin responsible for calculating rewards based on environment transitions.

    This includes:
        - Amortized rewards for completed rides + pricing decisions.
        - Penalties for rejected/cancelled requests.
        - Costs associated with vehicle operations (e.g., repositioning, idling).
    """


class TransitionMixin(RewardMixin):
    """
    Mixin responsible for handling environment transitions.

    This includes:
    - Applying pricing and dispatch actions.
    - Progressing vehicle tasks (rides, repositioning, charging).
    - Managing request lifecycles (spawning, waiting, cancellations).
    - Advancing the simulation clock.

    NOTE there's a mix of deterministic and stochastic transitions. They will be explicitly noted.
    """

    _rewards: dict[str, float] = {}
    _req_interim: RequestDF
    _veh_interim: VehicleDF
    _active_rides_interim: ActiveRideDF

    # DEBUGGING
    _discard_ride_ids: pd.Series
    _veh_ride_sanity_df: pd.DataFrame  # Must ALL be True
    _rides_debug: pd.DataFrame  # detailed ride transition debug info

    _request_sanity_df: pd.DataFrame  # Must ALL be True
    _request_debug: pd.DataFrame  # detailed request transition debug info

    _dispatch_debug: pd.DataFrame  # detailed dispatch transition debug info

    def _advance_clock(self):
        """
        Advance the simulation clock by one time step. Add to breadcrumbs.

        Deterministic
        """
        self.current_step += 1
        self.time_dt += self.config.time_step_delta

        self.calc_time_normed()
        self.bc_row.update({"step": self.current_step, "timestamp": self.time_dt})

    def get_observation(self, action: ActionDict) -> tuple[ObservationDict, float]:
        """
        Construct the observation dictionary from the current environment state. The returned observaion
        is post-transition, i.e., after all updates have been applied for the current step.

        Deterministic
        """
        self.observation_prev = self.observation_curr
        del self.observation_curr  # to avoid accidental usage
        self._rewards = {}
        self.bc_row = {}
        self._advance_clock()
        self.reset_debug_and_interim()

        requests, matches_orig = self._update_requests_from_action(action)
        new_rides = self._new_rides(requests, matches_orig)
        veh_interim, active_rides_updated, self._discard_ride_ids = self._update_vehicles(action, new_rides)
        self._veh_interim = VehicleDF(veh_interim)
        self._active_rides_interim = ActiveRideDF(active_rides_updated)

        self._req_interim = RequestDF(requests)
        self._cycle_requests(self._discard_ride_ids)
        validate_typed_df_keys(self._req_interim, RequestDF)

        self._assert_state_consistency(action)
        sd_ratio_post_step = get_sd_ratio(self.config, self._req_interim, self._veh_interim)
        observation = ObservationDict(
            {
                "globals": self.MetaState,
                "supply_demand_ratio": sd_ratio_post_step,
                "vehicles": self._veh_interim,
                "pending_requests": self._req_interim,
                "active_rides": self._active_rides_interim,
                "dispatch_mask": self._veh_interim.f_idle,
                "pricing_mask": self._req_interim.f_awaiting_price.astype(np.float32),
            }
        )
        self.observation_curr = observation

        rewards = sum(self._rewards.values())
        self.bc_row.update(
            {"rewards": rewards, "supply_demand_ratio": sd_ratio_post_step[0], "error_msg": self.error_msg}
        )

        self.append_breadcrumbs()

        return observation, rewards

    def _update_requests_from_action(self, action: ActionDict):
        """
        Update the status of all pending requests, including wait times and cancellations.
            - when price is offered, customer accepts or rejects
            - if price accepted in previous step, vehicle is dispatched (if possible)
            - if vehicle arrives at pickup, ride starts
            - if wait time exceeds threshold, request is cancelled

        Deterministic + Stochastic (acceptance)
        """
        # 1- copy over existing requests
        requests = RequestDF(self.observation_prev["pending_requests"].copy(deep=True))

        # 2 - price acceptance
        sd_ratio_avg = np.mean(self.breadcrumbs.tail(5)["supply_demand_ratio"], axis=0)
        if np.isnan(sd_ratio_avg).any():
            raise ValueError("Supply-demand ratio contains NaN values; cannot compute price acceptance.")

        # cust_df = self.cust_df[self.cust_df["cust_id"].isin(requests["cust_id"])]
        acceptance_prob, z = price_acceptance_probability(
            self.cust_df, requests, action["prices"], sd_ratio_avg, self.config
        )
        await_mask = requests["status"] == RequestStatusEnum.AWAITING_PRICE
        accept_mask = acceptance_prob >= np.random.uniform(0.0, 1.0, size=acceptance_prob.shape)
        new_status = np.select(
            [(await_mask & accept_mask), (await_mask & ~accept_mask)],
            [RequestStatusEnum.ACCEPTED, RequestStatusEnum.REJECTED],
            requests["status"],
        )
        self._accept_mask = accept_mask
        self._await_mask = await_mask

        requests["price"] = np.where(await_mask, action["prices"], requests["price"])

        # 3 - dispatches
        out = self._match_dispatches(action, requests)
        matches_orig = out.copy(deep=True)
        match_vehicle_id = out.pop("vehicle_id")
        r_dispatch = out

        new_status = np.select(  # update request statuses based on dispatch matches
            [is_newly_assigned := (requests.f_need_dispatch & (match_vehicle_id != self.config.no_action_id))],
            [RequestStatusEnum.ASSIGNED],
            new_status,
        )
        matches_orig["is_newly_assigned"] = is_newly_assigned
        self._dispatch_debug = matches_orig
        self._rewards.update(r_dispatch.sum().to_dict())

        # 5 - apply rewards from pricing and dispatching
        r_reject = (requests["status"] == RequestStatusEnum.REJECTED).sum() * self.config.penalty_rejected
        r_expire = (
            (requests["status"] == RequestStatusEnum.CANCEL_EXCEED_WAIT_TIME) & requests.f_valid
        ).sum() * self.config.penalty_expire
        self._rewards.update({"penalty_rejected": r_reject, "penalty_expire": r_expire})

        # 4- update wait times & cancellations
        requests["wait_time"] = self.time_dt - pd.to_datetime(requests["request_dt"])  # update wait times
        cancel_mask = requests["wait_time"] > requests["max_wait_time"]
        new_status = np.select(
            [cancel_mask],
            [RequestStatusEnum.CANCEL_EXCEED_WAIT_TIME],
            new_status,
        )
        requests["status"] = new_status

        return requests, matches_orig

    def _match_dispatches(self, action: ActionDict, requests: RequestDF):
        """
        Match dispatched vehicles to requests based on the action dict.
            - validate that vehicles are available for assignment
            - apply penalties for invalid assignments

        NOTE: requests are processed sequentially in the order they appear in the requests dataframe.
        This might not be ideal if multiple requests target the same vehicle.
            - Assuming that dispatches are ranking by wait time, you might prefer the longest-waiting
                request, instead of the most recent one

        NOTE: we currently do not allow dispatching of active vehicle to request.
            - This could be a future extension where vehicles can be assigned a new ride while en route

        Deterministic
        """
        RWD = self.config.shaped_reward_config
        rewards_df = RWD.reward_df_empty(len=len(requests))

        dispatches = action["dispatch"]
        vehicles = self.observation_prev["vehicles"]

        vehicle_id = np.full(shape=len(dispatches), fill_value=self.config.no_action_id, dtype=np.int64)

        for req_idx, veh_idx in enumerate(dispatches):
            should_skip = not requests.f_need_dispatch[req_idx]
            if veh_idx == self.config.no_action_id or should_skip:
                continue  # No vehicle assigned

            # is vehicle available for assignment (not busy)?
            avail = vehicles.f_idle[veh_idx]
            rewards_df.loc[req_idx, "penalty_assign_to_unavailable_vehicle"] = (
                1 - (2 * avail)
            ) * RWD.penalty_assign_to_unavailable_vehicle

            # has vehicle already been assigned to another request this step?
            multiple_assign = 1 if veh_idx in vehicle_id else -1
            rewards_df.loc[req_idx, "penalty_multiple_dispatch_assignment"] = (
                multiple_assign * RWD.penalty_multiple_dispatch_assignment
            )

            if avail and multiple_assign == -1:
                # valid assignment
                vehicle_id[req_idx] = veh_idx

        out = rewards_df.copy(deep=True)
        out["vehicle_id"] = vehicle_id
        out = out[["vehicle_id"] + [col for col in out.columns if col != "vehicle_id"]]
        self._dispatch_debug = out
        return out

    def _new_rides(self, requests: RequestDF, matches: pd.DataFrame):
        """
        Construct new ActiveRide rows for requests that were newly assigned
        a vehicle in this step.

        NOTE:
            `matches` has one row per *pre-existing* request (before any new
            requests are spawned later in the step). We therefore align the
            boolean mask from `matches` to the first `len(matches)` rows of
            `requests` and use a NumPy boolean mask to avoid index
            re-alignment issues.
        """
        # Boolean mask over the original (pre-spawn) requests
        f_new = matches["is_newly_assigned"].to_numpy()

        # Align requests with matches: matches only refers to the original
        # pending requests, not any future spawned ones.
        base_requests = requests.iloc[: len(matches)].reset_index(drop=True)

        # Vehicle IDs corresponding to newly assigned requests
        veh_ids = matches.loc[f_new, "vehicle_id"].to_numpy()

        # Build ActiveRideDF only for the newly assigned subset
        new_rides = ActiveRideDF.from_requests(base_requests[f_new], veh_ids)
        return new_rides

    def _update_vehicles(self, action: ActionDict, new_rides: ActiveRideDF):
        """
        1) Assign cars to rides (dispatched vehicles)
        2) Move assigned vehicles (pickup, dropoffs)
            - distribute rewards
        3) For completed rides: update requests / active rides dataframes
        4) Reposition idle vehicles based on action dict

        Update vehicle states based on their current tasks and positions.
            - if vehicle arrives at charging station, transition to CHARGING (not implemented)
            - if vehicle battery full, transition to IDLE
            - if vehicle arrives at pickup, transition to WITH_PASSENGER
            - if vehicle arrives at dropoff, transition to IDLE
            - if vehicle arrives at reposition target, transition to IDLE
            - if vehicle dispatched to pickup, transition to TO_PICKUP

        Deterministic
        """
        # 1 - Merge new rides into active rides
        prev_rides = self.observation_prev["active_rides"].copy(deep=True)
        active_rides = ActiveRideDF(prev_rides.copy(deep=True))

        f_new_rides = active_rides.vehicle_id.isin(new_rides["vehicle_id"]).to_numpy(dtype=bool)
        self._new_rides_df = new_rides
        self.f_new_rides = f_new_rides
        masked_assign(active_rides, f_new_rides, new_rides)  # ASSIGN NEW RIDES TO ACTIVE RIDES DF
        # print(active_rides)
        assert active_rides.vehicle_id.equals(prev_rides.vehicle_id)

        # 2 - This is a cheat (TODO fix) - just move newly_assigned vehicle to start of ride
        veh_old = VehicleDF(self.observation_prev["vehicles"].copy(deep=True))
        vehicles = VehicleDF(self.observation_prev["vehicles"].copy(deep=True))
        vehicles["status"] = np.where(f_new_rides, VehicleStatusEnum.WITH_PASSENGER, vehicles["status"])
        vehicles["ride_id"] = np.where(f_new_rides, active_rides["ride_id"], vehicles["ride_id"])
        vehicles["loc_x_norm"] = np.where(f_new_rides, active_rides["pickup_x_norm"], vehicles["loc_x_norm"])
        vehicles["loc_y_norm"] = np.where(f_new_rides, active_rides["pickup_y_norm"], vehicles["loc_y_norm"])
        assert vehicles["loc_x_norm"].isnull().sum() == 0, "Check 1"
        assert vehicles["loc_y_norm"].isnull().sum() == 0, "Check 2"

        # 2 - Move vehicles along their routes, update rides and vehicles
        updated_ride_status = step_along_route(self, ActiveRideDF(active_rides))  # type: ignore
        self.debug_active_rides = active_rides.copy(deep=True)
        self.updated_ride_status = updated_ride_status

        new_coords = interpolate_position_on_edge(
            self,  # type: ignore
            updated_ride_status.curr_start_node[updated_ride_status.f_has_route],
            updated_ride_status.curr_end_node[updated_ride_status.f_has_route],
            updated_ride_status.route_dist_on_edge[updated_ride_status.f_has_route],
        )
        # masked_assign(vehicles, updated_ride_status.f_has_route, vehicles, ["loc_x_norm", "loc_y_norm"])

        vehicles.loc[updated_ride_status.f_has_route, ["loc_x_norm", "loc_y_norm"]] = new_coords[
            ["x_norm", "y_norm"]
        ].to_numpy()
        assert vehicles["loc_x_norm"].isnull().sum() == 0, "Check 3"
        assert vehicles["loc_y_norm"].isnull().sum() == 0, "Check 4"
        ride_update_cols = [
            "curr_start_node",
            "curr_end_node",
            "route_dist_on_edge",
            "trip_distance_remaining_meters",
            "is_complete",
        ]
        active_rides[ride_update_cols] = updated_ride_status[ride_update_cols]

        # 3 - Compute rewards for rides
        f_reward_mask = veh_old.f_get_rewards | vehicles.f_get_rewards
        self._trip_rewards = compute_amortized_reward(updated_ride_status, prev_rides, self.config, f_reward_mask)  # type: ignore
        self._rewards.update({"ride_reward_total": self._trip_rewards.sum()})

        # 4 - Handle completed rides
        completed_mask = updated_ride_status["is_complete"]
        discard_ride_ids = updated_ride_status["ride_id"][completed_mask]
        vehicles["status"] = np.where(completed_mask, VehicleStatusEnum.IDLE, vehicles["status"])
        vehicles["ride_id"] = np.where(completed_mask, self.config.invalid_id, vehicles["ride_id"])
        assert vehicles["loc_x_norm"].isnull().sum() == 0, "Check 5"
        assert vehicles["loc_y_norm"].isnull().sum() == 0, "Check 6"

        # 5 - Handle repositioning of idle vehicles. NOTE this is a cheat - vehicles are instantly repositioned
        reposition_actions = action["reposition"]
        vehicles.loc[vehicles.f_idle, ["loc_x_norm", "loc_y_norm"]] = reposition_actions[vehicles.f_idle]
        assert vehicles["loc_x_norm"].isnull().sum() == 0, "Check 7"
        assert vehicles["loc_y_norm"].isnull().sum() == 0, "Check 8"

        # 6 - Reset completed rides
        gen_new_rides = init_active_ride_df(self, vehicles=vehicles)
        active_rides.reset_index(drop=True, inplace=True)
        # print(active_rides.shape, gen_new_rides.shape, completed_mask.shape)
        active_rides[completed_mask] = gen_new_rides[completed_mask]

        # Cleanup indexes
        vehicles.reset_index(drop=True, inplace=True)
        active_rides.reset_index(drop=True, inplace=True)
        self._active_rides_debug = active_rides.copy(deep=True)
        self._vehicles_debug = vehicles.copy(deep=True)

        return VehicleDF(vehicles), ActiveRideDF(active_rides), discard_ride_ids

    def _cycle_requests(self, discard_ride_ids: pd.Series) -> RequestDF:
        """
        Spawn new requests, remove completed/cancelled requests, and trim to max_pending_requests.

        Deterministic
        """
        requests = RequestDF(self._req_interim.copy(deep=True))

        # 1 - Remove requests associated with completed rides
        self._f_discard = (requests.f_inactive) | requests["request_id"].isin(discard_ride_ids)
        discarded_requests_ts = [RequestDF(requests[self._f_discard]).reset_index(drop=True)]
        requests = requests[~self._f_discard].reset_index(drop=True)

        # 2 - Trim to max_pending_requests based on wait time (longest waiting kept)
        spawned_req = RequestDF.spawn_requests(self, self.config.max_new_requests_per_step)
        filler_requests = RequestDF.generate_empty(  # this is safe if num_rows is <= 0
            num_rows=self.config.max_pending_requests - (len(requests) + len(spawned_req)), dt=self.time_dt
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            requests = RequestDF(pd.concat([spawned_req, requests, filler_requests]).reset_index(drop=True))

        if len(requests) > self.config.max_pending_requests:
            f_prune = requests.index >= self.config.max_pending_requests
            discarded_requests_ts.append(RequestDF(requests[f_prune & requests.f_valid]).reset_index(drop=True))
            requests = requests[~f_prune].reset_index(drop=True)

        for dr in discarded_requests_ts:
            dr["discard_time"] = self.time_dt
            dr["discard_step"] = self.current_step

        self._remove_requests.extend(discarded_requests_ts)
        validate_typed_df_keys(requests, RequestDF)
        self._req_interim = RequestDF(requests)
        return self._req_interim

    def _assert_state_consistency(self, action: ActionDict):
        """
        Assert the internal consistency of the environment state.
            - No two vehicles occupy the same node.
            - Active rides correspond to vehicles that are currently occupied.
            - Pending requests do not exceed maximum allowed.
            - Vehicle states are valid.
            - self.active_rides aligns with self.observation_prev["active_rides"]

        Deterministic
        """
        vehicles = self._veh_interim
        active_rides = self._active_rides_interim
        requests = self._req_interim
        self._veh_ride_sanity_df, veh_ride_sanity_df = pd.DataFrame(), pd.DataFrame()
        self._rides_debug, rides_debug = pd.DataFrame(), pd.DataFrame()
        self._request_sanity_df, request_sanity_df = pd.DataFrame(), pd.DataFrame()
        self._request_debug, request_debug = pd.DataFrame(), pd.DataFrame()

        """Vehicles and Active Rides"""

        # Check that vehicles and active rides are aligned
        veh_ride_sanity_df["vehicle_id"] = vehicles["vehicle_id"] == active_rides["vehicle_id"]
        veh_ride_sanity_df["ride_id"] = vehicles["ride_id"] == active_rides["ride_id"]
        veh_ride_sanity_df["validity_veh_ride"] = vehicles.f_should_have_ride_id == active_rides.f_valid

        # Active rides correspond to vehicles that are currently occupied
        veh_ride_sanity_df["status_valid"] = vehicles["status"].isin([status.value for status in VehicleStatusEnum])
        veh_ride_sanity_df["rides_incomplete"] = ~active_rides["is_complete"]
        veh_ride_sanity_df["ride_price"] = active_rides["price"].notna() | ~vehicles.f_valid
        veh_ride_sanity_df["ride_est_cost"] = active_rides["est_cost"].notna() | ~vehicles.f_valid

        """Dispatches"""
        f_req_dispatched = self._dispatch_debug["is_newly_assigned"]
        f_veh_dispatched = vehicles["vehicle_id"].isin(self._dispatch_debug["vehicle_id"]) & vehicles.f_valid

        veh_prev = self.observation_prev["vehicles"]
        veh_ride_sanity_df["dispatch_idle_only"] = veh_prev.f_idle | ~f_veh_dispatched
        self.observation_prev["active_rides"]

        """Requests"""
        request_sanity_df["status_valid"] = requests["status"].isin([status.value for status in RequestStatusEnum])
        request_sanity_df["price_non_negative"] = (
            (requests["price"] >= 0) | requests["price"].isna() | ~requests.f_valid
        )
        request_sanity_df["request_id_unique"] = (
            requests["request_id"][requests.f_valid].unique().shape[0] == requests.f_valid.sum()
        )
        request_sanity_df["price_not_set"] = (requests["price"].isna() & requests.f_awaiting_price) | (
            ~requests.f_awaiting_price
        )
        request_sanity_df["max_wait_time"] = (requests["max_wait_time"]) == pd.Timedelta(
            minutes=self.config.max_wait_time_minutes
        )
        request_sanity_df["wait_time_env_time"] = (
            requests["request_dt"] + requests["wait_time"] == self.time_dt
        ) | ~requests.f_valid
        request_sanity_df["pickup_node_id_valid"] = (
            requests["pickup_node_id"].isin(self.node_df["node_id"]) | ~requests.f_valid
        )
        request_sanity_df["dropoff_node_id_valid"] = (
            requests["dropoff_node_id"].isin(self.node_df["node_id"]) | ~requests.f_valid
        )
        request_sanity_df["max_pending_requests"] = len(requests) <= self.config.max_pending_requests
        request_sanity_df["dispatch_unique_vehicle"] = (
            self._dispatch_debug["vehicle_id"][f_req_dispatched].is_unique | ~f_req_dispatched
        )

        self._active_rides_debug = active_rides

        self._veh_ride_sanity_df = veh_ride_sanity_df
        self._rides_debug = rides_debug
        self._request_sanity_df = request_sanity_df
        self._request_debug = request_debug

        assert veh_ride_sanity_df.all().all(), f"Vehicle-ActiveRide consistency check failed"
        assert request_sanity_df.all().all(), f"Request consistency check failed"

    def reset_debug_and_interim(self):
        """
        Clear all interim and debug dataframes to free up memory.
        """
        # raise NotImplementedError("reset_debug_and_interim not yet implemented.")
        print("resetting debug and interim dataframes")
        for attr in [
            "_veh_ride_sanity_df",
            "_rides_debug",
            "_request_sanity_df",
            "_request_debug",
            "_dispatch_debug",
            "_req_interim",
            "_veh_interim",
            "_active_rides_interim",
            "_valid_active_rides_debug",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    def _update_metrics(self):
        """
        Update environment metrics based on completed rides, cancellations, rejections, etc.

        Deterministic
        """
        raise NotImplementedError("Metrics update implemented in get_observation.")

    def _update_routes(self):
        """
        Recalculate routes for vehicles if necessary (e.g., due to dynamic traffic conditions).

        Deterministic (for now, as traffic not implemented)
        """
        raise NotImplementedError("Dynamic re-routing not yet implemented.")

    def _update_latent_variables(self):
        """
        Update any latent variables in the environment that may affect transitions.
        For example: traffic conditions & lambda values, weather changes, etc.

        Stochastic (if implemented)
        """
        raise NotImplementedError("Latent variable updates not yet implemented.")

    def _update_battery(self):
        """
        Charge / drain vehicle batteries based on charging status and movement.

        Deterministic

        NOTE battery life not yet implemented
        """
        raise NotImplementedError("Battery life not yet implemented.")
