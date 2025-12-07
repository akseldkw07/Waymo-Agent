from __future__ import annotations


import numpy as np
import pandas as pd

from waymo_agent.data_classes import ActionDict, ObservationDict, RequestDF, price_acceptance_probability
from waymo_agent.data_classes.active_rides import ActiveRideDF
from waymo_agent.data_classes.requests import RequestStatusEnum
from waymo_agent.data_classes.vehicles import VehicleDF, VehicleStatusEnum
from waymo_agent.graph_env.cost_reward import compute_amortized_reward
from waymo_agent.graph_env.df_utils import validate_typed_df_keys
from waymo_agent.osmnx.euclidean_L2_embed import interpolate_position_on_edge
from waymo_agent.osmnx.traverse_graph import step_along_route
from waymo_agent.simulation.generate_obs_state import init_active_ride_df

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

    def get_observation(self, action: ActionDict) -> ObservationDict:
        """
        Construct the observation dictionary from the current environment state. The returned observaion
        is post-transition, i.e., after all updates have been applied for the current step.

        Deterministic
        """
        self.observation_prev = self.observation_curr
        del self.observation_curr  # to avoid accidental usage
        self._rewards = {}

        self._advance_clock()
        self._req_interim, matches = self._update_requests_from_action(action)
        new_rides = self._new_rides(self._req_interim, matches)
        self._veh_interim, self._active_rides_interim, discard_ride_ids = self._update_vehicles(action, new_rides)
        self._trim_requests(discard_ride_ids)
        self._update_metrics()
        self._assert_state_consistency()

        # TODO set self.observation_curr properly
        # TODO validate observation keys and types
        # TODO update metrics (eg completed rides, cancellations, rejections)
        # TODO add row to breadcrumbs
        # TODO sum up rewards
        raise NotImplementedError("get_observation not yet implemented.")

    def _update_requests_from_action(self, action: ActionDict):
        """
        Update the status of all pending requests, including wait times and cancellations.
            - when price is offered, customer accepts or rejects
            - if price accepted in previous step, vehicle is dispatched (if possible)
            - if vehicle arrives at pickup, ride starts
            - if wait time exceeds threshold, request is cancelled

        Deterministic + Stochastic (acceptance)
        """
        # 1- new requests & copy over existing requests
        spawned_req = RequestDF.spawn_requests(self)
        requests = self.observation_prev["pending_requests"].copy(deep=True)

        # 2 - price acceptance
        sd_values = [
            int(bc["supply_demand_ratio"])
            for bc in self._breadcrumbs[-5:]
            if bc.get("supply_demand_ratio") is not None and isinstance(bc["supply_demand_ratio"], (int, float))
        ]
        sd_ratio_avg = sum(sd_values) / len(sd_values) if sd_values else 1.0

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

        requests["price"] = np.where(await_mask, action["prices"], requests["price"])

        # 3 - dispatches
        out = self._match_dispatches(action, requests)
        matches_orig = out.copy(deep=True)
        match_vehicle_id = out.pop("vehicle_id")
        r_dispatch = out

        new_status = np.select(  # update request statuses based on dispatch matches
            [newly_assigned := (requests.f_need_dispatch & (match_vehicle_id != self.config.no_action_id))],
            [RequestStatusEnum.ASSIGNED],
            new_status,
        )
        matches_orig["newly_assigned"] = newly_assigned

        # 4- update wait times & cancellations
        requests["wait_time"] = requests["request_dt"] - self.time_dt  # update wait times
        cancel_mask = requests["wait_time"] > requests["max_wait_time"]
        new_status = np.select(
            [cancel_mask],
            [RequestStatusEnum.CANCEL_EXCEED_WAIT_TIME],
            new_status,
        )

        # 5 - apply rewards from pricing and dispatching
        self._rewards.update(r_dispatch.sum().to_dict())
        r_reject = (requests["status"] == RequestStatusEnum.REJECTED).sum() * self.config.penalty_rejected
        r_expire = (requests["status"] == RequestStatusEnum.CANCEL_EXCEED_WAIT_TIME).sum() * self.config.penalty_expire
        self._rewards.update({"penalty_rejected": r_reject, "penalty_expire": r_expire})

        # 6 - stack new requests NOTE ds might exceed max_pending_requests, will trim later
        requests["status"] = new_status
        requests = RequestDF(pd.concat([requests, spawned_req], ignore_index=True))
        validate_typed_df_keys(requests, RequestDF)
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
        rewards_df = RWD.pen_df_empty(len=len(requests))

        dispatches = action["dispatch"]
        vehicles = self.observation_curr["vehicles"]

        vehicle_id = np.full(shape=len(dispatches), fill_value=self.config.no_action_id, dtype=np.int64)

        for req_idx, veh_idx in enumerate(dispatches):
            if veh_idx == self.config.no_action_id:
                continue  # No vehicle assigned

            # is vehicle available for assignment (not busy)?
            avail = vehicles.f_available[veh_idx]
            rewards_df.loc[req_idx, "penalty_assign_to_unavailable_vehicle"] = (
                1 - avail
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
        return out

    def _new_rides(self, requests: RequestDF, matches: pd.DataFrame):
        f_new = matches["newly_assigned"]
        new_rides = ActiveRideDF.from_requests(requests[f_new], matches["vehicle_id"][f_new])
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
        active_rides = prev_rides.copy(deep=True)

        f_new_rides = np.isin(active_rides.vehicle_id, (new_rides["vehicle_id"]))
        active_rides[f_new_rides] = new_rides.reset_index(drop=True)
        assert pd.Series(active_rides.vehicle_id).equals(pd.Series(prev_rides.vehicle_id))

        # 2 - This is a cheat (TODO fix) - just move newly_assigned vehicle to start of ride
        veh_old = self.observation_prev["vehicles"].copy(deep=True)
        vehicles = self.observation_curr["vehicles"].copy(deep=True)
        vehicles["status"] = np.where(f_new_rides, VehicleStatusEnum.WITH_PASSENGER, vehicles["status"])
        vehicles["loc_x_norm"] = np.where(f_new_rides, active_rides["pickup_x_norm"], vehicles["loc_x_norm"])
        vehicles["loc_y_norm"] = np.where(f_new_rides, active_rides["pickup_y_norm"], vehicles["loc_y_norm"])

        # 2 - Move vehicles along their routes and update vehicles
        updated_ride_status = step_along_route(self, active_rides)  # type: ignore
        new_coords = interpolate_position_on_edge(
            self,  # type: ignore
            updated_ride_status.curr_start_node,
            updated_ride_status.curr_end_node,
            updated_ride_status.route_dist_on_edge,
        )
        vehicles["loc_x_norm"] = new_coords["x_norm"]
        vehicles["loc_y_norm"] = new_coords["y_norm"]

        # 3 - Compute rewards for rides
        f_reward_mask = veh_old.f_get_rewards & vehicles.f_get_rewards
        trip_rewards = compute_amortized_reward(updated_ride_status, prev_rides, self.config, f_reward_mask)  # type: ignore
        self._rewards.update({"ride_reward_total": trip_rewards.sum()})

        # 4 - Handle completed rides
        completed_mask = updated_ride_status["is_complete"]
        discard_ride_ids = updated_ride_status["ride_id"][completed_mask]
        vehicles["status"] = np.where(completed_mask, VehicleStatusEnum.IDLE, vehicles["status"])
        gen_new_rides = init_active_ride_df(self, vehicles)

        # 5 - Handle repositioning of idle vehicles
        reposition_actions = action["reposition"]
        f_reposition = vehicles.f_available
        vehicles.loc[f_reposition, ["loc_x_norm", "loc_y_norm"]] = reposition_actions[f_reposition]

        # 6 - Update vehicles and active rides
        active_rides[completed_mask] = gen_new_rides[completed_mask]

        return vehicles, active_rides, discard_ride_ids

    def _update_active_rides(self):
        """
        Update active rides based on vehicle movements and ride progress.
            - if vehicle arrives at dropoff, complete ride and free vehicle
            - update ride distances and times

        Deterministic
        """

    def _move_vehicles(self):
        """
        Move vehicles along their paths based on their current tasks (rides, repositioning).

        Deterministic
        """
        raise NotImplementedError("Vehicle movement not yet implemented.")

    def _trim_requests(self, discard_ride_ids: pd.Series) -> RequestDF:
        """
        Trim the requests dataframe to ensure it does not exceed max_pending_requests.

        Deterministic
        """
        requests = self._req_interim.copy(deep=True)

        # 1 - Remove requests associated with completed rides
        f_discard = requests["request_id"].isin(discard_ride_ids)
        requests = requests[~f_discard].reset_index(drop=True)

        # 2 - Trim to max_pending_requests based on wait time (longest waiting kept)
        if len(requests) > self.config.max_pending_requests:
            requests = requests.sort_values(by="wait_time", ascending=False).reset_index(drop=True)
            requests = requests.iloc[: self.config.max_pending_requests].reset_index(drop=True)

        validate_typed_df_keys(requests, RequestDF)
        self._req_interim = requests
        return requests

    def _advance_clock(self):
        """
        Advance the simulation clock by one time step.

        Deterministic
        """
        self.current_step += 1
        self.time_dt += self.config.time_step_delta

        self.calc_time_normed()

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

    def _update_metrics(self):
        """
        Update environment metrics based on completed rides, cancellations, rejections, etc.

        Deterministic
        """

    def _update_routes(self):
        """
        Recalculate routes for vehicles if necessary (e.g., due to dynamic traffic conditions).

        Deterministic (for now, as traffic not implemented)
        """
        raise NotImplementedError("Dynamic re-routing not yet implemented.")

    def _assert_state_consistency(self):
        """
        Assert the internal consistency of the environment state.
            - No two vehicles occupy the same node.
            - Active rides correspond to vehicles that are currently occupied.
            - Pending requests do not exceed maximum allowed.
            - Vehicle states are valid.
            - self.active_rides aligns with self.observation_curr["active_rides"]

        Deterministic
        """
