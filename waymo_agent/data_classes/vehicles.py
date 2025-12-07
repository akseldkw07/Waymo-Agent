from __future__ import annotations

import enum
import typing as t

import numpy as np

from waymo_agent.data_classes.config import EnvConfig
from waymo_agent.data_classes.enriched_df_base import EnrichedDF
from waymo_agent.graph_env.df_utils import validate_typed_df_keys

if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin.obs_space_mixin import ObservationSpaceMixin


class VehicleStatusEnum(enum.IntEnum):
    IDLE = 0  # idle means vehicle is not assigned to any request, can be repositioned
    TO_PICKUP = 1
    WITH_PASSENGER = 2
    CHARGING = 3  # vehicle is at a charging station, recharging. NOTE not currently applicable


class VehicleDF(EnrichedDF):
    ride_id: np.ndarray
    vehicle_id: np.ndarray  # unique identifier for the vehicle. [0, num_vehicles)
    loc_x_norm: np.ndarray  # [-1, 1]
    loc_y_norm: np.ndarray  # [-1, 1]
    battery: np.ndarray  # [0.0, 1.0]
    status: np.ndarray  # VehicleStatusEnum value

    # Training view: [loc_x_norm, loc_y_norm, battery, status_one_hot(4)] -> 7 dims
    cols_to_keep: t.ClassVar[list[str]] = ["loc_x_norm", "loc_y_norm", "battery"]
    enum_fields: t.ClassVar[dict[str, type[enum.IntEnum]]] = {"status": VehicleStatusEnum}
    target_dtypes: t.ClassVar[dict[str, type]] = {
        "ride_id": np.int64,
        "vehicle_id": np.int64,
        "loc_x_norm": np.float32,
        "loc_y_norm": np.float32,
        "battery": np.float32,
        "status": np.int8,
    }

    @property
    def f_available(self) -> np.ndarray:
        return self.status == VehicleStatusEnum.IDLE

    @property
    def f_get_rewards(self) -> np.ndarray:
        return self.status == VehicleStatusEnum.WITH_PASSENGER

    @classmethod
    def space_config(cls, config: EnvConfig, num_vehicles: int):
        width = cls.calc_width()
        shape = (num_vehicles, width)
        row_high = np.ones((width,), dtype=np.float32)
        low = np.zeros(shape, dtype=np.float32)
        high = np.tile(row_high, (num_vehicles, 1))
        return {"shape": shape, "low": low, "high": high}

    @classmethod
    def generate_empty(cls, env: ObservationSpaceMixin) -> VehicleDF:
        idx_offset = env.config.no_action_id + 1  # leave space for no-action vehicle if needed
        idx = np.arange(env.num_vehicles) + idx_offset
        nodes = env.node_df.sample(env.num_vehicles, replace=True).reset_index(drop=True)
        batteries = np.random.uniform(0.6, 1.0, size=env.num_vehicles)
        status = np.full(shape=env.num_vehicles, fill_value=int(VehicleStatusEnum.IDLE), dtype=np.int8)

        ret = VehicleDF(
            {
                "vehicle_id": idx,
                "loc_x_norm": nodes["x_norm"],
                "loc_y_norm": nodes["y_norm"],
                "battery": batteries,
                "status": status,
                "ride_id": -1 * np.ones(env.num_vehicles, dtype=np.int64),
            }
        )
        validate_typed_df_keys(ret, VehicleDF)
        return ret
