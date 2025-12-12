from __future__ import annotations

import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.config import EnvConfig

BASE_CLASS_ATTRS = {"cols_to_keep", "enum_fields", "target_dtypes", "default_vals"}
if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin.interface import GymEnvInterface


class EnrichedDF(pd.DataFrame):
    """
    Base mixin for typed pandas DataFrames that have a corresponding
    “training view” used for RL / PyTorch models.

    Each subclass should specify:
      - `cols_to_keep`: the scalar columns that should be included
        in the model input (e.g. coordinates, distances, costs, etc.).
    """

    cols_to_keep: t.ClassVar[list[str]] = []
    target_dtypes: t.ClassVar[dict[str, type]] = {}
    default_vals: t.ClassVar[dict[str, t.Any]] = {}

    @classmethod
    def column_order(cls) -> list[str]:
        return list(cls.target_dtypes.keys())

    @property
    def f_valid(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement f_valid property.")

    def validate_dtypes(self):
        """
        Validate that the DataFrame's columns have the expected dtypes
        as specified in `target_dtypes`.
        """
        for col, target_dtype in self.target_dtypes.items():
            if col not in self.columns:
                raise KeyError(f"{self.__class__.__name__}.validate_dtypes: column '{col}' not found in DataFrame.")

            actual_dtype = self[col].dtype.type
            if actual_dtype != target_dtype:
                raise TypeError(
                    f"{self.__class__.__name__}.validate_dtypes: column '{col}' has dtype "
                    f"{actual_dtype}, expected {target_dtype}."
                )

    @classmethod
    def calc_width(cls) -> int:
        """
        Compute the width (number of features) of the training
        representation for this class.
        """
        return len(cls.cols_to_keep)

    def to_training_df(self) -> pd.DataFrame:
        """ """
        training_df = pd.DataFrame(self)[self.cols_to_keep]
        # Sanity check: width matches our calculation
        assert training_df.shape[1] == self.calc_width(), (
            f"{self.__class__.__name__}.to_training_df produced width {training_df.shape[1]}, "
            f"but calc_width()={self.calc_width()}."
        )

        return training_df

    def to_obs_numpy(self):
        """ """
        training_df = self.to_training_df()
        ret = training_df.to_numpy()
        return ret

    @classmethod
    def from_obs_numpy(cls, obs_array: np.ndarray | pd.DataFrame) -> EnrichedDF:
        """ """
        training_df = pd.DataFrame(obs_array, columns=cls.cols_to_keep)

        return cls(training_df)

    @classmethod
    def space_config(cls, config: EnvConfig, *args, **kwargs):
        """
        Stub to be overridden by subclasses that want to expose a
        Gymnasium Box space based on the training feature width.

        Child classes should implement:

            @classmethod
            def space_config(cls, config: EnvConfig, ...):
                width = cls.calc_width()
                shape = (batch_size, width)
                ...

        This stub exists so that pylint/mypy can see a common interface.
        """
        raise NotImplementedError(f"{cls.__name__}.space_config must be implemented in subclasses.")

    @classmethod
    def generate_empty(cls, *args, num_rows: int | None = None, env: GymEnvInterface | None = None, **kwargs):
        """
        Generate an empty typed dataframe using the class's `__annotations__`
        and `target_dtypes`.
        """
        cfg = env.config if env is not None else EnvConfig()
        data = {}
        num_rows = num_rows if num_rows is not None else (env.num_vehicles if env is not None else 0)

        for col, dtype in cls.target_dtypes.items():
            def_val = cls.default_vals.get(col) or (cfg.invalid_id if "id" in col else None)

            if dtype in (np.int64, np.float64, np.int32, np.float32):
                def_val = def_val if def_val is not None else cfg.invalid_id
                data[col] = np.full(num_rows, def_val, dtype=dtype)

            elif dtype in (bool, np.bool_):
                def_val = def_val if def_val is not None else False
                data[col] = np.full(num_rows, def_val, dtype=bool)

            elif dtype is np.datetime64:
                # Pandas requires datetime64[ns] explicitly
                def_val = def_val if def_val is not None else "2025-01-01"
                data[col] = pd.to_datetime([def_val] * num_rows)  # type: ignore
            elif dtype in (np.timedelta64, pd.Timedelta):
                def_val = def_val if def_val is not None else np.timedelta64(0, "s")
                data[col] = pd.Timedelta(def_val, unit="m")  # type: ignore
            elif dtype in (object, np.object_):
                # Object dtype → often used for lists (route_nodes)
                data[col] = np.empty(num_rows, dtype=dtype)

                # initialize empty lists if appropriate
                def_val = def_val if def_val is not None else []
                for i in range(num_rows):
                    data[col][i] = def_val

            else:
                raise TypeError(f"Unsupported dtype {dtype} for column {col}.")

        ret = cls(data)
        validate_typed_df_keys(ret, cls)
        ret = ret[cls.column_order()]
        return cls(ret)


T = t.TypeVar("T", bound=pd.DataFrame)


def validate_typed_df_keys(
    df: pd.DataFrame | dict, df_type: t.Type[T], action: t.Literal["warn", "raise"] = "raise"
) -> bool:
    """
    Validate that a pandas DataFrame conforms to the specified typed DataFrame structure.
    TODO convert down to warning instead of raising
    """
    exp_cols = df_type.target_dtypes.keys() if isinstance(df_type, EnrichedDF) else df_type.__annotations__.keys()
    expected_columns = set(exp_cols) - BASE_CLASS_ATTRS
    actual_columns = set(df.keys()) - BASE_CLASS_ATTRS

    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    if action == "raise":
        assert (
            not missing and not extra
        ), f"DataFrame columns do not match expected structure. Missing: {missing}, Extra: {extra}"

    if missing:
        print(f"WARNING: Missing columns in DataFrame: {missing}")
    if extra:
        print(f"WARNING: Extra columns in DataFrame: {extra}")

    return not missing and not extra
