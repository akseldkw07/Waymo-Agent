from __future__ import annotations

import enum
import typing as t

import numpy as np
import pandas as pd

from waymo_agent.data_classes.config import EnvConfig

BASE_CLASS_ATTRS = {"cols_to_keep", "enum_fields", "target_dtypes"}
if t.TYPE_CHECKING:
    from waymo_agent.graph_env.mixin.interface import GymEnvInterface


class EnrichedDF(pd.DataFrame):
    """
    Base mixin for typed pandas DataFrames that have a corresponding
    “training view” used for RL / PyTorch models.

    Each subclass should specify:
      - `cols_to_keep`: the scalar columns that should be included
        in the model input (e.g. coordinates, distances, costs, etc.).
      - `enum_fields`: a mapping from column name -> IntEnum type,
        indicating which columns should be expanded into one-hot
        vectors in the training representation.

    The combination of `cols_to_keep` and `enum_fields` defines:

      - the feature dimension (`calc_width()`)
      - the one-hot-expanded column names (`enum_one_hot_names`)
      - the transformation from a typed DF to a model-friendly DF
        (`to_training_df`).
    """

    cols_to_keep: t.ClassVar[list[str]] = []
    enum_fields: t.ClassVar[dict[str, type[enum.IntEnum]]] = {}

    target_dtypes: t.ClassVar[dict[str, type]] = {}

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

        This counts:
          - one dimension for each non-enum column in `cols_to_keep`
          - one dimension per enum member for each enum column listed
            in `enum_fields`
        """
        cols_no_enum = [col for col in cls.cols_to_keep if col not in cls.enum_fields]
        width = len(cols_no_enum)
        for enum_col, enum_type in cls.enum_fields.items():
            # each enum becomes one-hot over all members
            width += len(enum_type)
        return width

    @classmethod
    def enum_one_hot_names(cls, enum_class_name: str) -> list[str]:
        """
        Given the string name of an enum class (e.g. "VehicleStatusEnum"),
        return the list of one-hot column names that would be generated
        for that enum in the training representation.

        The names have the form: "{field_name}_{member.name.lower()}"
        for each enum member, where `field_name` is the key in
        `enum_fields` that uses this enum type.
        """
        for field_name, enum_type in cls.enum_fields.items():
            if enum_type.__name__ == enum_class_name:
                return [f"{field_name}_{member.name.lower()}" for member in enum_type]

        raise KeyError(
            f"Enum class name '{enum_class_name}' not found in "
            f"{cls.__name__}.enum_fields: "
            f"{[t.__name__ for t in cls.enum_fields.values()]}"
        )

    @classmethod
    def to_training_df(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a PyTorch-friendly DataFrame from the given typed DataFrame.

        - Keeps the scalar columns in `cols_to_keep` that are *not* enums,
          casting them to float32.
        - For each enum column in `enum_fields`, expands it into one-hot
          columns of the form "{field_name}_{member.name.lower()}" with
          float32 0.0/1.0 values.

        The resulting DataFrame has width equal to `cls.calc_width()`,
        and column order:
            [non-enum cols_to_keep..., enum one-hot columns...]
        """
        import numpy as np  # local import to avoid cycles if any

        non_enum_cols = [c for c in cls.cols_to_keep if c not in cls.enum_fields]

        out_data: dict[str, pd.Series] = {}

        # 1) Copy scalar (non-enum) columns
        for col in non_enum_cols:
            if col not in df.columns:
                raise KeyError(f"{cls.__name__}.to_training_df: column '{col}' not found in DataFrame.")
            out_data[col] = df[col].astype(np.float32)

        # 2) Expand enums into one-hot columns
        for enum_col, enum_type in cls.enum_fields.items():
            if enum_col not in df.columns:
                raise KeyError(f"{cls.__name__}.to_training_df: enum column '{enum_col}' not found in DataFrame.")

            # Underlying data should be integer codes or IntEnum; coerce to int
            vals = df[enum_col].astype(int)

            for member in enum_type:
                col_name = f"{enum_col}_{member.name.lower()}"
                out_data[col_name] = (vals == int(member)).astype(np.float32)

        # Order: non-enum columns, then one-hot columns in enum_fields order
        ordered_cols: list[str] = []
        ordered_cols.extend(non_enum_cols)
        for enum_col, enum_type in cls.enum_fields.items():
            for member in enum_type:
                ordered_cols.append(f"{enum_col}_{member.name.lower()}")

        training_df = pd.DataFrame(out_data)[ordered_cols]

        # Sanity check: width matches our calculation
        assert training_df.shape[1] == cls.calc_width(), (
            f"{cls.__name__}.to_training_df produced width {training_df.shape[1]}, "
            f"but calc_width()={cls.calc_width()}."
        )

        return training_df

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
    def generate_empty(
        cls, *args, num_rows: int | None = None, env: GymEnvInterface | None = None, **kwargs
    ) -> EnrichedDF:
        """
        Generate an empty typed dataframe using the class's `__annotations__`
        and `target_dtypes`.
        """
        data = {}
        num_rows = num_rows if num_rows is not None else (env.num_vehicles if env is not None else 0)

        for col, dtype in cls.target_dtypes.items():
            # numpy dtypes
            if dtype in (np.int64, np.float64, np.int32, np.float32):
                data[col] = np.zeros(num_rows, dtype=dtype)

            elif dtype in (bool, np.bool_):
                data[col] = np.full(num_rows, False, dtype=bool)

            elif dtype is np.datetime64:
                # Pandas requires datetime64[ns] explicitly
                data[col] = pd.to_datetime(["2025-01-01"] * num_rows)

            elif dtype in (object, np.object_):
                # Object dtype → often used for lists (route_nodes)
                data[col] = np.empty(num_rows, dtype=dtype)
                # initialize empty lists if appropriate
                for i in range(num_rows):
                    data[col][i] = []

            else:
                raise TypeError(f"Unsupported dtype {dtype} for column {col}.")

        return cls(data)
