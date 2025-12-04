"""
TODO
"""

from dataclasses import dataclass

from waymo_agent.data_classes.dataclasses import RequestState


@dataclass
class Customer:
    id: int
    customer_bias: float

    def request_acceptance_probability(self, request: RequestState) -> float:
        """
        Compute the probability that a customer will accept a ride request
        based on the pricing and other factors.
        TODO implement the acceptance model based price, distance, wait time, supply-demand ratio, and customer bias
        Return a float between 0.0 and 1.0
        """
        ...
