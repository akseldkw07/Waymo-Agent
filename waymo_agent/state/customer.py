"""
TODO
"""

from waymo_agent.env.dataclasses import RequestState


class Customer:
    def request_acceptance_probability(self, request: RequestState) -> float:
        """
        Compute the probability that a customer will accept a ride request
        based on the pricing and other factors.
        TODO implement the acceptance model based price, distance, wait time, supply-demand ratio, and customer bias
        Return a float between 0.0 and 1.0
        """
        ...
