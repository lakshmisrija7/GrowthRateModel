from .exceptions import InvalidParameterError, ValuationError
from .logger import get_logger

logger = get_logger(__name__)

class DividendDiscountModel:
    def __init__(self, last_dividend: float, discount_rate: float):
        if last_dividend < 0:
            raise InvalidParameterError("Last dividend cannot be negative")
        if discount_rate <= 0:
            raise InvalidParameterError("Discount rate must be positive")
        self.last_dividend = last_dividend
        self.discount_rate = discount_rate

    def calculate_constant_growth(self, growth_rate: float) -> float:
        if growth_rate >= self.discount_rate:
            raise ValuationError("Growth rate must be strictly less than the discount rate for constant growth model")
        
        logger.info(f"Running Constant Growth DDM: dividend={self.last_dividend}, r={self.discount_rate}, g={growth_rate}")
        next_dividend = self.last_dividend * (1.0 + growth_rate)
        value = next_dividend / (self.discount_rate - growth_rate)
        logger.info(f"Constant Growth DDM result: {value}")
        return value

    def calculate_two_stage_growth(self, high_growth_rate: float, high_growth_years: int, stable_growth_rate: float) -> float:
        if high_growth_years < 0:
            raise InvalidParameterError("High growth years cannot be negative")
        if stable_growth_rate >= self.discount_rate:
            raise ValuationError("Stable growth rate must be strictly less than the discount rate")
        
        logger.info(f"Running Two-Stage Growth DDM: dividend={self.last_dividend}, r={self.discount_rate}, g1={high_growth_rate}, n={high_growth_years}, g2={stable_growth_rate}")
        
        present_value_dividends = 0.0
        current_dividend = self.last_dividend
        
        for t in range(1, high_growth_years + 1):
            current_dividend *= (1.0 + high_growth_rate)
            pv = current_dividend / ((1.0 + self.discount_rate) ** t)
            present_value_dividends += pv
            
        terminal_dividend = current_dividend * (1.0 + stable_growth_rate)
        terminal_value = terminal_dividend / (self.discount_rate - stable_growth_rate)
        pv_terminal_value = terminal_value / ((1.0 + self.discount_rate) ** high_growth_years)
        
        total_value = present_value_dividends + pv_terminal_value
        logger.info(f"Two-Stage Growth DDM result: {total_value}")
        return total_value
