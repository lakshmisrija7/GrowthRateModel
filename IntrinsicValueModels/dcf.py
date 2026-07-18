from typing import List, Union
from .exceptions import InvalidParameterError, ValuationError
from .logger import get_logger

logger = get_logger(__name__)

class DiscountedCashFlowModel:
    def __init__(self, initial_fcf: float, discount_rate: float):
        if initial_fcf <= 0:
            raise InvalidParameterError("Initial free cash flow must be positive")
        if discount_rate <= 0:
            raise InvalidParameterError("Discount rate must be positive")
        self.initial_fcf = initial_fcf
        self.discount_rate = discount_rate

    def calculate_valuation(
        self,
        growth_rates: Union[float, List[float]],
        forecast_years: int,
        terminal_growth_rate: float,
        net_debt: float = 0.0,
        shares_outstanding: float = 1.0
    ) -> float:
        if forecast_years <= 0:
            raise InvalidParameterError("Forecast years must be positive")
        if terminal_growth_rate >= self.discount_rate:
            raise ValuationError("Terminal growth rate must be strictly less than the discount rate")
        if shares_outstanding <= 0:
            raise InvalidParameterError("Shares outstanding must be positive")

        if isinstance(growth_rates, (int, float)):
            rates = [float(growth_rates)] * forecast_years
        else:
            if len(growth_rates) < forecast_years:
                raise InvalidParameterError("Provided growth rates list is shorter than the forecast period")
            rates = [float(r) for r in growth_rates[:forecast_years]]

        logger.info(f"Running DCF: FCF0={self.initial_fcf}, r={self.discount_rate}, rates={rates}, terminal_g={terminal_growth_rate}")

        pv_cash_flows = 0.0
        current_fcf = self.initial_fcf
        for t in range(1, forecast_years + 1):
            g = rates[t - 1]
            current_fcf *= (1.0 + g)
            pv = current_fcf / ((1.0 + self.discount_rate) ** t)
            pv_cash_flows += pv

        terminal_value = current_fcf * (1.0 + terminal_growth_rate) / (self.discount_rate - terminal_growth_rate)
        pv_terminal_value = terminal_value / ((1.0 + self.discount_rate) ** forecast_years)

        enterprise_value = pv_cash_flows + pv_terminal_value
        equity_value = enterprise_value - net_debt
        value_per_share = equity_value / shares_outstanding

        logger.info(f"DCF EV={enterprise_value}, Equity={equity_value}, ValuePerShare={value_per_share}")
        return value_per_share
