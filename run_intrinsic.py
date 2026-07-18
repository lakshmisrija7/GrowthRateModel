import sys
from IntrinsicValueModels import DividendDiscountModel, DiscountedCashFlowModel
from IntrinsicValueModels.exceptions import IntrinsicModelError
from IntrinsicValueModels.logger import get_logger

logger = get_logger("runner")

def run_ddm():
    logger.info("=== Running DDM Valuation ===")
    ddm = DividendDiscountModel(last_dividend=2.00, discount_rate=0.08)
    
    constant_growth_val = ddm.calculate_constant_growth(growth_rate=0.04)
    logger.info(f"Constant Growth DDM Intrinsic Value: ${constant_growth_val:.2f}")

    two_stage_val = ddm.calculate_two_stage_growth(
        high_growth_rate=0.10,
        high_growth_years=5,
        stable_growth_rate=0.03
    )
    logger.info(f"Two-Stage DDM Intrinsic Value: ${two_stage_val:.2f}")

def run_dcf():
    logger.info("=== Running DCF Valuation ===")
    dcf = DiscountedCashFlowModel(initial_fcf=150.0, discount_rate=0.09)
    
    val_single_growth = dcf.calculate_valuation(
        growth_rates=0.06,
        forecast_years=5,
        terminal_growth_rate=0.03,
        net_debt=200.0,
        shares_outstanding=10.0
    )
    logger.info(f"DCF Single Growth Intrinsic Value per Share: ${val_single_growth:.2f}")

    val_multi_growth = dcf.calculate_valuation(
        growth_rates=[0.12, 0.10, 0.08, 0.06, 0.04],
        forecast_years=5,
        terminal_growth_rate=0.03,
        net_debt=200.0,
        shares_outstanding=10.0
    )
    logger.info(f"DCF Multi Growth Intrinsic Value per Share: ${val_multi_growth:.2f}")

def main():
    try:
        run_ddm()
        run_dcf()
    except IntrinsicModelError as e:
        logger.error(f"Valuation model error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
