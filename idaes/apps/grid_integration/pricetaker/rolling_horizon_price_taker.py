#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################

from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    Param,
    RangeSet,
    Objective,
    Constraint,
    NonNegativeReals,
    Expression,
    maximize,
    SolverFactory
)

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeInt,
    NonNegativeFloat,
    ListOf,
    PositiveInt,
)
import numpy as np
from idaes.core.util.config import ConfigurationError, is_in_range
from idaes.apps.grid_integration import RHPTForecaster, PriceTakerModel, DesignModel, OperationModel
import idaes.logger as idaeslog

_logger = idaeslog.getLogger(__name__)

# Defining a ConfigDict to simplify the domain validation of
# arguments needed for all the methods of the PriceTakerModel class
CONFIG = ConfigDict()

# List of arguments needed for adding startup/shutdown constraints
CONFIG.declare(
    "up_time",
    ConfigValue(
        domain=PositiveInt,
        doc="Minimum uptime [in hours]",
    ),
)
CONFIG.declare(
    "down_time",
    ConfigValue(
        domain=PositiveInt,
        doc="Minimum downtime [in hours]",
    ),
)

# List of arguments for NPV calculation
CONFIG.declare(
    "lifetime",
    ConfigValue(
        domain=PositiveInt,
        doc="Total lifetime of the system [in years]",
    ),
)
CONFIG.declare(
    "discount_rate",
    ConfigValue(
        domain=is_in_range(0, 1),
        doc="Discount rate for annualization [fraction]",
    ),
)
CONFIG.declare(
    "corporate_tax_rate",
    ConfigValue(
        domain=is_in_range(0, 1),
        doc="Effective corporate tax rate [fraction]",
    ),
)
CONFIG.declare(
    "annualization_factor",
    ConfigValue(
        domain=is_in_range(0, 1),
        doc="Capital cost annualization factor [fraction]",
    ),
)
CONFIG.declare(
    "cash_inflow_scale_factor",
    ConfigValue(
        domain=NonNegativeFloat,
        doc="Scaling factor for net cash inflow calculations",
    ),
)

class RHPTModel(ConcreteModel):
    """Builds a price-taker model for a given system"""

    def __init__(self, forecaster, *args, **kwds):
        super().__init__(*args, **kwds)
        self.forecaster = forecaster
        self._config = CONFIG()
        self._has_hourly_cashflows = False
        self._has_overall_cashflows = False
        self._op_blk_uptime_downtime = {}

        # Set the horizon/planning_horizon from the forecaster
        self._horizon = self.forecaster.horizon
        self._planning_horizon = self.forecaster.planning_horizon
        self._scenario = self.forecaster.scenario
        
        # if planning horizon == horizon, no look ahead.
        if self._horizon == self._planning_horizon:
            self.look_ahead = False
        else:
            self.look_ahead = True

        self.lmp_data_check()
        # self.model_type_check()
    

    def lmp_data_check(self):
        """
        Output the LMP data information
        """
        lmp_reshaped = self.forecaster.lmp_check()
        num_periods = np.shape(lmp_reshaped)[0]
        horizon_length = np.shape(lmp_reshaped)[1]
        
        _logger.info(f"The total number of periods in this rolling horizon is {num_periods}.")
        if not (horizon_length == self._horizon):
            raise ValueError(f"The length of the LMP horizon is {horizon_length}, but the price-taker horizon is {self._horizon}.")
        _logger.info(f"The length of the LMP horizon is {horizon_length}")

        return


    def _build_PT_model(self, LMP_data, ):
        """
        Build a stochastic optimization problem, each scenario is with the length of self._horizon
        
        Args: 

        Returns:
            m: pyomo model, return this model can allow the user to further add constraints.
            
        """
        # Build a standard PT class model for each scenario
        m = PriceTakerModel()
        
        # Append the LMP data to the PT model
        m.append_lmp_data(LMP_data)
        
        # Build the multiperiod model
        m.build_multiperiod_model(flowsheet_func=self.gen_flowsheet_func, flowsheet_options=self.gen_flowsheet_options)
        
        return m


    def build_stochasctic_PT_model(self, initial_state):
        """
        Build the stochastic price-taker model
        """
        m = ConcreteModel()
        m.set_scenarios = RangeSet(self._scenario)
        m.scenarios = Block(m.set_scenarios)
        for s in m.scenarios:
            scenario_model = self._build_PT_model(LMP_data)
            m.scenarios[s].transfer_attributes_from(scenario_model.clone())
        
        return
    
    
    def report_final_state(self):
        """
        Report the final state of the model. The  
        """
        return
    

    def record_solution(self, soln):
        """
        record the results from solved model.
        """
        return
    

class RHPTRunner:
    def __init__(self, periods, forecaster, flowsheet_func, flowsheet_options, model):
        self.periods = periods
        self.forecaster = forecaster
        self.model = model
        self.flowsheet_func = flowsheet_func
        self.flowsheet_options = flowsheet_options

    def _check_inputs(self):
        isinstance(self.period, int)
        return

    def run_rolling_horizon(self, init_state, solver="gurobi", solver_options={}):
        """
        Run the rolling horizon optimization. 

        Args:
            init_state: dictionary, the initial state at the beginning of rolling horizon optimization.
            solver: dictionary, the solver for solving the optimization or simulation problem.
            solver_options: dictionary, the solver options.

        Returns:
            results_dict: dictionary, keys are periods, values are results.
        """
        results_dict = {}
        for i in range(self.periods):
            _logger.info(f"Building rolling horizon optimization for period {i}.")
            model = self.model.build_multiperiod_problem(pointer=i, flowsheet_func=self.flowsheet_func, flowsheet_options=self.flowsheet_options)
            opt_solver = SolverFactory(solver)
            soln = opt_solver.solve(model, tee=True, options=solver_options)
            results_dict[i] = self.model.read_solution(soln)
            init_state = self.model.report_final_states()

        return results_dict
    
    def visual_results(self, results_dict):
        """
        Visualize the results of the rolling horizon optimization

        Args:
            results_dict: results dictionary from run_rolling_horizon function.

        Returns:
            None
        """
        return

