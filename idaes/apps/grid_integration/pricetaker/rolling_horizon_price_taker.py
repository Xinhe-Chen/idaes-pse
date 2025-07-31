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

from typing import Optional, Union, Callable, List
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
    value,
    maximize,
    SolverFactory
)

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeInt,
    NonNegativeFloat,
    PositiveInt,
)
import numpy as np
import pandas as pd
from idaes.core.util.config import ConfigurationError
from idaes.apps.grid_integration import PriceTakerModel, DesignModel, OperationModel
import idaes.logger as idaeslog

_logger = idaeslog.getLogger(__name__)

# Defining a ConfigDict to simplify the domain validation of
# arguments needed for all the methods of the PriceTakerModel class
CONFIG = ConfigDict()

# List of arguments needed for adding startup/shutdown constraints
CONFIG.declare(
    "scenario",
    ConfigValue(
        domain=PositiveInt,
        doc="Number of scenarios.",
    ),
)
CONFIG.declare(
    "horizon",
    ConfigValue(
        domain=PositiveInt,
        doc="The length of the horizon.",
    ),
)
CONFIG.declare(
    "planning_horizon",
    ConfigValue(
        domain=PositiveInt,
        doc="The length of the planning horizon.",
    ),
)


class StochasticPriceTaker(ConcreteModel):
    """
    Builds a scenario-based stochastic price-taker model for a given system.
    """

    def __init__(self, scenario: int, 
                 horizon: int, 
                 planning_horizon: int, 
                 gen_dict: dict, 
                 max_scenario: Optional[int] = 10, 
                 max_horizon: Optional[int]=24*31, 
                 *args, **kwds):
        """
        Args:
            scenario: int, number of scenarios in the stochastic price-taker.
            horizon: int, the number of periods in each scenario.
            planning_horizon: int, the number of periods that need non-anticipativity constraints.
            gen_dict: dict, dictionary contains the information of the generator.
        """
        super().__init__(*args, **kwds)
        self.scenario = scenario
        self.horizon = horizon
        self.planning_horizon = planning_horizon
        self.gen_dict = gen_dict
        # Users can change the max_scenario and max_horizon to fit their needs.
        self.max_scenario = max_scenario    # default max scenario is 10
        self.max_horizon = max_horizon    # default max horizon is 24*31 (total number of hours in a month)

        self._config = CONFIG()
        # check the inputs
        self._config.horizon = horizon
        self._config.planning_horizon = planning_horizon
        self._config.scenario = scenario
        self._has_hourly_cashflows = False
        self._has_overall_cashflows = False
        self._op_blk_uptime_downtime = {}

        # if planning horizon == horizon, no look ahead.
        if self.horizon == self.planning_horizon:
            self.look_ahead = False
        elif self.horizon > self.planning_horizon:
            self.look_ahead = True
        else:
            raise ValueError(f"The planning horizon {self.planning_horizon} should not exceed {self.horizon}.")

        self.param_check()
    

    @property
    def scenario(self):
        """
        Property getter for scenario.

        Returns:
            int: saved scenario value
        """

        return self._scenario

    @scenario.setter
    def scenario(self, value):
        """
        Property setter for scenario.

        Args:
            value: intended value for scenario

        Returns:
            None
        """
        self._scenario = value


    @property
    def horizon(self):
        """
        Property getter for horizon.

        Returns:
            int: saved horizon value
        """

        return self._horizon

    @horizon.setter
    def horizon(self, value):
        """
        Property setter for horizon.

        Args:
            value: intended value for horizon

        Returns:
            None
        """
        self._horizon = value


    @property
    def planning_horizon(self):
        """
        Property getter for planning_horizon.

        Returns:
            int: saved planning_horizon value
        """

        return self._planning_horizon


    @planning_horizon.setter
    def planning_horizon(self, value):
        """
        Property setter for planning_horizon

        Args:
            value: intended value for planning_horizon

        Returns:
            None
        """
        self._planning_horizon = value

    @property
    def gen_dict(self):
        """
        Property getter for gen_dict.

        Returns:
            dict: saved gen_dict value
        """
        return self._gen_dict


    @gen_dict.setter
    def gen_dict(self, value):
        """
        Property setter for gen_dict

        Args:
            value: intended value for gen_dict

        Returns:
            None
        """
        self._gen_dict = value


    def _gen_dict_check(self):
        """
        Check the inputs from the gen_dict
        """
        return


    def param_check(self):
        """
        Check the input parameters.

        Args:
            lmp_data, array like lmp signals with shape (self.scenario, self.horizon)

        Returns:
            None
        """
        # make sure the scenario and horizon do not excceed the max length
        if self.scenario > self.max_scenario:
            raise ValueError(f"The number of scenarios {self.scenario} exceeds the maximum allowed {self.max_scenario}.")

        if self.horizon > self.max_horizon:
            raise ValueError(f"The number of horizon {self.horizon} exceeds the maximum allowed {self.max_horizon}.")

        # check the self.scenario is a valid int and print it.
        if isinstance(self.scenario, int):
            _logger.info(f"The total number of periods in this rolling horizon is {self.scenario}.")
        
        # check the self.horizon is a valid int and print it.
        if isinstance(self.horizon, int):
            _logger.info(f"The length of the LMP horizon is {self.horizon}")

        return

    
    def lmp_check(self, lmp_data: Union[List, np.ndarray, pd.DataFrame]):
        """
        Check the LMP from the forecaster.

        Args:
            lmp_data: array like lmp signals with shape (self.scenario, self.horizon)

        Returns:
            None
        """
        num_periods = np.shape(lmp_data)[0]
        horizon_length = np.shape(lmp_data)[1]
        
        # Check the given LMP signal should be in the shape of (scenario, horizon)
        if not num_periods == self.scenario:
            raise ValueError(f"The number of scenario is {self.scenario}, but there are only {num_periods} LMPs.")

        if not (horizon_length == self.horizon):
            raise ValueError(f"The length of each LMP scenario is {horizon_length}, but the price-taker problem horizon is {self.horizon}.")

        return


    def _assert_mp_model_exists(self, s: int):
        """
        Raise an error if the multiperiod model does not exist
        
        Args:
            s: int, the index of the scenario.

        Returns:
            None
        """
        if not hasattr(self.scenarios[s], "period"):
            raise ConfigurationError(
                "Unable to find the multiperiod model. Please use the "
                "build_multiperiod_model method to construct one."
            )
    
    
    def _get_operation_vars(self, s: int, var_name: str):
        """
        Returns a dictionary of pointers to the var_name variable located in each flowsheet
        instance. If the variable is not present, then an error is raised.

        Args:
            s: int, the index of the scenario.
            var_name: str, the variable name you want to get

        Returns:
            None
        """
        # Ensure that the multiperiod model exists
        self._assert_mp_model_exists(s)

        # pylint: disable=not-an-iterable
        op_vars = {
            d: {t: self.scenarios[s].period[d, t].find_component(var_name) for t in self.scenarios[s].set_time}
            for d in self.scenarios[s].set_days
        }

        # NOTE: It is sufficient to perform checks only for one variable
        if op_vars[1][1] is None:
            raise AttributeError(
                f"Variable {var_name} does not exist in the multiperiod model."
            )

        return op_vars


    def _get_operation_blocks(self, s: int, blk_name: str, attribute_list: List):
        """
        Returns a dictionary of operational blocks named 'blk_name'.
        In addition, it also checks the existence of the operational
        blocks, and the existence of specified attributes.
        
        Args:
            s: int, the index of the scenario.
            var_name: str, the variable name you want to get

        Returns:
            None
        """
        # Ensure that the multiperiod model exists
        self._assert_mp_model_exists(s)

        # pylint: disable=not-an-iterable
        op_blocks = {
            d: {t: self.scenarios[s].period[d, t].find_component(blk_name) for t in self.scenarios[s].set_time}
            for d in self.scenarios[s].set_days
        }

        # NOTE: It is sufficient to perform checks only for one block, because
        # the rest of them are clones.
        blk = op_blocks[1][1]  # This object always exists

        # First, check for the existence of the operational block
        if blk is None:
            raise AttributeError(f"Operational block {blk_name} does not exist.")

        # Next, check for the existence of attributes.
        for attribute in attribute_list:
            if not hasattr(blk, attribute):
                raise AttributeError(
                    f"Required attribute {attribute} is not found in "
                    f"the operational block {blk_name}."
                )

        return op_blocks


    @staticmethod
    def build_PT_model(LMP_data: Union[List, np.ndarray, pd.DataFrame], 
                       design_func_dict: dict,
                       design_params: dict, 
                       flowsheet_func: Callable, 
                       flowsheet_options: dict):
        """
        Build a stochastic optimization problem, each scenario is with the length of self.horizon.
        
        Args: 
            LMP_data: array like, the LMP data used for building the price-taker problem.
            design_func: list, list of functions. Making this into list enables define different design funcs.

        Returns:
            m: pyomo model, return this model can allow the user to further add constraints.
            
        """
        # Build a standard PT class model for each scenario
        m = PriceTakerModel()
        
        # Append the LMP data to the PT model
        m.append_lmp_data(LMP_data)
        
        for key in list(design_params.keys()):
            setattr(m, 
                    f"gen_design_{design_params[key]['name']}", 
                    DesignModel(model_func=design_func_dict[key],
                                model_args={"params": design_params[key]},
                )
            )

        # Build the multiperiod model
        m.build_multiperiod_model(flowsheet_func, flowsheet_options)
        
        return m


    def _populate_multiperiod_model_capacity(self, m, commodity, capacity_names=None):
        """
        Populate the multiperiod model for the price-taker model. Add the capacity limit constraints.

        Args:
            m: scenario pyomo model instance.
            commodity: str, name of the commodity on the model the capacity constraints.

        Returns:
            None
        """        
        # add capacity limits
        if capacity_names:
            for name in capacity_names:
                m.add_capacity_limits(
                    op_block_name=name,
                    commodity=commodity,
                    capacity=getattr(getattr(m, f"gen_design_{name}"), "capacity"),
                    op_range_lb=self.gen_dict[name]["min_p"]/self.gen_dict[name]["max_p"],
                )

        return

    def _populate_multiperiod_model_startup_shutdown(self, m, startup_names=None):
        """
        Populate the multiperiod model for the price-taker model. Add the startup/shutdown constraints.

        Args:
            m: scenario pyomo model instance.
            commodity: str, name of the commodity on the model the capacity constraints.

        Returns:
            None
        """        
        # add start up and shutdown constraints
        if startup_names:
            # if we have startup and shutdown we need to consider the initial state.
            self._skip_initialization = False
            for name in startup_names:
                m.add_startup_shutdown(
                    op_block_name=name,
                    up_time=self.gen_dict[name]["min_up_time"],
                    down_time=self.gen_dict[name]["min_down_time"],
                )

        return
    

    def _populate_multiperiod_model_ramping(self, m, commodity, ramping_names=None):
        """
        Populate the multiperiod model for the price-taker model. Add the startup/shutdown constraints.

        Args:
            m: scenario pyomo model instance.
            commodity: str, name of the commodity on the model the capacity constraints.

        Returns:
            None
        """        
        # add ramping constraints
        if ramping_names:
            for name in ramping_names:
                m.add_ramping_limits(
                    op_block_name=name,
                    commodity=commodity,
                    capacity=getattr(getattr(m, f"gen_design_{name}"), "capacity"),
                    startup_rate=self.gen_dict[name]["min_p"]/self.gen_dict[name]["max_p"],
                    shutdown_rate=self.gen_dict[name]["min_p"]/self.gen_dict[name]["max_p"],
                    rampdown_rate=min(self.gen_dict[name]["ramp"], 1),
                    rampup_rate=min(self.gen_dict[name]["ramp"], 1),
                )

        return


    def default_weight_rule(self):
        return 1/len(self.set_scenarios)


    def generate_scenario_model_list(self,
                                     initial_state,
                                     LMP_data, 
                                     design_func_dict,
                                     design_params,
                                     flowsheet_func,
                                     flowsheet_options,
                                     commodity,
                                     revenue_streams=["elec_revenue"],
                                     operational_costs=None, 
                                     corporate_tax_rate=0,
                                     weight_rule=default_weight_rule,
                                     capacity_names=None,
                                     startup_names=None,
                                     ramping_names=None,
                                     ):
        """
        Returns a list containing the scenario models.
        This enables user to customize the scenario models before they become a part of the stochastic program.
        
        """
        # set of the scenarios
        self.set_scenarios = RangeSet(self.scenario)

        # build block for scenarios
        self.scenarios = Block(self.set_scenarios)

        # set of the plannig horizon, the horizon has nonantipativity constraints.
        self.set_planning_horizon = RangeSet(self.planning_horizon)

        # the weight rule can be defined as an external function.
        self.scenario_weight = Param(self.set_scenarios, rule=weight_rule)

        scenario_model_list = []

        for s in self.set_scenarios:
            # build the scenario model, we need to pass the LMP data for each scenario
            _logger.info(f"Building scenario {s} model.")

            scenario_model = self.build_PT_model(
                LMP_data=LMP_data[s-1],
                design_func_dict=design_func_dict,
                design_params=design_params,
                flowsheet_func=flowsheet_func,
                flowsheet_options=flowsheet_options,
            )
            
            # populate the scenario model
            # adding capacity limit constraints
            if capacity_names == None:
                capacity_names = [self.gen_dict[key]["name"] for key in list(self.gen_dict.keys())]
            self._populate_multiperiod_model_capacity(scenario_model, commodity, capacity_names=capacity_names)

            # adding startup_shutdown constraints
            if startup_names == None:
                startup_names = [self.gen_dict[key]["name"] for key in list(self.gen_dict.keys())]
            self._populate_multiperiod_model_startup_shutdown(scenario_model, startup_names=startup_names)

            # adding ramping constraints
            if ramping_names == None:
                ramping_names = [self.gen_dict[key]["name"] for key in list(self.gen_dict.keys())]
            self._populate_multiperiod_model_ramping(scenario_model, commodity, ramping_names=ramping_names)
            
            # initialize the scenario model, if it is empty, we just skip it. 
            if initial_state:
                for key in initial_state.keys():
                    _logger.info(f"Initialize scenario model {s}.")
                    self._initialize_scenario_model(scenario_model, initial_state[key], skip=self._skip_initialization)

            # add the cashflow for each scenario
            scenario_model.add_hourly_cashflows(
                revenue_streams=revenue_streams,
                operational_costs=operational_costs,
            )
            # add the overall cashflows, since rolling horizon is a optimization for days or weeks, we do not want to include capex.
            scenario_model.add_overall_cashflows(corporate_tax_rate=corporate_tax_rate)

            # make sure the capex is 0 for rolling horizon optimization
            scenario_model.cashflows.capex.set_value(0)

            scenario_model_list.append(scenario_model)

        return scenario_model_list



    def build_stochastic_PT_model(self, scenario_model_list, nonanti_varnames):
        """
        Build the stochastic price-taker model

        Args:
            initial_state: dict, the initial state of the model.
            LMP_data: list, the LMP data for the scenarios, generated from forecaster, with shape (self.scenario, self.horizon).
            flowsheet_func: function, the function to build the flowsheet model.
            flowsheet_options: dict, the options for the flowsheet model.
            nonanti_varnames: list, the variable names that need to be nonantipative.
            weight_rule: function, the weight rule for the scenarios, default is 1/number of scenarios.
        
        Returns:
            None
        """
        if len(scenario_model_list) != len(self.set_scenarios):
            raise ValueError("The scenario_list length does not match the scenarios provided")
        
        for s, scenario_model in zip(self.set_scenarios, scenario_model_list):
            _logger.info(f"Building scenario {s} model.")
            # transfer attributes from the scenario model to the scenario[s]
            self.scenarios[s].transfer_attributes_from(scenario_model.clone())

        # add nonantipativity constraints
        for var_name in nonanti_varnames:
            for s in self.set_scenarios:
                self._add_nonantipativity_constraints(s, var_name)

        return


    def _add_nonantipativity_constraints(self, s, var_name):
        """
        Add nonantipativity constraints.

        Args:
            s: int, the scenario number. 
            var_name: str, the variable name that needs to be nonantipative.
        
        Returns:
            None
        """
        # nonantipativity constraints at scenario[s] == nonantipativity constraints at scenario[1]
        nonantipativity_vars_at_1 = self._get_operation_vars(1, var_name)
        def _rule_nonantipativity_constraints(_, d, t):
            if s == 1:
                return Constraint.Skip
            nonantipativity_vars = self._get_operation_vars(s, var_name)

            return nonantipativity_vars[d][t] == nonantipativity_vars_at_1[d][t]

        setattr(
            self, 
            f"Constraint_nonantipativity_{var_name}_" + str(s),
            Constraint(self.scenarios[1].set_days, self.set_planning_horizon, rule=_rule_nonantipativity_constraints)
        )

        _logger.info(
            f"Setting nonantipativity constraints for {var_name} for scenario {s}."
        )

        return


    def set_objective_function(self):
        """
        Set the objective function of the rolling horizon price taker model. 
        
        Args:
            None

        Returns:
            None
        """

        # Here, maximize the npv = maximize the profit because we do not consider capex.
        self.expected_profit = Expression(expr = sum(self.scenario_weight[s] * self.scenarios[s].cashflows.npv for s in self.scenarios))
        self.obj = Objective(expr=self.expected_profit, sense=maximize)

        return
    

    def _initialize_scenario_model(self, scenario_model, initial_state, external_function=None, skip=False, *args, **kwargs):
        """
        Initialize the multiperiod model based on the results of the previous optimization.
        Consider the following initial states:
            1. the unit commitment status, if the generator is on or off.
            2. the minimum up time and down time.
            3. if there is a storage, the storage state of charge.
        This function is called before transferring the attributes from the scenario model to the scenario[s].
        So the functions from the PriceTaker class can be used to set the initial state.

        Args:
            scenario_model: the scenario model to be initialized.
            initial_state: dict, the initial state of the model.
            skip: bool, if initialize the mode.

        Returns:
            None.
        """
        if skip:
            _logger.info("Skipping the initialization of the scenario model.")
            return
        
        if external_function:
            # if an external function is provided, we use it to initialize the model.
            external_function(scenario_model, initial_state, *args, **kwargs)
        
        else:
            up_time = initial_state["up_time"]
            down_time = initial_state["down_time"]

            if up_time == 0 and down_time == 0:
                # if the up time and down time are both 0, the initial state is wrong.
                raise ValueError("The initial state is not valid, both up time and down time are 0.")
            
            if down_time * up_time != 0:
                # if both up time and down time are not 0, the initial state is wrong.
                raise ValueError("The initial state is not valid, both up time and down time are not 0.")
            # get the operation blocks for scenario, the _get_operation_blocks function is from the PriceTaker class. 
            op_blks = scenario_model._get_operation_blocks(initial_state["name"], ["startup", "shutdown", "op_mode"])

            def forced_on_rule(_, d, t):
                
                if time_need_to_stay_on == 0 or t > time_need_to_stay_on:
                    return Constraint.Skip
                
                else: 
                    return op_blks[d][t].op_mode == 1
                
            def forced_off_rule(_, d, t):

                if time_need_to_stay_off == 0 or t > time_need_to_stay_off:
                    return Constraint.Skip

                else:
                    return op_blks[d][t].op_mode == 0

            if down_time > 0:
                # if the down time is greater than 0, the generator is off.
                # constraint the first x_hour to be off where x = max(min_down_time - down_time, 0).
                # The min_down_time should not exceed the horizon length.
                time_need_to_stay_off = min(max(initial_state["min_down_time"] - down_time, 0), self.horizon)
                scenario_model.initial_state_constraints = Constraint(
                    scenario_model.set_days,
                    scenario_model.set_time,
                    rule=forced_off_rule,
                )

            if up_time > 0:
                # if the up time is greater than 0, the generator is on.
                # constraint the first x_hour to be on where x = max(min_up_time - up_time, 0).
                # The min_up_time should not exceed the horizon length.
                time_need_to_stay_on = min(max(initial_state["min_up_time"] - up_time, 0), self.horizon)
                scenario_model.initial_state_constraints = Constraint(
                    scenario_model.set_days,
                    scenario_model.set_time,
                    rule=forced_on_rule,
                )
        
        return
    

    def _get_startup_shutdown_states(self, op_block_name):
        """
        Get the number of startups for the given operational block.

        Args:
            op_block_name: str, the name of the operational block.

        Returns:
            num_startups: int, the number of startups.
        """
        # get the operation blocks for scenario 1
        op_blocks = self._get_operation_blocks(1, op_block_name, ["startup", "shutdown", "op_mode"])
        
        # get the number of startups, shudowns and op_mode for the planning horizon
        startups = {
            d:{t: value(op_blocks[d][t].startup) for t in self.set_planning_horizon}
              for d in self.scenarios[1].set_days
            }
        shutdowns = {
            d:{t: value(op_blocks[d][t].shutdown) for t in self.set_planning_horizon}
              for d in self.scenarios[1].set_days
            }
        
        op_mode = {
            d:{t: value(op_blocks[d][t].op_mode) for t in self.set_planning_horizon}
              for d in self.scenarios[1].set_days
            }
        
        return startups, shutdowns, op_mode


    def _get_up_down_time(self, op_block_name):
        """
        Calculate the up time and down time for scenario by the end of planning horizon.

        Args:
            op_block_name: str, the name of the operational block.

        Returns:
            up_time: int, the minimum up time.
            down_time: int, the minimum down time.
        """
        # get the operation blocks for scenario 1
        startups, shutdowns, op_mode = self._get_startup_shutdown_states(op_block_name)
        
        # The set of d should be {1}, so here we make the opmode as a list
        op_mode_list = [op_mode[1][t] for t in self.set_planning_horizon]
        # startups_sum = sum([startups[1][t] for t in self.set_planning_horizon])
        # shutdowns_sum = sum([shutdowns[1][t] for t in self.set_planning_horizon])

        down_time = 0
        up_time = 0

        # when the last hour is off, we need to count the down time
        if op_mode_list[-1] == 0: 
            # count from the end of the list
            for i in reversed(op_mode_list):
                # if the operation mode is off, we count the downtime
                if not i:
                    down_time += 1
                # if the operation mode is on, we stop counting
                else:
                    break
        # when the last hour is on, we need to count the up time
        else:
            # count from the end of the list
            for i in reversed(op_mode_list):
                # if the operation mode is on, we count the uptime
                if i:
                    up_time += 1
                # if the operation mode is off, we stop counting
                else:
                    break
        
        return up_time, down_time


    def report_final_state(self):
        """
        Report the final state of the model.

        Args:
            None
        
        Returns:
            final_state: dict, this is used as the initial_state for the next optimization
        """
        final_state = {}
        for key in self.gen_dict.keys():
            final_state[key] = {}
            # Because of the nonantipativity constraints, we only need to report the state of scenario 1.
            up_time, down_time = self._get_up_down_time(self.gen_dict[key]["name"])
            final_state[key]["name"] = self.gen_dict[key]["name"]
            final_state[key]["up_time"] = up_time
            final_state[key]["down_time"] = down_time
            final_state[key]["min_down_time"] = self.gen_dict[key]["min_down_time"]
            final_state[key]["min_up_time"] = self.gen_dict[key]["min_up_time"]

        return final_state
    

    def calculate_actual_revenue(self, actual_price, external_func=None, *args, **kwargs):
        """
        Calculate the actual revenue based on the actual price and the power output.

        Args:
            actual_price: list, the actual price for each time period.
            external_func: function, an external function to calculate the revenue, if not provided, the default calculation will be used.

        Returns:
            actual_revenue: float, the actual revenue.
        """
        # get the power output from the model
        if not external_func:
            # if no external function is provided, we use the default calculation.
            for key in self.gen_dict.keys():
                op_blks = self._get_operation_blocks(1, self.gen_dict[key]['name'], ["power", "startup", "shutdown"])
                
                # calculate the actual revenue, the actual price is indexed from 0.
                actual_elec_revenue = sum(actual_price[t-1] * value(op_blks[1][t].power) for t in self.set_planning_horizon)
                actual_vom = sum(value(op_blks[1][t].op_mode)*(self.gen_dict[key]["cost_curve"]["slope"] * value(op_blks[1][t].power) + self.gen_dict[key]["cost_curve"]["intercept"]) for t in self.set_planning_horizon)

                # calculate the startup and shutdown costs
                actual_startup_cost = sum(value(op_blks[1][t].startup) * self.gen_dict[key]["fuel_p"] * self.gen_dict[key]["start_heat_cold"] for t in self.set_planning_horizon)
                actual_shutdown_cost = sum(value(op_blks[1][t].shutdown) * self.gen_dict[key]["fuel_p"] * 0 for t in self.set_planning_horizon)

                # calculate the actual revenue
                actual_profit = actual_elec_revenue - actual_vom - actual_startup_cost - actual_shutdown_cost
        
        else:
            # if an external function is provided, we use it to calculate the revenue.
            actual_profit = external_func(actual_price, *args, **kwargs)

        return float(actual_profit)


    def record_solution(self, soln, actual_price, external_func_record=None, *args, **kwargs):
        """
        record the results from solved model.
        """
        results = {}
        results["TerminationCondition"] = str(soln.solver.termination_condition)
        results["SolverStatus"] = str(soln.solver.status)

        for key in self.gen_dict.keys():
            if not external_func_record:
                results["ObjectiveValue"] = value(self.obj)
                results["ActualProfit"] = self.calculate_actual_revenue(actual_price)
                # if no external function is provided, we use the default calculation.
                operation_var_name = ["power", "startup", "shutdown", "op_mode"]
                results[key] = {}
                # record the objective value
                for var_name in operation_var_name:
                    pyomo_blks = self._get_operation_blocks(1, self.gen_dict[key]['name'], [var_name])
                    results[key][f"OperationVariables_{var_name}"] = {
                        d: {t: value(getattr(pyomo_blks[d][t], var_name)) for t in self.set_planning_horizon}
                        for d in self.scenarios[1].set_days
                    }

                results["IdeaProfit"] = sum(self.scenario_weight[s] * value(self.scenarios[s].period[1, t].net_hourly_cash_inflow) for s in self.set_scenarios for t in self.set_planning_horizon)

            else:
                # if an external function is provided, we use it to record the results.
                results = external_func_record(self, soln, actual_price, *args, **kwargs)

        return results


class RollinghorizonPriceTaker:
    """
    Rolling horizon stochastic price-taker optimization.
    
    Args:
        solver: dictionary, the solver for solving the optimization or simulation problem.
        solver_options: dictionary, the solver options.
    """
    def __init__(self, 
                 gen_dict: dict,
                 solver: Optional[str]="gurobi",
                 solver_options: Optional[dict]={},
                 initial_on: Optional[bool]=True,
                 ):
        
        self.gen_dict = gen_dict
        self.solver = SolverFactory(solver)
        self.solver_options = solver_options
        self.initial_on = initial_on

        # generate a default initial state
        self.default_initial_state = {}
        for gen in self.gen_dict.keys():
            self.default_initial_state[gen] = {}
            self.default_initial_state[gen]["name"] = gen
            # the next line allows the generator to freely shutdown/startup at the first period.
            if self.initial_on:
                self.default_initial_state[gen]["up_time"] = gen_dict[gen]["min_up_time"] + 1
                self.default_initial_state[gen]["down_time"] = 0
            else:
                self.default_initial_state[gen]["up_time"] = 0
                self.default_initial_state[gen]["down_time"] = gen_dict[gen]["min_down_time"] + 1
                    

    def run_rolling_horizon(self,
                            pt_building_func: Callable,
                            forecaster: Callable,
                            initial_state: Optional[dict]=None,
                            start_period: Optional[int]=0,
                            end_period: Optional[int]=None,
                            *args, **kwargs
                            ):
        """
        Run the rolling horizon optimization. 

        Args:
            initial_state: dictionary, the initial state at the beginning of rolling horizon optimization.

        Returns:
            results_dict: dictionary, keys are periods, values are results.
        """
        if end_period == None:
            # set the end period equal to the length of lmp signals
            end_period = len(forecaster.reshaped_signals)

        if initial_state == None:
            # if no initial state, use the default initial state
            initial_state = self.default_initial_state
        
        if end_period - start_period < 0:
            # check the valid start and end period
            raise ValueError("The end_period should be equal or greater than the starting period.")

        # set up dictionarys to store the results.
        results_dict = {}

        # for each period, build and solve the stochastic price-taker model, and record the results.
        for i in range(start_period, end_period):
            _logger.info(f"Building price-taker optimization for period {i}.")
            # each i is the index of the period (e.g., day). Forecast the prices at that day.
            lmp_data = forecaster.forecast_prices(pointer=i)
            
            m = pt_building_func(initial_state=initial_state, *args, **kwargs)

            # solve the stochastic price-taker model
            soln = self.solver.solve(m, tee=True, options=self.solver_options)

            _logger.info("Solver status:", soln.solver.status)
            _logger.info("Termination condition:", soln.solver.termination_condition)
            _logger.info("Objective value:", value(m.obj))

            # record results
            actual_price = forecaster.fetch_original_signal(pointer=i)
            results_dict[i] = m.record_solution(soln, actual_price)
            
            initial_state = m.report_final_state()

        return results_dict[i]