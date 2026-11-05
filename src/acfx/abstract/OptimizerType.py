from enum import Enum

class OptimizerType(Enum):
    """
       Identifies the optimizer strategy used for counterfactual generation.

       ``OptimizerType`` determines which model-based optimizer ACFX uses to
       generate additional initial candidates for the main counterfactual
       optimization procedure.

       The optimizer type is normally configured by a concrete ``ACFX``
       subclass during ``fit``. The counterfactual-generation code then uses
       this value to select a built-in optimizer or a user-provided custom
       optimizer.

       Attributes
       ----------
       EBM : int
           Selects the optimizer for an
           ``ExplainableBoostingClassifier``.

           In the current implementation, this option creates an
           ``EBMCounterOptimizer`` using the predictive model and a one-row
           DataFrame constructed from the query instance.

       LinearAdditive : int
           Selects the optimizer for a linear additive classifier.

           In the current implementation, this option creates a
           ``LogisticRegressionCounterOptimizer`` using the predictive model,
           the query instance, and the configured feature bounds.

           When this optimizer type is used through ``ACFX.counterfactual``,
           the black-box model must expose a ``coef_`` attribute.

       Custom : int
           Selects a user-provided optimizer implementing the
           ``ModelBasedCounterOptimizer`` interface.

           The optimizer instance must be supplied separately. In the
           ``AcfxCustom`` interface, it is provided to ``fit`` and stored in
           ``self.optimizer`` before counterfactual generation.

       Notes
       -----
       The enumeration values are implementation identifiers:

       * ``OptimizerType.EBM.value`` is ``1``;
       * ``OptimizerType.LinearAdditive.value`` is ``2``;
       * ``OptimizerType.Custom.value`` is ``3``.

       The numeric values do not define an ordering, priority, or quality
       relationship between the optimizer strategies.

       The selected optimizer is used for model-based generation of initial
       candidate points when ``sampling_from_model`` is enabled and data-based
       sampling does not provide all requested initial points. The final
       counterfactual is selected by the subsequent Optuna optimization rather
       than returned directly by the model-based optimizer.

       An unsupported optimizer type causes the optimizer-selection code to
       raise ``NotImplementedError``.
       """
    EBM = 1
    LinearAdditive = 2
    Custom = 3