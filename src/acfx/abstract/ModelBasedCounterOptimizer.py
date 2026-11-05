from abc import ABC, abstractmethod
from typing import List, Dict


class ModelBasedCounterOptimizer(ABC):
    """
        Abstract interface for model-based counterfactual candidate optimizers.

        Implementations of this interface generate a candidate instance intended
        to increase the predictive model's probability for a requested target
        class. The candidate is produced by modifying a selected set of features.

        ACFX uses these optimizers to generate initial candidate points before the
        main counterfactual optimization. The returned candidate is not, by
        itself, guaranteed to be a valid counterfactual. It is subsequently
        bounded, checked for uniqueness, and enqueued as an initial trial in the
        counterfactual-generation procedure.

        Concrete implementations are responsible for storing any model, reference
        instance, reference data, and feature bounds required by their
        optimization procedure.

        Notes
        -----
        This class is abstract and cannot be instantiated directly. Subclasses
        must implement :meth:`optimize_proba`.

        ACFX selects a non-empty subset of modifiable features and passes it to
        :meth:`optimize_proba`. Different subsets may be used when the optimizer
        is called repeatedly during model-based initial sampling.

        The optimizer is expected to return a mapping that can be interpreted as
        a complete candidate instance. In the current counterfactual-generation
        workflow, the returned mapping is clipped using the configured bounds and
        then checked for the feature names required by ``features_order``.

        The interface does not require a particular optimization algorithm.
        Implementations may use model-specific information, random search, or
        another procedure, provided that they satisfy the input and return-value
        contract defined by :meth:`optimize_proba`.
    """
    @abstractmethod
    def optimize_proba(self, target_class: int, feature_masked: List[str]) -> Dict[str, float]:
        """
        Generate a candidate that favors the requested target class.

        The method should generate feature values intended to increase the
        predictive model's probability for ``target_class``. Only features
        listed in ``feature_masked`` should be treated as candidates for
        modification. Values for the remaining features should be retained
        or reconstructed according to the concrete optimizer's stored
        reference instance or reference data.

        The method returns a feature-value mapping rather than the target
        probability itself. ACFX uses the returned mapping as a candidate
        initial point for its subsequent counterfactual optimization.

        Parameters
        ----------
        target_class : int
            Index of the class whose predicted probability the optimizer
            attempts to increase.

            The value is passed directly by the counterfactual-generation
            procedure. The concrete optimizer and its associated predictive
            model determine how this index is interpreted.

        feature_masked : list[str]
            Names of features that the optimizer may modify in the generated
            candidate.

            In the ACFX model-based sampling procedure, this argument is a
            randomly selected non-empty subset of the configured modifiable
            features. Despite the name ``feature_masked``, the supplied
            features are the ones selected for adjustment.

        Returns
        -------
        dict[str, float]
            Candidate instance represented as a mapping from feature names to
            generated feature values.

            The returned dictionary should provide values for all features
            required by the configured feature order, not only those listed
            in ``feature_masked``. This is necessary because the current ACFX
            workflow converts and validates the result as a complete
            candidate point.

        Notes
        -----
        This interface does not require the method to return the achieved
        target-class probability.

        It also does not require the returned candidate to be classified as
        ``target_class``. In the current workflow, the candidate is used to
        initialize the main optimization rather than accepted directly as a
        final counterfactual.

        The calling ACFX procedure clips returned values to the configured
        feature bounds before enqueuing the candidate. Nevertheless, concrete
        implementations should preferably generate values compatible with
        the expected feature domains.

        Exceptions raised here may be caught by the model-based sampling
        procedure. In that case, the failed candidate is skipped and the
        remaining initial-sampling process continues.
        """
        pass  # pragma: no cover - abstract
