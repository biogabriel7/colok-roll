"""
Automatic parameter selection for background subtraction.

This module provides the zoom-search algorithm for automatically finding
optimal background subtraction parameters for a given image and channel.
"""

from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from ..constants import (
    IMPROVEMENT_EPS,
    MAX_EVALS_PER_METHOD,
    MIN_INT_STEP,
    MIN_FLOAT_STEP,
    TOPN_COARSE,
    TESTED_VALUES_MAX,
    REFINEMENT_SHRINK_FACTOR,
    AUTO_CACHE_SCORE_TOLERANCE,
    DEFAULT_AUTO_WEIGHTS,
)

if TYPE_CHECKING:
    from .scorer import BackgroundScorer

logger = logging.getLogger(__name__)

# Cache filename
AUTO_CACHE_FILENAME = "auto_bg_cache.json"

# Auto methods available
AUTO_METHODS: Tuple[str, ...] = ("rolling_ball", "gaussian", "two_stage", "morphological")

# Default coarse search seeds
DEFAULT_COARSE_SEEDS: Dict[str, Dict[str, List[Any]]] = {
    "rolling_ball": {"radius": [16, 32, 64, 96, 128], "light_background": [False, True]},
    "gaussian": {"sigma": [5.0, 20.0, 50.0, 100.0, 200.0]},
    "two_stage": {
        "sigma_stage1": [5.0, 15.0, 30.0, 60.0],
        "radius_stage2": [16, 32, 64, 96],
        "light_background": [False, True],
    },
    "morphological": {"size": [5, 15, 30, 60, 90], "shape": ["disk", "square"]},
}

# Refinement points per method
REFINE_POINTS: Dict[str, int] = {
    "rolling_ball": 7, 
    "gaussian": 8, 
    "two_stage": 5, 
    "morphological": 7
}
SECOND_REFINE_POINTS: Dict[str, int] = {
    "rolling_ball": 5, 
    "gaussian": 6, 
    "two_stage": 4, 
    "morphological": 5
}


class AutoModeSelector:
    """
    Automatic parameter selection using zoom-search algorithm.
    
    This class implements a coarse-to-fine search strategy:
    1. Coarse grid search over predefined parameter ranges
    2. Zoom into the best candidates with finer grids
    3. Cache results for similar channels to avoid redundant computation
    """
    
    def __init__(
        self,
        scorer: "BackgroundScorer",
        cache_path: Optional[Path] = None,
        weights: Tuple[float, float, float, float] = DEFAULT_AUTO_WEIGHTS,
    ):
        """
        Initialize the auto selector.
        
        Args:
            scorer: BackgroundScorer instance for evaluating candidates
            cache_path: Path to cache file for storing results
            weights: Scoring weights (bg_removal, contrast, detail, zero_penalty)
        """
        self.scorer = scorer
        self.cache_path = cache_path or Path(__file__).with_name(AUTO_CACHE_FILENAME)
        self.weights = weights
        self.logger = logging.getLogger(__name__)
    
    def normalize_channel_key(self, channel_name: Optional[str]) -> str:
        """Normalize channel name for cache lookup."""
        return (channel_name or "unknown").strip().lower()
    
    def load_cache(self) -> Dict[str, Any]:
        """Load the auto-selection cache from disk."""
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load auto cache: {e}")
        return {}
    
    def save_cache(self, cache: Dict[str, Any]) -> None:
        """Save the auto-selection cache to disk."""
        try:
            with self.cache_path.open("w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save auto cache: {e}")
    
    def merge_seed_ranges(
        self, method: str, channel_entry: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """
        Merge default seeds with cached best values for a method.
        
        If we have a cached result, we expand the search around it.
        """
        base = dict(DEFAULT_COARSE_SEEDS.get(method, {}))
        cached_params = channel_entry.get("params", {})
        
        for key, default_vals in base.items():
            if key not in cached_params:
                continue
            cached_val = cached_params[key]
            
            # For numeric values, add nearby values
            if isinstance(cached_val, (int, float)) and default_vals:
                # Find bracket around cached value
                new_vals = set(default_vals)
                new_vals.add(cached_val)
                
                # Add interpolated values
                sorted_vals = sorted(new_vals)
                idx = sorted_vals.index(cached_val)
                if idx > 0:
                    mid = (sorted_vals[idx - 1] + cached_val) / 2
                    if isinstance(default_vals[0], int):
                        new_vals.add(int(mid))
                    else:
                        new_vals.add(float(mid))
                if idx < len(sorted_vals) - 1:
                    mid = (cached_val + sorted_vals[idx + 1]) / 2
                    if isinstance(default_vals[0], int):
                        new_vals.add(int(mid))
                    else:
                        new_vals.add(float(mid))
                
                base[key] = sorted(new_vals)
        
        return base
    
    def refine_numeric_values(
        self,
        current: Any,
        neighbors: List[Any],
        num_points: int,
        is_int: bool,
    ) -> List[Any]:
        """
        Generate refined values around a current best value.
        
        Args:
            current: Current best value
            neighbors: Neighboring values from coarse search
            num_points: Number of refinement points
            is_int: Whether values should be integers
            
        Returns:
            List of refined values to evaluate
        """
        if not neighbors:
            return [current]
        
        # Find the range to refine within
        sorted_neighbors = sorted(set(neighbors + [current]))
        idx = sorted_neighbors.index(current)
        
        lo = sorted_neighbors[max(0, idx - 1)]
        hi = sorted_neighbors[min(len(sorted_neighbors) - 1, idx + 1)]
        
        # Shrink the range
        span = (hi - lo) * REFINEMENT_SHRINK_FACTOR
        center = current
        lo = center - span / 2
        hi = center + span / 2
        
        if is_int:
            lo = max(1, int(lo))
            hi = max(lo + 1, int(hi))
            step = max(MIN_INT_STEP, (hi - lo) // num_points)
            values = list(range(lo, hi + 1, step))
        else:
            step = max(MIN_FLOAT_STEP, (hi - lo) / num_points)
            values = [lo + i * step for i in range(num_points + 1)]
        
        # Always include current
        if current not in values:
            values.append(current)
        
        return sorted(set(values))[:num_points + 3]  # Cap the number
    
    def build_refined_ranges(
        self,
        method: str,
        best_params: Dict[str, Any],
        tested_values: Dict[str, List[Any]],
        refine_points: int,
    ) -> Dict[str, List[Any]]:
        """
        Build refined parameter ranges around the best parameters.
        
        Args:
            method: Background subtraction method
            best_params: Current best parameters
            tested_values: Values that were tested in the coarse search
            refine_points: Number of refinement points per parameter
            
        Returns:
            Dictionary of parameter names to refined value lists
        """
        refined = {}
        
        for key, val in best_params.items():
            neighbors = tested_values.get(key, [])
            
            if isinstance(val, bool):
                refined[key] = [val]
            elif isinstance(val, str):
                refined[key] = [val]
            elif isinstance(val, int):
                refined[key] = self.refine_numeric_values(val, neighbors, refine_points, True)
            elif isinstance(val, float):
                refined[key] = self.refine_numeric_values(val, neighbors, refine_points, False)
            else:
                refined[key] = [val]
        
        return refined
    
    def expand_param_grid(self, grid_spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Expand a parameter grid specification into a list of parameter combinations.
        
        Args:
            grid_spec: Dictionary mapping parameter names to lists of values
            
        Returns:
            List of parameter dictionaries
        """
        if not grid_spec:
            return [{}]
        
        keys = list(grid_spec.keys())
        values = [grid_spec[k] for k in keys]
        
        combos = []
        for combo in product(*values):
            combos.append(dict(zip(keys, combo)))
        
        return combos
    
    def evaluate_candidates(
        self,
        image: np.ndarray,
        method: str,
        candidates: List[Dict[str, Any]],
        run_single_fn: Callable,
        is_negative_control: bool = False,
        max_evals: int = MAX_EVALS_PER_METHOD,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a list of parameter candidates and return scored results.
        
        Args:
            image: Input 3D image
            method: Background subtraction method
            candidates: List of parameter dictionaries to evaluate
            run_single_fn: Function to run subtraction with given params
            is_negative_control: Whether this is a negative control channel
            max_evals: Maximum number of evaluations
            
        Returns:
            List of result dictionaries sorted by score (best first)
        """
        results = []
        
        for i, params in enumerate(candidates[:max_evals]):
            try:
                corrected = run_single_fn(image, method, params)
                
                if is_negative_control:
                    score, components = self.scorer.score_volume_negative_control(
                        image, corrected
                    )
                else:
                    score, components = self.scorer.score_volume(
                        image, corrected, self.weights
                    )
                
                results.append({
                    "method": method,
                    "params": params,
                    "score": score,
                    "components": components,
                })
                
            except Exception as e:
                self.logger.debug(f"Candidate {params} failed: {e}")
                continue
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def collect_tested_values(
        self, scores: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """
        Collect all tested values from scored results for refinement.
        
        Args:
            scores: List of scored result dictionaries
            
        Returns:
            Dictionary mapping parameter names to lists of tested values
        """
        tested: Dict[str, List[Any]] = {}
        
        for entry in scores:
            params = entry.get("params", {})
            for k, v in params.items():
                if k not in tested:
                    tested[k] = []
                if v not in tested[k]:
                    tested[k].append(v)
        
        # Sort numeric values
        for k in tested:
            vals = tested[k]
            if vals and isinstance(vals[0], (int, float)):
                tested[k] = sorted(vals)
        
        return tested
    
    def zoom_search_method(
        self,
        image: np.ndarray,
        method: str,
        run_single_fn: Callable,
        channel_entry: Optional[Dict[str, Any]] = None,
        is_negative_control: bool = False,
    ) -> Tuple[Dict[str, Any], float, Dict[str, float]]:
        """
        Perform zoom search for optimal parameters for a single method.
        
        Args:
            image: Input 3D image
            method: Background subtraction method
            run_single_fn: Function to run subtraction
            channel_entry: Optional cached entry for this channel
            is_negative_control: Whether this is a negative control
            
        Returns:
            Tuple of (best_params, best_score, score_components)
        """
        self.logger.info(f"Zoom search for method: {method}")
        
        # Get coarse seeds, potentially enhanced by cache
        if channel_entry and channel_entry.get("method") == method:
            coarse_seeds = self.merge_seed_ranges(method, channel_entry)
        else:
            coarse_seeds = dict(DEFAULT_COARSE_SEEDS.get(method, {}))
        
        # Coarse search
        coarse_candidates = self.expand_param_grid(coarse_seeds)
        self.logger.info(f"  Coarse search: {len(coarse_candidates)} candidates")
        
        coarse_results = self.evaluate_candidates(
            image, method, coarse_candidates, run_single_fn, 
            is_negative_control, max_evals=MAX_EVALS_PER_METHOD,
        )
        
        if not coarse_results:
            return {}, 0.0, {}
        
        # First refinement
        top_n = coarse_results[:TOPN_COARSE]
        tested_values = self.collect_tested_values(coarse_results)
        
        all_refined_results = []
        for top_entry in top_n:
            best_params = top_entry["params"]
            refine_points = REFINE_POINTS.get(method, 5)
            refined_ranges = self.build_refined_ranges(
                method, best_params, tested_values, refine_points
            )
            
            refined_candidates = self.expand_param_grid(refined_ranges)
            refined_results = self.evaluate_candidates(
                image, method, refined_candidates, run_single_fn,
                is_negative_control, max_evals=MAX_EVALS_PER_METHOD // 2,
            )
            all_refined_results.extend(refined_results)
        
        if all_refined_results:
            all_refined_results.sort(key=lambda x: x["score"], reverse=True)
            best = all_refined_results[0]
            
            # Check if we improved enough to warrant second refinement
            if best["score"] > coarse_results[0]["score"] + IMPROVEMENT_EPS:
                # Second refinement
                tested_values = self.collect_tested_values(all_refined_results)
                second_refine_points = SECOND_REFINE_POINTS.get(method, 4)
                second_ranges = self.build_refined_ranges(
                    method, best["params"], tested_values, second_refine_points
                )
                
                second_candidates = self.expand_param_grid(second_ranges)
                second_results = self.evaluate_candidates(
                    image, method, second_candidates, run_single_fn,
                    is_negative_control, max_evals=MAX_EVALS_PER_METHOD // 4,
                )
                
                if second_results and second_results[0]["score"] > best["score"]:
                    best = second_results[0]
            
            return best["params"], best["score"], best.get("components", {})
        
        # Fallback to best coarse result
        best = coarse_results[0]
        return best["params"], best["score"], best.get("components", {})
    
    def select_best_parameters(
        self,
        image: np.ndarray,
        run_single_fn: Callable,
        channel_name: Optional[str] = None,
        is_negative_control: bool = False,
        methods: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any], float, Dict[str, float]]:
        """
        Select the best method and parameters for an image.
        
        Args:
            image: Input 3D image
            run_single_fn: Function to run subtraction
            channel_name: Channel name for cache lookup
            is_negative_control: Whether this is a negative control
            methods: Optional list of methods to try (defaults to AUTO_METHODS)
            
        Returns:
            Tuple of (best_method, best_params, best_score, components)
        """
        methods = methods or list(AUTO_METHODS)
        channel_key = self.normalize_channel_key(channel_name)
        
        # Check cache
        cache = self.load_cache()
        channel_entry = cache.get(channel_key)
        
        if channel_entry:
            cached_score = channel_entry.get("score", 0)
            if cached_score > AUTO_CACHE_SCORE_TOLERANCE:
                self.logger.info(
                    f"Found cached result for channel '{channel_key}': "
                    f"method={channel_entry.get('method')}, score={cached_score:.4f}"
                )
        
        # Run zoom search for each method
        best_overall = {
            "method": methods[0],
            "params": {},
            "score": float("-inf"),
            "components": {},
        }
        
        for method in methods:
            try:
                params, score, components = self.zoom_search_method(
                    image, method, run_single_fn, channel_entry, is_negative_control
                )
                
                self.logger.info(f"  Method {method}: score={score:.4f}")
                
                if score > best_overall["score"]:
                    best_overall = {
                        "method": method,
                        "params": params,
                        "score": score,
                        "components": components,
                    }
                    
            except Exception as e:
                self.logger.warning(f"Method {method} failed: {e}")
                continue
        
        # Update cache if we found something good
        if best_overall["score"] > AUTO_CACHE_SCORE_TOLERANCE:
            cache[channel_key] = {
                "method": best_overall["method"],
                "params": best_overall["params"],
                "score": best_overall["score"],
            }
            self.save_cache(cache)
        
        return (
            best_overall["method"],
            best_overall["params"],
            best_overall["score"],
            best_overall["components"],
        )
