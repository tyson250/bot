from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import math
from .base_strategy import BaseStrategy
from datetime import datetime, timedelta

class FibonacciStrategy(BaseStrategy):
    """
    Fibonacci trading strategy.
    
    This strategy uses Fibonacci retracement and extension levels
    to identify high-probability entry and exit points.
    """
    
    def analyze(self) -> Dict:
        """Main strategy analysis method required by the BaseStrategy interface"""
        setup = self.check_for_setup()
        
        # Set trade signal based on setup validity
        if setup["valid_setup"]:
            trade_signal = setup["direction"]
        else:
            trade_signal = "none"
            
        # Get sentiment analysis
        sentiment_data = self._analyze_market_sentiment()
        
        result = {
            "trade_signal": trade_signal,
            "entry_price": setup.get("entry_price", 0),
            "stop_loss": setup.get("stop_loss", 0),
            "take_profit": setup.get("take_profit", 0),
            "setup_key": setup.get("setup_key", ""),
            "description": setup.get("description", ""),
            "type": setup.get("type", "FIBONACCI"),
            "strength": setup.get("strength", 0),
            "confluence_score": setup.get("confluence_score", 0),
            "confluence_factors": setup.get("confluence_factors", []),
            # Include sentiment and liquidity analysis 
            "sentiment": {
                "bias": sentiment_data.get("overall_bias", "neutral"),
                "score": sentiment_data.get("sentiment_score", 0),
                "confidence": sentiment_data.get("confidence", 0),
                "key_factors": sentiment_data.get("key_factors", [])
            }
        }
        
        # Add liquidity analysis if available
        if "liquidity" in setup:
            result["liquidity"] = setup.get("liquidity_analysis", {})
        
        return result
    
    def check_for_setup(self) -> Dict:
        """Check for Fibonacci-based trade setups with enhanced filtering"""
        current_price = self.trader.get_current_price()
        if current_price is None:
            return {"valid_setup": False, "trade_signal": "none"}
            
        setup = {"valid_setup": False, "trade_signal": "none"}
        
        # Get current timeframe's Fibonacci levels
        if not hasattr(self.trader, 'fib_levels') or self.trader.timeframe not in self.trader.fib_levels:
            return setup
        
        # Calculate adaptive volatility for dynamic parameters
        volatility = self._calculate_adaptive_volatility()
        
        # Get market context for improved decision making
        market_context = self._analyze_market_context()
        
        # NEW: Get sentiment analysis for enhanced market understanding
        sentiment_data = self._analyze_market_sentiment()
            
        fib_data = self.trader.fib_levels[self.trader.timeframe]
        levels = fib_data['levels']
        
        # Check for Fibonacci confluence across timeframes
        multi_tf_fib_levels = self._get_multi_timeframe_levels(current_price, volatility)
        
        # Check for harmonic patterns for stronger setups
        harmonic_patterns = self._detect_harmonic_patterns()
        
        # IMPROVEMENT: Validate harmonic patterns with additional criteria
        validated_patterns = {"bullish": [], "bearish": []}
        for pattern in harmonic_patterns.get("bullish", []):
            validation = self._validate_harmonic_pattern(pattern)
            if validation["valid"]:
                pattern["strength"] = validation["strength"]
                pattern["confidence_factors"] = validation.get("confidence_factors", [])
                validated_patterns["bullish"].append(pattern)
                
        for pattern in harmonic_patterns.get("bearish", []):
            validation = self._validate_harmonic_pattern(pattern)
            if validation["valid"]:
                pattern["strength"] = validation["strength"] 
                pattern["confidence_factors"] = validation.get("confidence_factors", [])
                validated_patterns["bearish"].append(pattern)
        
        # NEW IMPROVEMENT: Real-Time Pattern Quality Assessment
        pattern_quality = self._assess_pattern_quality(validated_patterns, market_context)
        
        # Filter out low-quality patterns based on assessment
        high_quality_patterns = {"bullish": [], "bearish": []}
        for pattern in validated_patterns.get("bullish", []):
            if pattern.get("id") in pattern_quality and pattern_quality[pattern.get("id")]["quality_score"] > 0.6:
                pattern["quality_score"] = pattern_quality[pattern.get("id")]["quality_score"]
                pattern["quality_factors"] = pattern_quality[pattern.get("id")]["quality_factors"]
                high_quality_patterns["bullish"].append(pattern)
                
        for pattern in validated_patterns.get("bearish", []):
            if pattern.get("id") in pattern_quality and pattern_quality[pattern.get("id")]["quality_score"] > 0.6:
                pattern["quality_score"] = pattern_quality[pattern.get("id")]["quality_score"]
                pattern["quality_factors"] = pattern_quality[pattern.get("id")]["quality_factors"]
                high_quality_patterns["bearish"].append(pattern)
                
        # IMPROVEMENT: Get high-quality Fibonacci levels with clustering and volume
        fib_support_clusters, fib_resistance_clusters = self._find_fib_clusters_with_volume(levels, volatility)
        
        # Find dynamic Fibonacci compression zones (more reliable)
        compression_zones = self._detect_fibonacci_compression_zones(volatility)
        
        # NEWEST IMPROVEMENT: Detect Wyckoff springs and upthrusts
        wyckoff_patterns = self._detect_wyckoff_patterns(volatility)
        
        # If we have compression zones, prioritize them over regular clusters
        if compression_zones.get("support_zone") and not fib_support_clusters:
            fib_support_clusters.append((compression_zones["support_zone"], compression_zones["support_strength"]))
            
        if compression_zones.get("resistance_zone") and not fib_resistance_clusters:
            fib_resistance_clusters.append((compression_zones["resistance_zone"], compression_zones["resistance_strength"]))
        
        # If no clusters found, try standard levels
        if not fib_support_clusters and not fib_resistance_clusters:
            # Find nearest Fibonacci levels with standard approach
            nearest_support, nearest_resistance = self._find_nearest_levels(current_price, levels)
            
            if nearest_support is not None:
                fib_support_clusters.append((nearest_support, 1.0))  # Add with base strength
                
            if nearest_resistance is not None:
                fib_resistance_clusters.append((nearest_resistance, 1.0))  # Add with base strength
        
        # Get market structure from primary timeframe
        market_structure = self.trader.market_structure[self.trader.timeframe]
        trend = market_structure.get("trend", "neutral")
        
        # Get price action context for better timing
        price_action = self._analyze_price_action()
        
        # NEW IMPROVEMENT: Apply pattern quality assessment to price action
        price_action = self._enhance_price_action_with_quality(price_action)
        
        # NEW: Adjust decision thresholds based on market sentiment
        volatility_multiplier = 1.0
        if sentiment_data["overall_bias"] == "bullish" and sentiment_data["confidence"] > 0.5:
            # More aggressive on support with bullish sentiment
            volatility_multiplier = 1.0 + min(0.5, sentiment_data["sentiment_score"] * sentiment_data["confidence"]) 
        elif sentiment_data["overall_bias"] == "bearish" and sentiment_data["confidence"] > 0.5:
            # More aggressive on resistance with bearish sentiment
            volatility_multiplier = 1.0 + min(0.5, abs(sentiment_data["sentiment_score"]) * sentiment_data["confidence"])
            
        # Check for potential setups using the highest quality cluster points
        if fib_support_clusters and (trend != "downtrend" or market_context["structural_inflection"]):
            # Sort support clusters by quality (descending)
            fib_support_clusters.sort(key=lambda x: x[1], reverse=True)
            best_support = fib_support_clusters[0][0]
            support_quality = fib_support_clusters[0][1]
            
            # IMPROVEMENT: Check historical validation of support level
            support_validation_score = self._validate_historical_fib_reactions(best_support, "support")
            support_quality *= support_validation_score
            
            # NEW: Check liquidity at this support level
            liquidity_at_support = self._analyze_liquidity(best_support, "buy")
            
            # Skip setups with extremely poor liquidity
            if not liquidity_at_support["sufficient_liquidity"] and liquidity_at_support["liquidity_score"] < 0.2:
                pass  # Skip this setup due to liquidity concerns
            else:
                # Check if price is near the support level (adjusted by sentiment-based volatility multiplier)
                if abs(current_price - best_support) < volatility * (1.5 if trend == "uptrend" else 2.5) * volatility_multiplier:
                    # Skip if momentum is strong downward and no bullish reversal signal
                    if market_context["momentum"] < -volatility * 3 and not price_action["bullish_reversal"]:
                        pass  # Skip this setup
                    else:
                        # IMPROVEMENT: Check entry timing precision
                        entry_timing = self._check_entry_timing("buy", current_price, best_support)
                        
                        # Calculate adaptive stop based on volatility and market conditions
                        stop_buffer = volatility * (
                            1.2 if trend == "uptrend" else
                            2.0 if trend == "downtrend" else
                            1.5
                        )
                        
                        # Additional buffer for high volatility or against trend
                        if market_context["volatility_regime"] == "high":
                            stop_buffer *= 1.3
                        
                        # Check for optimal stop placement based on recent swing lows
                        optimal_stop = self._find_optimal_stop_level("buy", best_support, stop_buffer)
                        stop_loss = optimal_stop if optimal_stop else best_support - stop_buffer
                        
                        # Find optimal take profit level
                        take_profit = self._find_optimal_target(
                            "bullish", 
                            current_price, 
                            volatility, 
                            market_context,
                            fib_resistance_clusters,
                            high_quality_patterns.get("bullish", [])  # Use validated and quality-filtered patterns
                        )
                        
                        if take_profit:
                            # Calculate reward-to-risk ratio
                            reward = take_profit - current_price
                            risk = current_price - stop_loss
                            
                            # Calculate required R:R based on market conditions
                            min_rr = self._calculate_required_rr("bullish", trend, market_context)
                            
                            # NEW: Adjust minimum required R:R based on sentiment
                            if sentiment_data["overall_bias"] == "bullish" and sentiment_data["confidence"] > 0.5:
                                min_rr *= max(0.7, 1.0 - (sentiment_data["sentiment_score"] * 0.3))
                            elif sentiment_data["overall_bias"] == "bearish" and sentiment_data["confidence"] > 0.5:
                                min_rr *= min(1.5, 1.0 + (abs(sentiment_data["sentiment_score"]) * 0.5))
                            
                            if reward >= risk * min_rr:
                                # Calculate dynamic setup strength
                                setup_strength = self._calculate_setup_strength(
                                    "buy",
                                    support_quality,  # Using the historically validated quality
                                    trend, 
                                    price_action, 
                                    market_context,
                                    multi_tf_fib_levels
                                )
                                
                                # IMPROVEMENT: Apply entry timing precision if available
                                if entry_timing["entry_ready"]:
                                    setup_strength *= entry_timing["confidence"]
                                
                                # Enhance setup strength if we have a harmonic pattern
                                for pattern in high_quality_patterns.get("bullish", []):
                                    if abs(pattern["price"] - best_support) < volatility * 2:
                                        # NEW: Use quality score for strength adjustment
                                        quality_boost = pattern.get("quality_score", 1.0) * 0.3
                                        setup_strength *= 1.3 + quality_boost
                                        break
                                        
                                # NEW: Enhance setup strength if we have a Wyckoff spring pattern
                                if wyckoff_patterns.get("springs"):
                                    for spring in wyckoff_patterns["springs"]:
                                        if abs(spring["level"] - best_support) < volatility * 2:
                                            setup_strength *= spring.get("strength_multiplier", 1.5)
                                        break
                                
                                # Assign setup type based on confluence factors
                                setup_type = "FIBONACCI"
                                if compression_zones.get("support_zone") and abs(compression_zones["support_zone"] - best_support) < volatility:
                                    setup_type = "FIB_COMPRESSION"
                                elif any(abs(p["price"] - best_support) < volatility * 2 for p in high_quality_patterns.get("bullish", [])):
                                    setup_type = "HARMONIC_PATTERN"
                                
                                # NEW: Check if we have a spring pattern at this level
                                spring_at_level = None
                                if wyckoff_patterns.get("springs"):
                                    for spring in wyckoff_patterns["springs"]:
                                        if abs(spring["level"] - best_support) < volatility * 2:
                                            spring_at_level = spring
                                            setup_type = "WYCKOFF_SPRING"
                                            break
                                
                                setup = {
                                    "valid_setup": True,
                                    "type": setup_type,
                                    "direction": "buy",
                                    "entry_price": current_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "strength": setup_strength,
                                    "setup_key": f"FIB_BULL_{self.trader.timeframe}",
                                    "description": f"Bullish setup at Fibonacci support level",
                                    "confluence_factors": []
                                }
                                
                                # NEW: Update description for Wyckoff patterns
                                if spring_at_level:
                                    setup["description"] = f"Bullish Wyckoff spring at Fibonacci support"
                                
                                # IMPROVEMENT: Check order flow confirmation for better entries
                                order_flow = self._check_order_flow_confirmation(best_support, "buy")
                                if order_flow["confirmed"]:
                                    setup_strength *= order_flow["strength_multiplier"]
                                    for reason in order_flow["reasons"]:
                                        if "confluence_factors" not in setup:
                                            setup["confluence_factors"] = []
                                        setup["confluence_factors"].append(reason)
                                
                                # Add confluence factors based on multi-timeframe alignment
                                if multi_tf_fib_levels["support_aligned"]:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                        
                                    for tf in multi_tf_fib_levels["aligned_timeframes"]:
                                        factor = f"FIB_SUPPORT_ALIGNED_{tf}"
                                        if factor not in setup["confluence_factors"]:
                                            setup["confluence_factors"].append(factor)
                                
                                # Add price action confluence
                                if price_action["bullish_reversal"]:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("BULLISH_REVERSAL")
                                    # NEW: Add quality factor if available
                                    if "reversal_quality" in price_action and price_action["reversal_quality"] > 1.2:
                                        setup["confluence_factors"].append("HIGH_QUALITY_REVERSAL")
                                    
                                # NEW: Add Wyckoff spring confluence if applicable
                                if spring_at_level:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("WYCKOFF_SPRING")
                                    # Add spring quality if available
                                    if spring_at_level.get("quality") == "high":
                                        setup["confluence_factors"].append("HIGH_QUALITY_SPRING")
                                    
                                # Add harmonic pattern confluence if applicable
                                for pattern in high_quality_patterns.get("bullish", []):
                                    if abs(pattern["price"] - best_support) < volatility * 2:
                                        if "confluence_factors" not in setup:
                                            setup["confluence_factors"] = []
                                        setup["confluence_factors"].append(f"HARMONIC_{pattern['type']}")
                                        # Update description to include harmonic pattern
                                        if not spring_at_level:  # Prefer spring pattern description if present
                                            setup["description"] = f"Bullish setup at Fibonacci support with {pattern['type']} harmonic pattern"
                                        # Add any confidence factors from validation
                                        for cf in pattern.get("confidence_factors", []):
                                            if cf not in setup["confluence_factors"]:
                                                setup["confluence_factors"].append(cf)
                                        # NEW: Add quality factors from assessment
                                        for qf in pattern.get("quality_factors", []):
                                            if qf not in setup["confluence_factors"]:
                                                setup["confluence_factors"].append(qf)
                                        break
                                        
                                # Add Fibonacci compression zone confluence if applicable
                                if compression_zones.get("support_zone") and abs(compression_zones["support_zone"] - best_support) < volatility:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("FIB_COMPRESSION_ZONE")
                                    # Update description to include compression zone
                                    if not spring_at_level and "HARMONIC_" not in str(setup.get("confluence_factors", [])):
                                        setup["description"] = "Bullish setup at Fibonacci compression zone"
                                        
                                # IMPROVEMENT: Add historical validation factor if score is high
                                if support_validation_score > 1.3:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("HISTORICAL_VALIDATION")
                                    
                                # IMPROVEMENT: Add precise entry timing factor if applicable
                                if entry_timing["entry_ready"]:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("PRECISE_ENTRY")
                                
                                # NEW: Add liquidity factors if applicable
                                if liquidity_at_support["liquidity_score"] > 0.7:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("HIGH_LIQUIDITY")
                                
                                # Calculate confluence score
                                setup = self._calculate_confluence_score(setup)
                                
                                # NEW: Integrate sentiment and liquidity analysis into the setup
                                setup = self._integrate_sentiment_liquidity_analysis(setup)
        
        if fib_resistance_clusters and (trend != "uptrend" or market_context["structural_inflection"]):
            # Sort resistance clusters by quality (descending)
            fib_resistance_clusters.sort(key=lambda x: x[1], reverse=True)
            best_resistance = fib_resistance_clusters[0][0]
            resistance_quality = fib_resistance_clusters[0][1]
            
            # IMPROVEMENT: Check historical validation of resistance level
            resistance_validation_score = self._validate_historical_fib_reactions(best_resistance, "resistance")
            resistance_quality *= resistance_validation_score
            
            # NEW: Check liquidity at this resistance level
            liquidity_at_resistance = self._analyze_liquidity(best_resistance, "sell")
            
            # Skip setups with extremely poor liquidity
            if not liquidity_at_resistance["sufficient_liquidity"] and liquidity_at_resistance["liquidity_score"] < 0.2:
                pass  # Skip this setup due to liquidity concerns
            else:
                # Check if price is near the resistance level (adjusted by sentiment-based volatility multiplier) 
                if abs(current_price - best_resistance) < volatility * (1.5 if trend == "downtrend" else 2.5) * volatility_multiplier:
                    # Skip if momentum is strong upward and no bearish reversal signal
                    if market_context["momentum"] > volatility * 3 and not price_action["bearish_reversal"]:
                        pass  # Skip this setup
                    else:
                        # IMPROVEMENT: Check entry timing precision
                        entry_timing = self._check_entry_timing("sell", current_price, best_resistance)
                        
                        # Calculate adaptive stop based on volatility and market conditions
                        stop_buffer = volatility * (
                            1.2 if trend == "downtrend" else
                            2.0 if trend == "uptrend" else
                            1.5
                        )
                        
                        # Additional buffer for high volatility or against trend
                        if market_context["volatility_regime"] == "high":
                            stop_buffer *= 1.3
                        
                        # Check for optimal stop placement based on recent swing highs
                        optimal_stop = self._find_optimal_stop_level("sell", best_resistance, stop_buffer)
                        stop_loss = optimal_stop if optimal_stop else best_resistance + stop_buffer
                        
                        # Find optimal take profit level
                        take_profit = self._find_optimal_target(
                            "bearish", 
                            current_price, 
                            volatility, 
                            market_context,
                            fib_support_clusters,
                            high_quality_patterns.get("bearish", [])  # Use validated and quality-filtered patterns
                        )
                        
                        if take_profit:
                            # Calculate reward-to-risk ratio
                            reward = current_price - take_profit
                            risk = stop_loss - current_price
                            
                            # Calculate required R:R based on market conditions
                            min_rr = self._calculate_required_rr("bearish", trend, market_context)
                            
                            # NEW: Adjust minimum required R:R based on sentiment
                            if sentiment_data["overall_bias"] == "bearish" and sentiment_data["confidence"] > 0.5:
                                min_rr *= max(0.7, 1.0 - (abs(sentiment_data["sentiment_score"]) * 0.3))
                            elif sentiment_data["overall_bias"] == "bullish" and sentiment_data["confidence"] > 0.5:
                                min_rr *= min(1.5, 1.0 + (sentiment_data["sentiment_score"] * 0.5))
                            
                            if reward >= risk * min_rr:
                                # Calculate dynamic setup strength
                                setup_strength = self._calculate_setup_strength(
                                    "sell",
                                    resistance_quality,  # Using the historically validated quality
                                    trend, 
                                    price_action, 
                                    market_context,
                                    multi_tf_fib_levels
                                )
                                
                                # IMPROVEMENT: Apply entry timing precision if available
                                if entry_timing["entry_ready"]:
                                    setup_strength *= entry_timing["confidence"]
                                
                                # Enhance setup strength if we have a harmonic pattern
                                for pattern in high_quality_patterns.get("bearish", []):
                                    if abs(pattern["price"] - best_resistance) < volatility * 2:
                                        quality_boost = pattern.get("quality_score", 1.0) * 0.3
                                        setup_strength *= 1.3 + quality_boost
                                        break
                                        
                                # NEW: Enhance setup strength if we have a Wyckoff upthrust pattern
                                if wyckoff_patterns.get("upthrusts"):
                                    for upthrust in wyckoff_patterns["upthrusts"]:
                                        if abs(upthrust["level"] - best_resistance) < volatility * 2:
                                            setup_strength *= upthrust.get("strength_multiplier", 1.5)
                                        break
                                        
                                # Assign setup type based on confluence factors
                                setup_type = "FIBONACCI"
                                if compression_zones.get("resistance_zone") and abs(compression_zones["resistance_zone"] - best_resistance) < volatility:
                                    setup_type = "FIB_COMPRESSION"
                                elif any(abs(p["price"] - best_resistance) < volatility * 2 for p in high_quality_patterns.get("bearish", [])):
                                    setup_type = "HARMONIC_PATTERN"
                                    
                                # NEW: Check if we have an upthrust pattern at this level
                                upthrust_at_level = None
                                if wyckoff_patterns.get("upthrusts"):
                                    for upthrust in wyckoff_patterns["upthrusts"]:
                                        if abs(upthrust["level"] - best_resistance) < volatility * 2:
                                            upthrust_at_level = upthrust
                                            setup_type = "WYCKOFF_UPTHRUST"
                                            break
                                
                                setup = {
                                    "valid_setup": True,
                                    "type": setup_type,
                                    "direction": "sell",
                                    "entry_price": current_price,
                                    "stop_loss": stop_loss,
                                    "take_profit": take_profit,
                                    "strength": setup_strength,
                                    "setup_key": f"FIB_BEAR_{self.trader.timeframe}",
                                    "description": f"Bearish setup at Fibonacci resistance level",
                                    "confluence_factors": []
                                }
                                
                                # NEW: Update description for Wyckoff patterns
                                if upthrust_at_level:
                                    setup["description"] = f"Bearish Wyckoff upthrust at Fibonacci resistance"
                                
                                # IMPROVEMENT: Check order flow confirmation for better entries
                                order_flow = self._check_order_flow_confirmation(best_resistance, "sell")
                                if order_flow["confirmed"]:
                                    setup_strength *= order_flow["strength_multiplier"]
                                    for reason in order_flow["reasons"]:
                                        if "confluence_factors" not in setup:
                                            setup["confluence_factors"] = []
                                        setup["confluence_factors"].append(reason)
                                
                                # Add confluence factors based on multi-timeframe alignment
                                if multi_tf_fib_levels["resistance_aligned"]:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                        
                                    for tf in multi_tf_fib_levels["aligned_timeframes"]:
                                        factor = f"FIB_RESISTANCE_ALIGNED_{tf}"
                                        if factor not in setup["confluence_factors"]:
                                            setup["confluence_factors"].append(factor)
                                
                                # Add price action confluence
                                if price_action["bearish_reversal"]:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("BEARISH_REVERSAL")
                                    # NEW: Add quality factor if available
                                    if "reversal_quality" in price_action and price_action["reversal_quality"] > 1.2:
                                        setup["confluence_factors"].append("HIGH_QUALITY_REVERSAL")
                                    
                                # NEW: Add Wyckoff upthrust confluence if applicable
                                if upthrust_at_level:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("WYCKOFF_UPTHRUST")
                                    # Add upthrust quality if available
                                    if upthrust_at_level.get("quality") == "high":
                                        setup["confluence_factors"].append("HIGH_QUALITY_UPTHRUST")
                                    
                                # Add harmonic pattern confluence if applicable
                                for pattern in high_quality_patterns.get("bearish", []):
                                    if abs(pattern["price"] - best_resistance) < volatility * 2:
                                        if "confluence_factors" not in setup:
                                            setup["confluence_factors"] = []
                                        setup["confluence_factors"].append(f"HARMONIC_{pattern['type']}")
                                        # Update description to include harmonic pattern
                                        if not upthrust_at_level:  # Prefer upthrust pattern description if present
                                            setup["description"] = f"Bearish setup at Fibonacci resistance with {pattern['type']} harmonic pattern"
                                        # Add any confidence factors from validation
                                        for cf in pattern.get("confidence_factors", []):
                                            if cf not in setup["confluence_factors"]:
                                                setup["confluence_factors"].append(cf)
                                        # NEW: Add quality factors from assessment
                                        for qf in pattern.get("quality_factors", []):
                                            if qf not in setup["confluence_factors"]:
                                                setup["confluence_factors"].append(qf)
                                        break
                                        
                                # Add Fibonacci compression zone confluence if applicable
                                if compression_zones.get("resistance_zone") and abs(compression_zones["resistance_zone"] - best_resistance) < volatility:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("FIB_COMPRESSION_ZONE")
                                    # Update description to include compression zone
                                    if not upthrust_at_level and "HARMONIC_" not in str(setup.get("confluence_factors", [])):
                                        setup["description"] = "Bearish setup at Fibonacci compression zone"
                                        
                                # IMPROVEMENT: Add historical validation factor if score is high
                                if resistance_validation_score > 1.3:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("HISTORICAL_VALIDATION")
                                    
                                # IMPROVEMENT: Add precise entry timing factor if applicable
                                if entry_timing["entry_ready"]:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("PRECISE_ENTRY")
                                
                                # NEW: Add liquidity factors if applicable 
                                if liquidity_at_resistance["liquidity_score"] > 0.7:
                                    if "confluence_factors" not in setup:
                                        setup["confluence_factors"] = []
                                    setup["confluence_factors"].append("HIGH_LIQUIDITY")
                                
                                # Calculate confluence score
                                setup = self._calculate_confluence_score(setup)
                                
                                # NEW: Integrate sentiment and liquidity analysis into the setup
                                setup = self._integrate_sentiment_liquidity_analysis(setup)
        
        return setup
    
    def _calculate_adaptive_volatility(self) -> float:
        """Calculate adaptive volatility based on market conditions"""
        # Default volatility if calculation fails
        default_volatility = 0.0005
        
        try:
            # Get volatility from primary timeframe
            if hasattr(self.trader, 'current_volatility') and self.trader.current_volatility:
                return self.trader.current_volatility
            
            # Fallback calculation if trader's volatility is not available
            if hasattr(self.trader, 'price_data') and self.trader.timeframe in self.trader.price_data:
                df = self.trader.price_data[self.trader.timeframe]
                if len(df) >= 20:
                    # Calculate ATR-like volatility
                    high_low = df['high'].iloc[-20:] - df['low'].iloc[-20:]
                    return high_low.mean()
            
            return default_volatility
        except Exception:
            return default_volatility
    
    def _analyze_market_context(self) -> Dict:
        """Analyze market context for better decision making"""
        context = {
            "session": "unknown",
            "range_bound": False,
            "momentum": 0,
            "volatility_regime": "normal",
            "structural_inflection": False
        }
        
        try:
            # Determine current trading session
            if hasattr(self.trader, 'price_data') and self.trader.timeframe in self.trader.price_data:
                df = self.trader.price_data[self.trader.timeframe]
                
                if len(df) > 0 and hasattr(df.index[-1], 'hour'):
                    current_hour = df.index[-1].hour
                    
                    # Simple session determination
                    if 8 <= current_hour < 16:
                        context["session"] = "london"
                    elif 13 <= current_hour < 21:
                        context["session"] = "new_york"
                    elif 0 <= current_hour < 8:
                        context["session"] = "tokyo"
                    else:
                        context["session"] = "overlap"
                
                # Detect range-bound conditions (last 20 candles)
                if len(df) >= 20:
                    high_range = df['high'].iloc[-20:].max() - df['high'].iloc[-20:].min()
                    if hasattr(self.trader, 'current_volatility') and high_range < self.trader.current_volatility * 4:
                        context["range_bound"] = True
                
                # Calculate momentum (last 5 candles vs previous 5)
                if len(df) >= 10:
                    recent_momentum = df['close'].iloc[-5:].mean() - df['close'].iloc[-10:-5].mean()
                    context["momentum"] = recent_momentum
                
                # Determine volatility regime
                if hasattr(self.trader, 'current_volatility') and hasattr(self.trader, 'avg_volatility'):
                    if self.trader.current_volatility > self.trader.avg_volatility * 1.3:
                        context["volatility_regime"] = "high"
                    elif self.trader.current_volatility < self.trader.avg_volatility * 0.7:
                        context["volatility_regime"] = "low"
                
                # Check for structural inflection points
                if hasattr(self.trader, 'market_structure'):
                    swings = self.trader.market_structure[self.trader.timeframe].get("swings", [])
                    if len(swings) >= 4:
                        # Check if we're at a potential structure change point
                        highs = [s for s in swings if s['type'] == 'high'][-2:]
                        lows = [s for s in swings if s['type'] == 'low'][-2:]
                        
                        if len(highs) == 2 and len(lows) == 2:
                            # Check for potential structure shift
                            if (highs[-1]['price'] > highs[-2]['price'] and 
                                lows[-1]['price'] <= lows[-2]['price']):
                                context["structural_inflection"] = True
                            elif (highs[-1]['price'] <= highs[-2]['price'] and 
                                  lows[-1]['price'] > lows[-2]['price']):
                                context["structural_inflection"] = True
        except Exception:
            # If analysis fails, return default context
            pass
            
        return context
    
    def _analyze_price_action(self) -> Dict:
        """Analyze recent price action for reversal signals"""
        price_action = {
            "bullish_reversal": False,
            "bearish_reversal": False,
            "bullish_strength": 0.0,
            "bearish_strength": 0.0
        }
        
        try:
            if hasattr(self.trader, 'price_data') and self.trader.timeframe in self.trader.price_data:
                df = self.trader.price_data[self.trader.timeframe]
                
                if len(df) < 3:
                    return price_action
                
                # Get last 3 candles
                last_candles = df.iloc[-3:]
                
                # Analyze for bullish reversal patterns
                
                # Check for bullish engulfing
                if (last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1] and  # Bullish candle
                    last_candles['open'].iloc[-1] <= last_candles['close'].iloc[-2] and  # Open below prev close
                    last_candles['close'].iloc[-1] > last_candles['open'].iloc[-2]):  # Close above prev open
                    
                    price_action["bullish_reversal"] = True
                    price_action["bullish_strength"] = 1.5
                
                # Check for hammer pattern
                last_candle = last_candles.iloc[-1]
                if last_candle['close'] > last_candle['open']:  # Bullish
                    body_size = last_candle['close'] - last_candle['open']
                    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    
                    if lower_wick > body_size * 2 and upper_wick < body_size:
                        price_action["bullish_reversal"] = True
                        price_action["bullish_strength"] = 1.5 + (lower_wick / body_size) * 0.2
                
                # Analyze for bearish reversal patterns
                
                # Check for bearish engulfing
                if (last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1] and  # Bearish candle
                    last_candles['open'].iloc[-1] >= last_candles['close'].iloc[-2] and  # Open above prev close
                    last_candles['close'].iloc[-1] < last_candles['open'].iloc[-2]):  # Close below prev open
                    
                    price_action["bearish_reversal"] = True
                    price_action["bearish_strength"] = 1.5
                
                # Check for shooting star pattern
                if last_candle['close'] < last_candle['open']:  # Bearish
                    body_size = last_candle['open'] - last_candle['close']
                    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    
                    if upper_wick > body_size * 2 and lower_wick < body_size:
                        price_action["bearish_reversal"] = True
                        price_action["bearish_strength"] = 1.5 + (upper_wick / body_size) * 0.2
        
        except Exception:
            pass
            
        return price_action
    
    def _find_nearest_levels(self, current_price: float, levels: Dict) -> Tuple[Optional[float], Optional[float]]:
        """Find the nearest Fibonacci support and resistance levels"""
        # Find nearest support below current price
        nearest_support = None
        nearest_support_dist = float('inf')
        
        # Find nearest resistance above current price
        nearest_resistance = None
        nearest_resistance_dist = float('inf')
        
        for level_name, level_price in levels.items():
            if level_price < current_price:
                # Potential support
                dist = current_price - level_price
                if dist < nearest_support_dist:
                    nearest_support = level_price
                    nearest_support_dist = dist
            elif level_price > current_price:
                # Potential resistance
                dist = level_price - current_price
                if dist < nearest_resistance_dist:
                    nearest_resistance = level_price
                    nearest_resistance_dist = dist
        
        return nearest_support, nearest_resistance
    
    def _find_fib_clusters(self, levels: Dict, volatility: float) -> tuple:
        """Find clusters of Fibonacci levels for stronger support/resistance"""
        # Define key Fibonacci levels for clustering
        key_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.272, 1.618, 2.0]
        
        # Get current price
        current_price = self.trader.get_current_price()
        
        # Initialize clusters
        support_clusters = []  # (price, strength)
        resistance_clusters = []  # (price, strength)
        
        # Sort levels by price
        sorted_prices = [(level, price) for level, price in levels.items() if level not in [0, 1.0]]
        sorted_prices.sort(key=lambda x: x[1])
        
        # Find clusters - when multiple levels are close to each other
        cluster_distance = volatility * 1.5  # Max distance for clustering
        
        i = 0
        while i < len(sorted_prices):
            cluster_start = i
            cluster_end = i
            current_level, current_price = sorted_prices[i]
            
            # Find all levels within clustering distance
            for j in range(i+1, len(sorted_prices)):
                if sorted_prices[j][1] - current_price < cluster_distance:
                    cluster_end = j
                else:
                    break
            
            # If we found a cluster (more than one level)
            if cluster_end > cluster_start:
                # Calculate average price for cluster
                cluster_prices = [sorted_prices[j][1] for j in range(cluster_start, cluster_end+1)]
                cluster_levels = [sorted_prices[j][0] for j in range(cluster_start, cluster_end+1)]
                
                cluster_avg_price = sum(cluster_prices) / len(cluster_prices)
                
                # Calculate cluster strength based on number of levels and if key levels are included
                key_level_count = sum(1 for level in cluster_levels if level in key_levels)
                cluster_strength = 1.0 + (len(cluster_prices) * 0.2) + (key_level_count * 0.3)
                
                # Add to appropriate cluster list
                if cluster_avg_price < current_price:
                    support_clusters.append((cluster_avg_price, cluster_strength))
                else:
                    resistance_clusters.append((cluster_avg_price, cluster_strength))
                
                i = cluster_end + 1
            else:
                # Single level - check if it's a key level
                is_key = current_level in key_levels
                level_strength = 1.0 + (0.3 if is_key else 0)
                
                # Add to appropriate list
                if current_price > current_price:
                    support_clusters.append((current_price, level_strength))
                else:
                    resistance_clusters.append((current_price, level_strength))
                    
                i += 1
        
        return support_clusters, resistance_clusters
    
    def _get_multi_timeframe_levels(self, current_price: float, volatility: float) -> Dict:
        """Check for Fibonacci confluence across multiple timeframes"""
        result = {
            "support_aligned": False,
            "resistance_aligned": False,
            "aligned_timeframes": [],
            "support_strength": 1.0,
            "resistance_strength": 1.0
        }
        
        if not hasattr(self.trader, 'fib_levels') or not hasattr(self.trader, 'secondary_timeframes'):
            return result
        
        # Get levels from primary timeframe
        if self.trader.timeframe not in self.trader.fib_levels:
            return result
            
        primary_levels = self.trader.fib_levels[self.trader.timeframe]['levels']
        
        # Find nearest levels in primary timeframe
        primary_support, primary_resistance = self._find_nearest_levels(current_price, primary_levels)
        
        if primary_support is None or primary_resistance is None:
            return result
        
        # Check each secondary timeframe for alignment
        aligned_support_tfs = []
        aligned_resistance_tfs = []
        
        for tf in self.trader.secondary_timeframes:
            if tf in self.trader.fib_levels:
                tf_levels = self.trader.fib_levels[tf]['levels']
                
                # Find nearest levels in this timeframe
                tf_support, tf_resistance = self._find_nearest_levels(current_price, tf_levels)
                
                # Check for support alignment
                if tf_support is not None and primary_support is not None:
                    if abs(tf_support - primary_support) < volatility * 2:
                        aligned_support_tfs.append(tf)
                
                # Check for resistance alignment
                if tf_resistance is not None and primary_resistance is not None:
                    if abs(tf_resistance - primary_resistance) < volatility * 2:
                        aligned_resistance_tfs.append(tf)
        
        # Update result based on findings
        if aligned_support_tfs:
            result["support_aligned"] = True
            result["aligned_timeframes"].extend(aligned_support_tfs)
            result["support_strength"] = 1.0 + (len(aligned_support_tfs) * 0.2)
        
        if aligned_resistance_tfs:
            result["resistance_aligned"] = True
            result["aligned_timeframes"].extend(aligned_resistance_tfs)
            result["resistance_strength"] = 1.0 + (len(aligned_resistance_tfs) * 0.2)
        
        # Remove duplicates
        result["aligned_timeframes"] = list(set(result["aligned_timeframes"]))
        
        return result
    
    def _find_optimal_stop_level(self, direction: str, fib_level: float, default_buffer: float) -> Optional[float]:
        """Find optimal stop loss level based on swing points for better placement"""
        try:
            if not hasattr(self.trader, 'market_structure') or self.trader.timeframe not in self.trader.market_structure:
                return None
                
            swings = self.trader.market_structure[self.trader.timeframe].get("swings", [])
            if not swings or len(swings) < 2:
                return None
            
            # For buy orders, look for recent swing lows
            if direction == "buy":
                # Get recent swing lows
                recent_lows = [s for s in swings if s['type'] == 'low'][-3:]  # Last 3 swing lows
                
                if not recent_lows:
                    return None
                
                # Find nearest swing low below the Fibonacci level
                valid_lows = [low for low in recent_lows if low['price'] < fib_level]
                
                if valid_lows:
                    # Use the most recent valid low that's not too far from the Fibonacci level
                    volatility = self._calculate_adaptive_volatility()
                    max_distance = volatility * 5  # Maximum acceptable distance
                    
                    closest_low = None
                    closest_distance = float('inf')
                    
                    for low in valid_lows:
                        distance = fib_level - low['price']
                        if distance < max_distance and distance < closest_distance:
                            closest_distance = distance
                            closest_low = low['price']
                    
                    if closest_low is not None:
                        # Add small buffer below the swing low
                        return closest_low - (volatility * 0.5)
            
            # For sell orders, look for recent swing highs
            elif direction == "sell":
                # Get recent swing highs
                recent_highs = [s for s in swings if s['type'] == 'high'][-3:]  # Last 3 swing highs
                
                if not recent_highs:
                    return None
                
                # Find nearest swing high above the Fibonacci level
                valid_highs = [high for high in recent_highs if high['price'] > fib_level]
                
                if valid_highs:
                    # Use the most recent valid high that's not too far from the Fibonacci level
                    volatility = self._calculate_adaptive_volatility()
                    max_distance = volatility * 5  # Maximum acceptable distance
                    
                    closest_high = None
                    closest_distance = float('inf')
                    
                    for high in valid_highs:
                        distance = high['price'] - fib_level
                        if distance < max_distance and distance < closest_distance:
                            closest_distance = distance
                            closest_high = high['price']
                    
                    if closest_high is not None:
                        # Add small buffer above the swing high
                        return closest_high + (volatility * 0.5)
            
            return None
        except Exception:
            return None
    
    def _detect_harmonic_patterns(self) -> Dict[str, List[Dict]]:
        """Detect harmonic patterns in the chart for high-probability reversals"""
        result = {
            "bullish": [],
            "bearish": []
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return result
            
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 20:
                return result
            
            # Find recent swing points for pattern detection
            if hasattr(self.trader, 'market_structure') and self.trader.timeframe in self.trader.market_structure:
                swings = self.trader.market_structure[self.trader.timeframe].get("swings", [])
                
                if len(swings) < 5:  # Need enough swing points
                    return result
                
                # Get recent swing points
                recent_swings = swings[-5:]
                
                # Calculate Fibonacci ratios between swing points
                patterns = self._identify_harmonic_patterns(recent_swings)
                
                # Add valid patterns to the result
                for pattern in patterns:
                    if pattern["direction"] == "bullish":
                        result["bullish"].append(pattern)
                    else:
                        result["bearish"].append(pattern)
            
            return result
        except Exception:
            return result
    
    def _identify_harmonic_patterns(self, swings: List[Dict]) -> List[Dict]:
        """Identify harmonic patterns from swing points"""
        patterns = []
        
        try:
            if len(swings) < 5:
                return patterns
            
            # Get recent XABCD points
            points = []
            for i, swing in enumerate(swings[-5:]):
                points.append({
                    "label": chr(65 + i),  # A, B, C, D, E
                    "price": swing["price"],
                    "type": swing["type"]
                })
            
            # Function to calculate ratio between two legs
            def calculate_ratio(start, end, reference_start, reference_end):
                if reference_end - reference_start == 0:
                    return 0
                return abs((end - start) / (reference_end - reference_start))
            
            # Check for Butterfly pattern (bullish and bearish)
            # XA=1, AB=0.786, BC=0.382-0.886, CD=1.618-2.618, AD=0.786
            if points[1]["price"] > points[0]["price"]:  # X to A is down
                # Potential bearish butterfly
                xPrice, aPrice, bPrice, cPrice, dPrice = [p["price"] for p in points]
                
                # Calculate ratios
                ab_ratio = calculate_ratio(bPrice, aPrice, aPrice, xPrice)
                bc_ratio = calculate_ratio(cPrice, bPrice, bPrice, aPrice)
                cd_ratio = calculate_ratio(dPrice, cPrice, cPrice, bPrice)
                ad_ratio = calculate_ratio(dPrice, aPrice, aPrice, xPrice)
                
                # Check if ratios match butterfly pattern
                if (0.7 < ab_ratio < 0.85 and
                    0.3 < bc_ratio < 0.9 and
                    1.5 < cd_ratio < 2.7 and
                    0.7 < ad_ratio < 0.85):
                    patterns.append({
                        "type": "BUTTERFLY",
                        "direction": "bearish",
                        "price": dPrice,
                        "strength": 2.0
                    })
            else:  # X to A is up
                # Potential bullish butterfly
                xPrice, aPrice, bPrice, cPrice, dPrice = [p["price"] for p in points]
                
                # Calculate ratios
                ab_ratio = calculate_ratio(bPrice, aPrice, aPrice, xPrice)
                bc_ratio = calculate_ratio(cPrice, bPrice, bPrice, aPrice)
                cd_ratio = calculate_ratio(dPrice, cPrice, cPrice, bPrice)
                ad_ratio = calculate_ratio(dPrice, aPrice, aPrice, xPrice)
                
                # Check if ratios match butterfly pattern
                if (0.7 < ab_ratio < 0.85 and
                    0.3 < bc_ratio < 0.9 and
                    1.5 < cd_ratio < 2.7 and
                    0.7 < ad_ratio < 0.85):
                    patterns.append({
                        "type": "BUTTERFLY",
                        "direction": "bullish",
                        "price": dPrice,
                        "strength": 2.0
                    })
            
            # Check for Gartley pattern (bullish and bearish)
            # XA=1, AB=0.618, BC=0.382-0.886, CD=1.272-1.618, AD=0.786
            if points[1]["price"] > points[0]["price"]:  # X to A is down
                # Potential bearish Gartley
                xPrice, aPrice, bPrice, cPrice, dPrice = [p["price"] for p in points]
                
                # Calculate ratios
                ab_ratio = calculate_ratio(bPrice, aPrice, aPrice, xPrice)
                bc_ratio = calculate_ratio(cPrice, bPrice, bPrice, aPrice)
                cd_ratio = calculate_ratio(dPrice, cPrice, cPrice, bPrice)
                ad_ratio = calculate_ratio(dPrice, aPrice, aPrice, xPrice)
                
                # Check if ratios match Gartley pattern
                if (0.58 < ab_ratio < 0.65 and
                    0.3 < bc_ratio < 0.9 and
                    1.2 < cd_ratio < 1.7 and
                    0.7 < ad_ratio < 0.85):
                    patterns.append({
                        "type": "GARTLEY",
                        "direction": "bearish",
                        "price": dPrice,
                        "strength": 1.8
                    })
            else:  # X to A is up
                # Potential bullish Gartley
                xPrice, aPrice, bPrice, cPrice, dPrice = [p["price"] for p in points]
                
                # Calculate ratios
                ab_ratio = calculate_ratio(bPrice, aPrice, aPrice, xPrice)
                bc_ratio = calculate_ratio(cPrice, bPrice, bPrice, aPrice)
                cd_ratio = calculate_ratio(dPrice, cPrice, cPrice, bPrice)
                ad_ratio = calculate_ratio(dPrice, aPrice, aPrice, xPrice)
                
                # Check if ratios match Gartley pattern
                if (0.58 < ab_ratio < 0.65 and
                    0.3 < bc_ratio < 0.9 and
                    1.2 < cd_ratio < 1.7 and
                    0.7 < ad_ratio < 0.85):
                    patterns.append({
                        "type": "GARTLEY",
                        "direction": "bullish",
                        "price": dPrice,
                        "strength": 1.8
                    })
            
            # Check for Bat pattern (bullish and bearish)
            # XA=1, AB=0.382-0.5, BC=0.382-0.886, CD=1.618-2.618, AD=0.886
            if points[1]["price"] > points[0]["price"]:  # X to A is down
                # Potential bearish Bat
                xPrice, aPrice, bPrice, cPrice, dPrice = [p["price"] for p in points]
                
                # Calculate ratios
                ab_ratio = calculate_ratio(bPrice, aPrice, aPrice, xPrice)
                bc_ratio = calculate_ratio(cPrice, bPrice, bPrice, aPrice)
                cd_ratio = calculate_ratio(dPrice, cPrice, cPrice, bPrice)
                ad_ratio = calculate_ratio(dPrice, aPrice, aPrice, xPrice)
                
                # Check if ratios match Bat pattern
                if (0.35 < ab_ratio < 0.55 and
                    0.3 < bc_ratio < 0.9 and
                    1.5 < cd_ratio < 2.7 and
                    0.85 < ad_ratio < 0.95):
                    patterns.append({
                        "type": "BAT",
                        "direction": "bearish",
                        "price": dPrice,
                        "strength": 1.9
                    })
            else:  # X to A is up
                # Potential bullish Bat
                xPrice, aPrice, bPrice, cPrice, dPrice = [p["price"] for p in points]
                
                # Calculate ratios
                ab_ratio = calculate_ratio(bPrice, aPrice, aPrice, xPrice)
                bc_ratio = calculate_ratio(cPrice, bPrice, bPrice, aPrice)
                cd_ratio = calculate_ratio(dPrice, cPrice, cPrice, bPrice)
                ad_ratio = calculate_ratio(dPrice, aPrice, aPrice, xPrice)
                
                # Check if ratios match Bat pattern
                if (0.35 < ab_ratio < 0.55 and
                    0.3 < bc_ratio < 0.9 and
                    1.5 < cd_ratio < 2.7 and
                    0.85 < ad_ratio < 0.95):
                    patterns.append({
                        "type": "BAT",
                        "direction": "bullish",
                        "price": dPrice,
                        "strength": 1.9
                    })
            
            return patterns
        except Exception:
            return patterns
    
    def _calculate_custom_slope(self, price_data: pd.Series, window: int = 20) -> float:
        """Calculate a custom slope indicator for trend direction and strength"""
        if len(price_data) < window:
            return 0
        
        # Use linear regression to calculate slope
        y = price_data[-window:].values
        x = np.array(range(window))
        slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
        return slope[0]
    
    def _detect_fibonacci_compression_zones(self, volatility: float) -> Dict:
        """Detect Fibonacci compression zones where multiple levels converge"""
        result = {
            "support_zone": None,
            "resistance_zone": None,
            "support_strength": 1.0,
            "resistance_strength": 1.0
        }
        
        try:
            if not hasattr(self.trader, 'fib_levels') or self.trader.timeframe not in self.trader.fib_levels:
                return result
                
            # Get current price
            current_price = self.trader.get_current_price()
            if current_price is None:
                return result
            
            # Get Fibonacci levels
            fib_data = self.trader.fib_levels[self.trader.timeframe]
            levels = fib_data['levels']
            
            # Check for multi-timeframe Fibonacci alignments
            timeframes = ["15m", "30m", "1h", "4h", "1d"]
            if self.trader.timeframe not in timeframes:
                timeframes.append(self.trader.timeframe)
                
            # Create bins for clustering - separate support and resistance levels
            support_bins = {}  # price -> [count, strength]
            resistance_bins = {}  # price -> [count, strength]
            
            # Process current timeframe
            for level_type, level_price in levels.items():
                # Skip specific level types
                if level_type in ['swing_high', 'swing_low', 'trend_high', 'trend_low']:
                    continue
                    
                # Skip if level is too far from current price (outside reasonable range)
                if abs(level_price - current_price) > volatility * 15:
                    continue
                    
                # Assign to appropriate bin with some proximity tolerance
                bin_key = round(level_price / (volatility * 0.5)) * (volatility * 0.5)
                
                # Determine if support or resistance
                if level_price < current_price:  # Support
                    if bin_key not in support_bins:
                        support_bins[bin_key] = [0, 0]
                    support_bins[bin_key][0] += 1
                    support_bins[bin_key][1] += self._get_level_quality(level_type)
                else:  # Resistance
                    if bin_key not in resistance_bins:
                        resistance_bins[bin_key] = [0, 0]
                    resistance_bins[bin_key][0] += 1
                    resistance_bins[bin_key][1] += self._get_level_quality(level_type)
            
            # Add other timeframes if available
            for tf in timeframes:
                if tf == self.trader.timeframe:
                    continue  # Skip current TF as we already processed it
                
                if hasattr(self.trader, 'fib_levels') and tf in self.trader.fib_levels:
                    tf_levels = self.trader.fib_levels[tf]['levels']
                    
                    for level_type, level_price in tf_levels.items():
                        # Skip specific level types
                        if level_type in ['swing_high', 'swing_low', 'trend_high', 'trend_low']:
                            continue
                            
                        # Skip if level is too far from current price
                        if abs(level_price - current_price) > volatility * 15:
                            continue
                            
                        # Assign to appropriate bin with increased tolerance for higher timeframes
                        bin_key = round(level_price / (volatility * 0.5)) * (volatility * 0.5)
                        
                        # Scale quality based on timeframe importance
                        tf_multiplier = self._get_timeframe_importance(tf)
                        
                        # Determine if support or resistance
                        if level_price < current_price:  # Support
                            if bin_key not in support_bins:
                                support_bins[bin_key] = [0, 0]
                            support_bins[bin_key][0] += 1
                            support_bins[bin_key][1] += self._get_level_quality(level_type) * tf_multiplier
                        else:  # Resistance
                            if bin_key not in resistance_bins:
                                resistance_bins[bin_key] = [0, 0]
                            resistance_bins[bin_key][0] += 1
                            resistance_bins[bin_key][1] += self._get_level_quality(level_type) * tf_multiplier
            
            # Find best support and resistance compression zones
            best_support_bin = None
            best_support_score = 0
            for bin_price, (count, strength) in support_bins.items():
                # Score based on count of levels and their combined strength
                bin_score = count * math.sqrt(strength)
                
                # Adjust score by proximity to current price (closer is better)
                proximity_factor = 1.0 - min(1.0, abs(current_price - bin_price) / (volatility * 10))
                bin_score *= (0.5 + proximity_factor)
                
                if bin_score > best_support_score and count >= 3:  # Require at least 3 levels
                    best_support_score = bin_score
                    best_support_bin = bin_price
            
            best_resistance_bin = None
            best_resistance_score = 0
            for bin_price, (count, strength) in resistance_bins.items():
                # Score based on count of levels and their combined strength
                bin_score = count * math.sqrt(strength)
                
                # Adjust score by proximity to current price (closer is better)
                proximity_factor = 1.0 - min(1.0, abs(current_price - bin_price) / (volatility * 10))
                bin_score *= (0.5 + proximity_factor)
                
                if bin_score > best_resistance_score and count >= 3:  # Require at least 3 levels
                    best_resistance_score = bin_score
                    best_resistance_bin = bin_price
            
            # Set results if we found compression zones
            if best_support_bin is not None:
                result["support_zone"] = best_support_bin
                result["support_strength"] = 1.0 + (best_support_score * 0.1)  # Scale to reasonable range
                
            if best_resistance_bin is not None:
                result["resistance_zone"] = best_resistance_bin
                result["resistance_strength"] = 1.0 + (best_resistance_score * 0.1)  # Scale to reasonable range
            
            return result
        except Exception:
            return result
            
    def _get_level_quality(self, level_type: str) -> float:
        """Get the quality score for a Fibonacci level type"""
        # Assign quality weights to different Fibonacci levels
        quality_weights = {
            # Major retracement levels
            'fib_0': 0.5,
            'fib_236': 1.0,
            'fib_382': 1.5,
            'fib_500': 1.8,
            'fib_618': 2.0,
            'fib_786': 1.5,
            'fib_886': 1.2,
            'fib_1000': 0.5,
            
            # Extension levels
            'ext_1272': 1.3,
            'ext_1618': 1.7,
            'ext_2000': 1.0,
            'ext_2618': 1.2,
            'ext_3618': 0.8,
            
            # Projection levels
            'proj_100': 1.5,
            'proj_1618': 1.8,
            'proj_2618': 1.2,
            'proj_4236': 0.9,
            
            # Structure levels
            'swing_high': 2.0,
            'swing_low': 2.0,
            'trend_high': 1.8,
            'trend_low': 1.8,
            'structure_high': 2.2,
            'structure_low': 2.2,
        }
        
        return quality_weights.get(level_type, 1.0)
        
    def _get_timeframe_importance(self, timeframe: str) -> float:
        """Get the importance multiplier for different timeframes"""
        importance = {
            '1m': 0.5,
            '5m': 0.7,
            '15m': 0.9,
            '30m': 1.1,
            '1h': 1.3,
            '4h': 1.5,
            '1d': 1.8,
            '1w': 2.0
        }
        
        return importance.get(timeframe, 1.0)
            
    def _find_optimal_target(self, direction: str, current_price: float, 
                           volatility: float, market_context: Dict,
                           fib_clusters: List, harmonic_patterns: List = None) -> Optional[float]:
        """Find an optimal take profit target with higher probability"""
        try:
            # If we have harmonic patterns, they provide high-quality targets
            if harmonic_patterns and len(harmonic_patterns) > 0:
                # Sort patterns by strength
                sorted_patterns = sorted(harmonic_patterns, key=lambda x: x["strength"], reverse=True)
                pattern = sorted_patterns[0]
                
                # For bullish patterns, target is usually a Fibonacci extension of the pattern
                if direction == "bullish" and pattern["price"] < current_price:
                    # Typical target is the 1.272 or 1.618 extension
                    pattern_range = abs(pattern["price"] - current_price)
                    return current_price + (pattern_range * 1.618)
                
                # For bearish patterns, target is usually a Fibonacci extension downward
                elif direction == "bearish" and pattern["price"] > current_price:
                    pattern_range = abs(pattern["price"] - current_price)
                    return current_price - (pattern_range * 1.618)
            
            # If we have Fibonacci clusters, use them
            if fib_clusters:
                sorted_clusters = sorted(fib_clusters, key=lambda x: x[1], reverse=True)
                best_cluster = sorted_clusters[0][0]
                
                # For bullish setups, find resistance above
                if direction == "bullish" and best_cluster > current_price:
                    return best_cluster
                
                # For bearish setups, find support below
                if direction == "bearish" and best_cluster < current_price:
                    return best_cluster
            
            # Fallback to standard S/R zones
            if direction == "bullish":
                # Find resistance levels above current price
                resistances = []
                
                for zone in self.trader.zones:
                    if zone.price_start > current_price:
                        # Calculate distance and quality score
                        distance = zone.price_start - current_price
                        if distance < volatility * 15:  # Only consider reasonably close targets
                            resistances.append(zone.price_start)
                
                if resistances:
                    return min(resistances)  # Closest resistance
                
                # If no resistance found, use volatility-based target
                return current_price + (volatility * 8)
            
            elif direction == "bearish":
                # Find support levels below current price
                supports = []
                
                for zone in self.trader.zones:
                    if zone.price_end < current_price:
                        # Calculate distance and quality score
                        distance = current_price - zone.price_end
                        if distance < volatility * 15:  # Only consider reasonably close targets
                            supports.append(zone.price_end)
                
                if supports:
                    return max(supports)  # Closest support
                
                # If no support found, use volatility-based target
                return current_price - (volatility * 8)
            
            return None
        except Exception:
            return None
    
    def _calculate_required_rr(self, direction: str, trend: str, market_context: Dict) -> float:
        """Calculate required risk-reward ratio based on market conditions"""
        # Base R:R
        base_rr = 1.5
        
        # Adjust for trend alignment
        if ((direction == "bullish" and trend == "uptrend") or
            (direction == "bearish" and trend == "downtrend")):
            base_rr -= 0.2  # Lower R:R required with trend
        elif ((direction == "bullish" and trend == "downtrend") or
              (direction == "bearish" and trend == "uptrend")):
            base_rr += 0.5  # Higher R:R required against trend
        
        # Adjust for structural inflection
        if market_context["structural_inflection"]:
            base_rr -= 0.3  # Lower R:R at structure changes
        
        # Adjust for volatility
        if market_context["volatility_regime"] == "high":
            base_rr += 0.3  # Higher R:R in high volatility
        elif market_context["volatility_regime"] == "low":
            base_rr -= 0.2  # Lower R:R in low volatility
        
        # Adjust for range conditions
        if market_context["range_bound"]:
            base_rr += 0.4  # Higher R:R in range-bound conditions
        
        # Ensure minimum R:R
        return max(1.2, base_rr)
    
    def _calculate_setup_strength(self, direction: str, fib_quality: float, 
                                trend: str, price_action: Dict, 
                                market_context: Dict, mtf_levels: Dict) -> float:
        """Calculate dynamic setup strength based on multiple factors"""
        # Base strength from Fibonacci quality
        base_strength = fib_quality
        
        # Adjust for trend alignment
        if ((direction == "buy" and trend == "uptrend") or 
            (direction == "sell" and trend == "downtrend")):
            base_strength *= 1.2  # Trend-aligned trades are stronger
        elif ((direction == "buy" and trend == "downtrend") or 
              (direction == "sell" and trend == "uptrend")):
            base_strength *= 0.8  # Counter-trend trades are weaker
        
        # Adjust for price action
        if direction == "buy" and price_action["bullish_reversal"]:
            base_strength *= price_action["bullish_strength"]
        elif direction == "sell" and price_action["bearish_reversal"]:
            base_strength *= price_action["bearish_strength"]
        
        # Adjust for market context
        if market_context["structural_inflection"]:
            base_strength *= 1.3  # Structure changes are significant
        
        # Adjust for multi-timeframe alignment
        if direction == "buy" and mtf_levels["support_aligned"]:
            base_strength *= mtf_levels["support_strength"]
        elif direction == "sell" and mtf_levels["resistance_aligned"]:
            base_strength *= mtf_levels["resistance_strength"]
        
        # Adjust for session
        if market_context["session"] in ["london", "new_york"]:
            base_strength *= 1.1  # Major sessions have more reliable moves
        elif market_context["session"] == "overlap":
            base_strength *= 1.05  # Session overlaps can be good
        
        # Cap strength at reasonable values
        return min(2.5, max(0.8, base_strength)) 
    
    def _validate_historical_fib_reactions(self, fib_level: float, level_type: str) -> float:
        """Validate how price has historically reacted to this Fibonacci level"""
        reaction_score = 1.0
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return reaction_score
                
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 50:
                return reaction_score
                
            # Calculate proximity threshold based on volatility
            volatility = self._calculate_adaptive_volatility()
            proximity_threshold = volatility * 1.5
            
            # Count how many times price reacted to this level
            reactions = 0
            touches = 0
            
            for i in range(10, len(df) - 15):  # Ensure we have room to look ahead
                # Check if price was close to the fib level
                if abs(df['low'].iloc[i] - fib_level) < proximity_threshold or abs(df['high'].iloc[i] - fib_level) < proximity_threshold:
                    touches += 1
                    
                    # Check if price reacted by moving at least 2x volatility away after touching
                    if level_type == "support" and df['close'].iloc[i+5:i+15].max() > fib_level + (volatility * 2):
                        reactions += 1
                    elif level_type == "resistance" and df['close'].iloc[i+5:i+15].min() < fib_level - (volatility * 2):
                        reactions += 1
            
            if touches > 0:
                reaction_rate = reactions / touches
                reaction_score = 1.0 + (reaction_rate * 1.5)  # Scale to 1.0-2.5 range
                
            return reaction_score
        except Exception:
            return reaction_score
    
    def _check_entry_timing(self, direction: str, current_price: float, level_price: float) -> Dict:
        """Improve entry timing precision"""
        result = {
            "entry_ready": False,
            "confidence": 0.0
        }
        
        try:
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 3:
                return result
                
            # Calculate parameters
            volatility = self._calculate_adaptive_volatility()
            proximity_threshold = volatility * 0.8
            
            # Check proximity to level
            if abs(current_price - level_price) > proximity_threshold:
                return result
                
            # Check for entry confirmation via momentum shift
            last_candles = df.iloc[-3:].copy()
            
            if direction == "buy":
                # For bullish entry, check for reversal of bearish momentum
                if last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1]:  # Current candle is bullish
                    if (last_candles['close'].iloc[-2] <= last_candles['open'].iloc[-2] and  # Previous candle was bearish
                        last_candles['low'].iloc[-1] > last_candles['low'].iloc[-2]):  # Higher low formed
                        
                        # Calculate confidence score
                        body_ratio = (last_candles['close'].iloc[-1] - last_candles['open'].iloc[-1]) / (last_candles['high'].iloc[-1] - last_candles['low'].iloc[-1])
                        result["entry_ready"] = True
                        result["confidence"] = min(1.5, 1.0 + body_ratio)
                        
            elif direction == "sell":
                # For bearish entry, check for reversal of bullish momentum
                if last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1]:  # Current candle is bearish
                    if (last_candles['close'].iloc[-2] >= last_candles['open'].iloc[-2] and  # Previous candle was bullish
                        last_candles['high'].iloc[-1] < last_candles['high'].iloc[-2]):  # Lower high formed
                        
                        # Calculate confidence score
                        body_ratio = (last_candles['open'].iloc[-1] - last_candles['close'].iloc[-1]) / (last_candles['high'].iloc[-1] - last_candles['low'].iloc[-1])
                        result["entry_ready"] = True
                        result["confidence"] = min(1.5, 1.0 + body_ratio)
            
            return result
        except Exception:
            return result
    
    def _find_fib_clusters_with_volume(self, levels: Dict, volatility: float) -> tuple:
        """Find clusters of Fibonacci levels with volume confirmation"""
        support_clusters = []
        resistance_clusters = []
        
        try:
            # Get current price
            current_price = self.trader.get_current_price()
            if current_price is None:
                return support_clusters, resistance_clusters
                
            # Get volume data
            df = self.trader.price_data[self.trader.timeframe]
            df = self.get_volume_data(df)  # Ensure volume data is available
            
            # Calculate average volume
            avg_volume = df['volume'].iloc[-50:].mean()
            
            # Find basic clusters
            basic_supports, basic_resistances = self._find_fib_clusters(levels, volatility)
            
            # Check volume confirmation for each cluster
            for level_price, base_strength in basic_supports:
                # Find candles near this level
                level_touches = []
                for i in range(max(0, len(df) - 50), len(df)):
                    if abs(df['low'].iloc[i] - level_price) < volatility:
                        level_touches.append(i)
                
                # If we have touches, check for volume confirmation
                if level_touches:
                    # Calculate average volume at touches
                    touch_volumes = [df['volume'].iloc[i] for i in level_touches]
                    avg_touch_volume = sum(touch_volumes) / len(touch_volumes)
                    
                    # Volume ratio (touch vol / average vol)
                    volume_ratio = avg_touch_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Adjust strength based on volume confirmation
                    adjusted_strength = base_strength * (1.0 + min(1.0, (volume_ratio - 1) * 0.5))
                    
                    support_clusters.append((level_price, adjusted_strength))
                else:
                    support_clusters.append((level_price, base_strength))
            
            # Same for resistance levels
            for level_price, base_strength in basic_resistances:
                # Find candles near this level
                level_touches = []
                for i in range(max(0, len(df) - 50), len(df)):
                    if abs(df['high'].iloc[i] - level_price) < volatility:
                        level_touches.append(i)
                
                # If we have touches, check for volume confirmation
                if level_touches:
                    # Calculate average volume at touches
                    touch_volumes = [df['volume'].iloc[i] for i in level_touches]
                    avg_touch_volume = sum(touch_volumes) / len(touch_volumes)
                    
                    # Volume ratio (touch vol / average vol)
                    volume_ratio = avg_touch_volume / avg_volume if avg_volume > 0 else 1.0
                    
                    # Adjust strength based on volume confirmation
                    adjusted_strength = base_strength * (1.0 + min(1.0, (volume_ratio - 1) * 0.5))
                    
                    resistance_clusters.append((level_price, adjusted_strength))
                else:
                    resistance_clusters.append((level_price, base_strength))
                    
            return support_clusters, resistance_clusters
        except Exception:
            return support_clusters, resistance_clusters
    
    def _validate_harmonic_pattern(self, pattern: Dict) -> Dict:
        """Validate harmonic pattern with additional criteria"""
        result = {
            "valid": True,
            "strength": pattern.get("strength", 1.0),
            "confidence_factors": []
        }
        
        try:
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 20:
                return result
            
            # Assign unique ID to pattern for tracking
            pattern_id = f"{pattern['type']}_{pattern['direction']}_{pattern['price']:.5f}"
            result["id"] = pattern_id
                
            # Check for historical validation of this level
            historical_score = self._validate_historical_fib_reactions(
                pattern["price"],
                "support" if pattern["direction"] == "bullish" else "resistance"
            )
            result["strength"] *= historical_score
            
            if historical_score > 1.2:
                result["confidence_factors"].append("HISTORICAL_VALIDATION")
                
            # Check for volume confirmation
            pattern_idx = None
            volatility = self._calculate_adaptive_volatility()
            for i in range(len(df) - 10, len(df)):
                if abs(df['low'].iloc[i] - pattern["price"]) < volatility:
                    pattern_idx = i
                    break
                    
            if pattern_idx is not None:
                # Check if volume is elevated at pattern completion
                avg_volume = df['volume'].iloc[-20:].mean()
                pattern_volume = df['volume'].iloc[pattern_idx]
                
                if pattern_volume > avg_volume * 1.5:
                    result["confidence_factors"].append("VOLUME_CONFIRMATION")
                    result["strength"] *= 1.3
                
            # Look for previous reactions at similar price levels
            reactions = 0
            volatility = self._calculate_adaptive_volatility()
            for i in range(20, len(df) - 20):
                if abs(df['low'].iloc[i] - pattern["price"]) < volatility * 2:
                    if pattern["direction"] == "bullish" and df['close'].iloc[i+5:i+15].max() > df['close'].iloc[i] * 1.01:
                        reactions += 1
                    elif pattern["direction"] == "bearish" and df['close'].iloc[i+5:i+15].min() < df['close'].iloc[i] * 0.99:
                        reactions += 1
            
            if reactions > 0:
                result["confidence_factors"].append("PREVIOUS_REACTIONS")
                result["strength"] *= min(1.5, 1.0 + (reactions * 0.1))
            
            # Verify pattern is not invalidated by current price action
            if pattern["direction"] == "bullish" and df['close'].iloc[-1] < pattern["price"]:
                result["valid"] = False
            elif pattern["direction"] == "bearish" and df['close'].iloc[-1] > pattern["price"]:
                result["valid"] = False
                
            return result
        except Exception:
            return result
    
    # NEW METHOD: Real-Time Pattern Quality Assessment
    def _assess_pattern_quality(self, validated_patterns: Dict[str, List[Dict]], market_context: Dict) -> Dict:
        """Assess the quality of patterns in real-time based on multiple factors"""
        pattern_quality = {}  # id -> quality assessment
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return pattern_quality
                
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 30:
                return pattern_quality
                
            volatility = self._calculate_adaptive_volatility()
            current_time = datetime.now() if hasattr(df.index[-1], 'hour') else None
                
            # Process bullish patterns
            for pattern in validated_patterns.get("bullish", []):
                if not pattern.get("id"):
                    continue
                    
                quality_assessment = {
                    "quality_score": 1.0,  # Base score
                    "quality_factors": []
                }
                
                # 1. Check for pattern symmetry and proportion
                if self._check_pattern_symmetry(pattern):
                    quality_assessment["quality_score"] *= 1.2
                    quality_assessment["quality_factors"].append("SYMMETRICAL_PROPORTIONS")
                
                # 2. Check for completeness of pattern development
                completeness = self._assess_pattern_completeness(pattern)
                if completeness > 0:
                    quality_assessment["quality_score"] *= (1.0 + min(0.5, completeness * 0.1))
                    if completeness >= 8:
                        quality_assessment["quality_factors"].append("FULLY_DEVELOPED")
                    elif completeness >= 5:
                        quality_assessment["quality_factors"].append("WELL_DEVELOPED")
                
                # 3. Check momentum alignment
                momentum_score = self._check_momentum_alignment(pattern["direction"], df)
                quality_assessment["quality_score"] *= momentum_score
                if momentum_score > 1.2:
                    quality_assessment["quality_factors"].append("MOMENTUM_ALIGNED")
                elif momentum_score < 0.8:
                    quality_assessment["quality_factors"].append("MOMENTUM_CONFLICT")
                
                # 4. Time-based pattern validity (freshness factor)
                if current_time and hasattr(pattern, 'timestamp'):
                    pattern_age = (current_time - pattern['timestamp']).total_seconds() / 3600  # In hours
                    if pattern_age < 4:  # Fresh pattern (< 4 hours old)
                        quality_assessment["quality_score"] *= 1.2
                        quality_assessment["quality_factors"].append("FRESH_PATTERN")
                    elif pattern_age > 24:  # Pattern older than 24 hours
                        quality_assessment["quality_score"] *= 0.8
                        quality_assessment["quality_factors"].append("AGED_PATTERN")
                
                # 5. Candle quality at pattern completion points
                candle_quality = self._assess_candle_quality(pattern, df)
                quality_assessment["quality_score"] *= candle_quality["score"]
                if candle_quality["factors"]:
                    quality_assessment["quality_factors"].extend(candle_quality["factors"])
                
                # 6. Consider market context
                if ((pattern["direction"] == "bullish" and market_context["trend"] == "uptrend") or
                    (pattern["direction"] == "bearish" and market_context["trend"] == "downtrend")):
                    quality_assessment["quality_score"] *= 1.2
                    quality_assessment["quality_factors"].append("TREND_ALIGNED")
                elif ((pattern["direction"] == "bullish" and market_context["trend"] == "downtrend") or
                      (pattern["direction"] == "bearish" and market_context["trend"] == "uptrend")):
                    quality_assessment["quality_score"] *= 0.8
                    quality_assessment["quality_factors"].append("COUNTER_TREND")
                
                # 7. Volatility-adjusted invalidation criteria
                threshold = pattern["price"] * (1 - (volatility * 2)) if pattern["direction"] == "bullish" else pattern["price"] * (1 + (volatility * 2))
                current_price = self.trader.get_current_price()
                
                if ((pattern["direction"] == "bullish" and current_price < threshold) or
                    (pattern["direction"] == "bearish" and current_price > threshold)):
                    quality_assessment["quality_score"] *= 0.5
                    quality_assessment["quality_factors"].append("NEAR_INVALIDATION")
                
                pattern_quality[pattern["id"]] = quality_assessment
            
            # Process bearish patterns (similar logic)
            for pattern in validated_patterns.get("bearish", []):
                if not pattern.get("id"):
                    continue
                    
                quality_assessment = {
                    "quality_score": 1.0,  # Base score
                    "quality_factors": []
                }
                
                # Apply same quality checks as for bullish patterns
                # 1. Check for pattern symmetry and proportion
                if self._check_pattern_symmetry(pattern):
                    quality_assessment["quality_score"] *= 1.2
                    quality_assessment["quality_factors"].append("SYMMETRICAL_PROPORTIONS")
                
                # 2. Check for completeness of pattern development
                completeness = self._assess_pattern_completeness(pattern)
                if completeness > 0:
                    quality_assessment["quality_score"] *= (1.0 + min(0.5, completeness * 0.1))
                    if completeness >= 8:
                        quality_assessment["quality_factors"].append("FULLY_DEVELOPED")
                    elif completeness >= 5:
                        quality_assessment["quality_factors"].append("WELL_DEVELOPED")
                
                # 3. Check momentum alignment
                momentum_score = self._check_momentum_alignment(pattern["direction"], df)
                quality_assessment["quality_score"] *= momentum_score
                if momentum_score > 1.2:
                    quality_assessment["quality_factors"].append("MOMENTUM_ALIGNED")
                elif momentum_score < 0.8:
                    quality_assessment["quality_factors"].append("MOMENTUM_CONFLICT")
                
                # 4. Time-based pattern validity (freshness factor)
                if current_time and hasattr(pattern, 'timestamp'):
                    pattern_age = (current_time - pattern['timestamp']).total_seconds() / 3600  # In hours
                    if pattern_age < 4:  # Fresh pattern (< 4 hours old)
                        quality_assessment["quality_score"] *= 1.2
                        quality_assessment["quality_factors"].append("FRESH_PATTERN")
                    elif pattern_age > 24:  # Pattern older than 24 hours
                        quality_assessment["quality_score"] *= 0.8
                        quality_assessment["quality_factors"].append("AGED_PATTERN")
                
                # 5. Candle quality at pattern completion points
                candle_quality = self._assess_candle_quality(pattern, df)
                quality_assessment["quality_score"] *= candle_quality["score"]
                if candle_quality["factors"]:
                    quality_assessment["quality_factors"].extend(candle_quality["factors"])
                
                # 6. Consider market context
                if ((pattern["direction"] == "bullish" and market_context["trend"] == "uptrend") or
                    (pattern["direction"] == "bearish" and market_context["trend"] == "downtrend")):
                    quality_assessment["quality_score"] *= 1.2
                    quality_assessment["quality_factors"].append("TREND_ALIGNED")
                elif ((pattern["direction"] == "bullish" and market_context["trend"] == "downtrend") or
                      (pattern["direction"] == "bearish" and market_context["trend"] == "uptrend")):
                    quality_assessment["quality_score"] *= 0.8
                    quality_assessment["quality_factors"].append("COUNTER_TREND")
                
                # 7. Volatility-adjusted invalidation criteria
                threshold = pattern["price"] * (1 - (volatility * 2)) if pattern["direction"] == "bullish" else pattern["price"] * (1 + (volatility * 2))
                current_price = self.trader.get_current_price()
                
                if ((pattern["direction"] == "bullish" and current_price < threshold) or
                    (pattern["direction"] == "bearish" and current_price > threshold)):
                    quality_assessment["quality_score"] *= 0.5
                    quality_assessment["quality_factors"].append("NEAR_INVALIDATION")
                
                pattern_quality[pattern["id"]] = quality_assessment
            
            return pattern_quality
        except Exception:
            return pattern_quality
            
    def _check_pattern_symmetry(self, pattern: Dict) -> bool:
        """Check if the harmonic pattern has symmetrical proportions"""
        try:
            # Different patterns have different ideal proportions
            if pattern["type"] == "BUTTERFLY":
                # Butterfly patterns typically have more symmetrical ratios
                return True if pattern.get("symmetry_score", 0) > 0.8 else False
            elif pattern["type"] == "GARTLEY":
                # Gartley patterns also tend to be symmetrical
                return True if pattern.get("symmetry_score", 0) > 0.75 else False
            elif pattern["type"] == "BAT":
                # Bat patterns have specific proportions
                return True if pattern.get("symmetry_score", 0) > 0.7 else False
            else:
                # Default check for other patterns
                return False
        except Exception:
            return False
            
    def _assess_pattern_completeness(self, pattern: Dict) -> float:
        """Assess how completely formed the pattern is on a scale of 0-10"""
        try:
            # Different pattern types have different completion criteria
            if "completeness_score" in pattern:
                return pattern["completeness_score"]
                
            # Default to using a base completeness based on pattern type
            if pattern["type"] == "BUTTERFLY":
                return 9  # Assuming butterfly patterns are fully formed when detected
            elif pattern["type"] == "GARTLEY":
                return 8  # Assuming gartley patterns are well formed
            elif pattern["type"] == "BAT":
                return 7  # Assuming bat patterns are mostly formed
            else:
                return 5  # Default middle score for other patterns
        except Exception:
            return 5  # Default middle score
            
    def _check_momentum_alignment(self, direction: str, df: pd.DataFrame) -> float:
        """Check if recent momentum aligns with the pattern direction"""
        try:
            if len(df) < 10:
                return 1.0  # Neutral if not enough data
                
            # Calculate momentum using last 5 vs previous 5 candles
            recent = df['close'].iloc[-5:].mean()
            previous = df['close'].iloc[-10:-5].mean()
            momentum = recent - previous
            
            # Calculate a normalized momentum score
            volatility = self._calculate_adaptive_volatility()
            normalized_momentum = momentum / (volatility * 5) if volatility > 0 else 0
            
            if direction == "bullish":
                if normalized_momentum > 0.5:
                    return 1.5  # Strong bullish alignment
                elif normalized_momentum > 0.1:
                    return 1.2  # Moderate bullish alignment
                elif normalized_momentum < -0.5:
                    return 0.7  # Strong bearish conflict
                elif normalized_momentum < -0.1:
                    return 0.9  # Moderate bearish conflict
                else:
                    return 1.0  # Neutral
            
            elif direction == "bearish":
                if normalized_momentum < -0.5:
                    return 1.5  # Strong bearish alignment
                elif normalized_momentum < -0.1:
                    return 1.2  # Moderate bearish alignment
                elif normalized_momentum > 0.5:
                    return 0.7  # Strong bullish conflict
                elif normalized_momentum > 0.1:
                    return 0.9  # Moderate bullish conflict
                else:
                    return 1.0  # Neutral
            
            return 1.0  # Default neutral
        except Exception:
            return 1.0  # Default neutral
            
    def _assess_candle_quality(self, pattern: Dict, df: pd.DataFrame) -> Dict:
        """Assess the quality of candles at pattern completion points"""
        result = {
            "score": 1.0,
            "factors": []
        }
        
        try:
            if len(df) < 3:
                return result
                
            # Look at the last 3 candles since pattern formation likely ended recently
            last_candles = df.iloc[-3:].copy()
            
            # Check for specific candle patterns at completion points
            if pattern["direction"] == "bullish":
                # Check for bullish engulfing
                if (last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1] and  # Current bullish
                    last_candles['open'].iloc[-1] <= last_candles['close'].iloc[-2] and  # Open below prev close
                    last_candles['close'].iloc[-1] > last_candles['open'].iloc[-2]):  # Close above prev open
                    
                    result["score"] = 1.3
                    result["factors"].append("BULLISH_ENGULFING")
                
                # Check for hammer pattern
                last_candle = last_candles.iloc[-1]
                if last_candle['close'] > last_candle['open']:
                    body_size = last_candle['close'] - last_candle['open']
                    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    
                    if lower_wick > body_size * 2 and upper_wick < body_size:
                        result["score"] = max(result["score"], 1.3)
                        result["factors"].append("HAMMER_PATTERN")
                
                # Check for strong bullish candle
                if (last_candle['close'] > last_candle['open'] and 
                    (last_candle['close'] - last_candle['open']) / (last_candle['high'] - last_candle['low']) > 0.6):
                    result["score"] = max(result["score"], 1.2)
                    result["factors"].append("STRONG_BULLISH_CANDLE")
                    
            elif direction == "bearish":
                # Check for bearish engulfing
                if (last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1] and  # Current bearish
                    last_candles['open'].iloc[-1] >= last_candles['close'].iloc[-2] and  # Open above prev close
                    last_candles['close'].iloc[-1] < last_candles['open'].iloc[-2]):  # Close below prev open
                    
                    result["score"] = 1.3
                    result["factors"].append("BEARISH_ENGULFING")
                
                # Check for shooting star pattern
                last_candle = last_candles.iloc[-1]
                if last_candle['close'] < last_candle['open']:
                    body_size = last_candle['open'] - last_candle['close']
                    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    
                    if upper_wick > body_size * 2 and lower_wick < body_size:
                        result["score"] = max(result["score"], 1.3)
                        result["factors"].append("SHOOTING_STAR")
                
                # Check for strong bearish candle
                if (last_candle['close'] < last_candle['open'] and 
                    (last_candle['open'] - last_candle['close']) / (last_candle['high'] - last_candle['low']) > 0.6):
                    result["score"] = max(result["score"], 1.2)
                    result["factors"].append("STRONG_BEARISH_CANDLE")
            
            return result
        except Exception:
            return result
            
    def _enhance_price_action_with_quality(self, price_action: Dict) -> Dict:
        """Enhance price action analysis with quality assessment"""
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return price_action
                
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 5:
                return price_action
                
            # Analyze bullish reversal quality if present
            if price_action["bullish_reversal"]:
                # Get the most recent 3 candles
                last_candles = df.iloc[-3:].copy()
                
                # Calculate quality factors
                
                # 1. Strength of reversal move
                candle_range = last_candles['high'].iloc[-1] - last_candles['low'].iloc[-1]
                body_size = abs(last_candles['close'].iloc[-1] - last_candles['open'].iloc[-1])
                relative_strength = body_size / candle_range if candle_range > 0 else 0
                
                # 2. Volume confirmation
                vol_increase = df['volume'].iloc[-1] / df['volume'].iloc[-5:-1].mean() if 'volume' in df.columns else 1.0
                
                # 3. Previous downtrend strength
                prev_move = df['close'].iloc[-4:-1].pct_change().sum()
                
                # Calculate quality score
                quality_score = 1.0
                quality_score *= (1.0 + min(0.5, relative_strength))
                quality_score *= min(1.5, vol_increase)
                quality_score *= (1.0 + abs(min(0, prev_move)) * 5)  # Stronger if previous move was down
                
                # Cap and store the quality
                price_action["reversal_quality"] = min(2.0, quality_score)
            
            # Analyze bearish reversal quality if present
            if price_action["bearish_reversal"]:
                # Get the most recent 3 candles
                last_candles = df.iloc[-3:].copy()
                
                # Calculate quality factors
                
                # 1. Strength of reversal move
                candle_range = last_candles['high'].iloc[-1] - last_candles['low'].iloc[-1]
                body_size = abs(last_candles['close'].iloc[-1] - last_candles['open'].iloc[-1])
                relative_strength = body_size / candle_range if candle_range > 0 else 0
                
                # 2. Volume confirmation
                vol_increase = df['volume'].iloc[-1] / df['volume'].iloc[-5:-1].mean() if 'volume' in df.columns else 1.0
                
                # 3. Previous uptrend strength
                prev_move = df['close'].iloc[-4:-1].pct_change().sum()
                
                # Calculate quality score
                quality_score = 1.0
                quality_score *= (1.0 + min(0.5, relative_strength))
                quality_score *= min(1.5, vol_increase)
                quality_score *= (1.0 + max(0, prev_move) * 5)  # Stronger if previous move was up
                
                # Cap and store the quality
                price_action["reversal_quality"] = min(2.0, quality_score)
            
            return price_action
        except Exception:
            return price_action
    
    def _check_pattern_symmetry(self, pattern: Dict) -> bool:
        """Check if the harmonic pattern has symmetrical proportions"""
        try:
            # Different patterns have different ideal proportions
            if pattern["type"] == "BUTTERFLY":
                # Butterfly patterns typically have more symmetrical ratios
                return True if pattern.get("symmetry_score", 0) > 0.8 else False
            elif pattern["type"] == "GARTLEY":
                # Gartley patterns also tend to be symmetrical
                return True if pattern.get("symmetry_score", 0) > 0.75 else False
            elif pattern["type"] == "BAT":
                # Bat patterns have specific proportions
                return True if pattern.get("symmetry_score", 0) > 0.7 else False
            else:
                # Default check for other patterns
                return False
        except Exception:
            return False
            
    def _assess_pattern_completeness(self, pattern: Dict) -> float:
        """Assess how completely formed the pattern is on a scale of 0-10"""
        try:
            # Different pattern types have different completion criteria
            if "completeness_score" in pattern:
                return pattern["completeness_score"]
                
            # Default to using a base completeness based on pattern type
            if pattern["type"] == "BUTTERFLY":
                return 9  # Assuming butterfly patterns are fully formed when detected
            elif pattern["type"] == "GARTLEY":
                return 8  # Assuming gartley patterns are well formed
            elif pattern["type"] == "BAT":
                return 7  # Assuming bat patterns are mostly formed
            else:
                return 5  # Default middle score for other patterns
        except Exception:
            return 5  # Default middle score
            
    def _check_momentum_alignment(self, direction: str, df: pd.DataFrame) -> float:
        """Check if recent momentum aligns with the pattern direction"""
        try:
            if len(df) < 10:
                return 1.0  # Neutral if not enough data
                
            # Calculate momentum using last 5 vs previous 5 candles
            recent = df['close'].iloc[-5:].mean()
            previous = df['close'].iloc[-10:-5].mean()
            momentum = recent - previous
            
            # Calculate a normalized momentum score
            volatility = self._calculate_adaptive_volatility()
            normalized_momentum = momentum / (volatility * 5) if volatility > 0 else 0
            
            if direction == "bullish":
                if normalized_momentum > 0.5:
                    return 1.5  # Strong bullish alignment
                elif normalized_momentum > 0.1:
                    return 1.2  # Moderate bullish alignment
                elif normalized_momentum < -0.5:
                    return 0.7  # Strong bearish conflict
                elif normalized_momentum < -0.1:
                    return 0.9  # Moderate bearish conflict
                else:
                    return 1.0  # Neutral
            
            elif direction == "bearish":
                if normalized_momentum < -0.5:
                    return 1.5  # Strong bearish alignment
                elif normalized_momentum < -0.1:
                    return 1.2  # Moderate bearish alignment
                elif normalized_momentum > 0.5:
                    return 0.7  # Strong bullish conflict
                elif normalized_momentum > 0.1:
                    return 0.9  # Moderate bullish conflict
                else:
                    return 1.0  # Neutral
            
            return 1.0  # Default neutral
        except Exception:
            return 1.0  # Default neutral
            
    def _assess_candle_quality(self, pattern: Dict, df: pd.DataFrame) -> Dict:
        """Assess the quality of candles at pattern completion points"""
        result = {
            "score": 1.0,
            "factors": []
        }
        
        try:
            if len(df) < 3:
                return result
                
            # Look at the last 3 candles since pattern formation likely ended recently
            last_candles = df.iloc[-3:].copy()
            
            # Check for specific candle patterns at completion points
            if pattern["direction"] == "bullish":
                # Check for bullish engulfing
                if (last_candles['close'].iloc[-1] > last_candles['open'].iloc[-1] and  # Current bullish
                    last_candles['open'].iloc[-1] <= last_candles['close'].iloc[-2] and  # Open below prev close
                    last_candles['close'].iloc[-1] > last_candles['open'].iloc[-2]):  # Close above prev open
                    
                    result["score"] = 1.3
                    result["factors"].append("BULLISH_ENGULFING")
                
                # Check for hammer pattern
                last_candle = last_candles.iloc[-1]
                if last_candle['close'] > last_candle['open']:
                    body_size = last_candle['close'] - last_candle['open']
                    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    
                    if lower_wick > body_size * 2 and upper_wick < body_size:
                        result["score"] = max(result["score"], 1.3)
                        result["factors"].append("HAMMER_PATTERN")
                
                # Check for strong bullish candle
                if (last_candle['close'] > last_candle['open'] and 
                    (last_candle['close'] - last_candle['open']) / (last_candle['high'] - last_candle['low']) > 0.6):
                    result["score"] = max(result["score"], 1.2)
                    result["factors"].append("STRONG_BULLISH_CANDLE")
                    
            elif direction == "bearish":
                # Check for bearish engulfing
                if (last_candles['close'].iloc[-1] < last_candles['open'].iloc[-1] and  # Current bearish
                    last_candles['open'].iloc[-1] >= last_candles['close'].iloc[-2] and  # Open above prev close
                    last_candles['close'].iloc[-1] < last_candles['open'].iloc[-2]):  # Close below prev open
                    
                    result["score"] = 1.3
                    result["factors"].append("BEARISH_ENGULFING")
                
                # Check for shooting star pattern
                last_candle = last_candles.iloc[-1]
                if last_candle['close'] < last_candle['open']:
                    body_size = last_candle['open'] - last_candle['close']
                    lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
                    upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
                    
                    if upper_wick > body_size * 2 and lower_wick < body_size:
                        result["score"] = max(result["score"], 1.3)
                        result["factors"].append("SHOOTING_STAR")
                
                # Check for strong bearish candle
                if (last_candle['close'] < last_candle['open'] and 
                    (last_candle['open'] - last_candle['close']) / (last_candle['high'] - last_candle['low']) > 0.6):
                    result["score"] = max(result["score"], 1.2)
                    result["factors"].append("STRONG_BEARISH_CANDLE")
            
            return result
        except Exception:
            return result
            
    def _enhance_price_action_with_quality(self, price_action: Dict) -> Dict:
        """Enhance price action analysis with quality assessment"""
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return price_action
                
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 5:
                return price_action
                
            # Analyze bullish reversal quality if present
            if price_action["bullish_reversal"]:
                # Get the most recent 3 candles
                last_candles = df.iloc[-3:].copy()
                
                # Calculate quality factors
                
                # 1. Strength of reversal move
                candle_range = last_candles['high'].iloc[-1] - last_candles['low'].iloc[-1]
                body_size = abs(last_candles['close'].iloc[-1] - last_candles['open'].iloc[-1])
                relative_strength = body_size / candle_range if candle_range > 0 else 0
                
                # 2. Volume confirmation
                vol_increase = df['volume'].iloc[-1] / df['volume'].iloc[-5:-1].mean() if 'volume' in df.columns else 1.0
                
                # 3. Previous downtrend strength
                prev_move = df['close'].iloc[-4:-1].pct_change().sum()
                
                # Calculate quality score
                quality_score = 1.0
                quality_score *= (1.0 + min(0.5, relative_strength))
                quality_score *= min(1.5, vol_increase)
                quality_score *= (1.0 + abs(min(0, prev_move)) * 5)  # Stronger if previous move was down
                
                # Cap and store the quality
                price_action["reversal_quality"] = min(2.0, quality_score)
            
            # Analyze bearish reversal quality if present
            if price_action["bearish_reversal"]:
                # Get the most recent 3 candles
                last_candles = df.iloc[-3:].copy()
                
                # Calculate quality factors
                
                # 1. Strength of reversal move
                candle_range = last_candles['high'].iloc[-1] - last_candles['low'].iloc[-1]
                body_size = abs(last_candles['close'].iloc[-1] - last_candles['open'].iloc[-1])
                relative_strength = body_size / candle_range if candle_range > 0 else 0
                
                # 2. Volume confirmation
                vol_increase = df['volume'].iloc[-1] / df['volume'].iloc[-5:-1].mean() if 'volume' in df.columns else 1.0
                
                # 3. Previous uptrend strength
                prev_move = df['close'].iloc[-4:-1].pct_change().sum()
                
                # Calculate quality score
                quality_score = 1.0
                quality_score *= (1.0 + min(0.5, relative_strength))
                quality_score *= min(1.5, vol_increase)
                quality_score *= (1.0 + max(0, prev_move) * 5)  # Stronger if previous move was up
                
                # Cap and store the quality
                price_action["reversal_quality"] = min(2.0, quality_score)
            
            return price_action
        except Exception:
            return price_action
    
    def _check_order_flow_confirmation(self, level_price: float, direction: str) -> Dict:
        """Check for order flow confirmation at Fibonacci level"""
        result = {
            "confirmed": False,
            "strength_multiplier": 1.0,
            "reasons": []
        }
        
        try:
            df = self.trader.price_data[self.trader.timeframe]
            volatility = self._calculate_adaptive_volatility()
            
            # Find candles that interacted with this level
            level_candles = []
            for i in range(max(0, len(df) - 50), len(df)):
                if direction == "buy" and abs(df['low'].iloc[i] - level_price) < volatility * 1.5:
                    level_candles.append(i)
                elif direction == "sell" and abs(df['high'].iloc[i] - level_price) < volatility * 1.5:
                    level_candles.append(i)
            
            if not level_candles:
                return result
                
            # Check for order absorption (large volume but small candle)
            for i in level_candles:
                candle_range = df['high'].iloc[i] - df['low'].iloc[i]
                body_range = abs(df['close'].iloc[i] - df['open'].iloc[i])
                relative_body_size = body_range / candle_range if candle_range > 0 else 0
                
                # Absorption occurs when high volume meets small body
                if df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5 and relative_body_size < 0.4:
                    result["confirmed"] = True
                    result["strength_multiplier"] *= 1.3
                    result["reasons"].append("ORDER_ABSORPTION")
                    break
            
            # Check for delta divergence (accumulation/distribution)
            if hasattr(df, 'delta') or hasattr(df, 'delta_ma'):
                # If we have order flow data like delta
                delta_key = 'delta' if 'delta' in df.columns else 'delta_ma' if 'delta_ma' in df.columns else None
                
                if delta_key:
                    if direction == "buy" and any(df[delta_key].iloc[i] > 0 for i in level_candles):
                        result["confirmed"] = True
                        result["strength_multiplier"] *= 1.2
                        result["reasons"].append("POSITIVE_DELTA")
                    elif direction == "sell" and any(df[delta_key].iloc[i] < 0 for i in level_candles):
                        result["confirmed"] = True
                        result["strength_multiplier"] *= 1.2
                        result["reasons"].append("NEGATIVE_DELTA")
                        
            return result
        except Exception:
            return result
    
    def _calculate_confluence_score(self, setup: Dict) -> Dict:
        """Calculate confluence score based on confluence factors"""
        score = 1.0
        for factor in setup.get("confluence_factors", []):
            if factor in ["HISTORICAL_VALIDATION", "VOLUME_CONFIRMATION", "PREVIOUS_REACTIONS"]:
                score *= 1.3
        setup["confluence_score"] = score
        return setup
    
    def get_volume_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure volume data is available in the dataframe"""
        if 'volume' not in df.columns:
            # Create dummy volume data if none exists
            df['volume'] = 1
        return df
    
    # New method to detect Wyckoff price action patterns
    def _detect_wyckoff_patterns(self, volatility: float) -> Dict:
        """Detect Wyckoff method price action patterns like springs and upthrusts"""
        result = {
            "springs": [],
            "upthrusts": []
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return result
                
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 20:  # Need sufficient data to detect patterns
                return result
                
            # Get recent zones that might act as support/resistance
            zones = []
            if hasattr(self.trader, 'zones'):
                zones = self.trader.zones
            
            # Get Fibonacci levels as potential support/resistance
            fib_levels = {}
            if hasattr(self.trader, 'fib_levels') and self.trader.timeframe in self.trader.fib_levels:
                fib_levels = self.trader.fib_levels[self.trader.timeframe]['levels']
            
            # Get market structure
            market_structure = {}
            if hasattr(self.trader, 'market_structure') and self.trader.timeframe in self.trader.market_structure:
                market_structure = self.trader.market_structure[self.trader.timeframe]
            
            # Find support/resistance levels from all sources
            support_levels = []
            resistance_levels = []
            
            # Extract from zones
            for zone in zones:
                if zone.type == "demand":
                    support_levels.append((zone.price_start + zone.price_end) / 2)
                else:
                    resistance_levels.append((zone.price_start + zone.price_end) / 2)
            
            # Extract from Fibonacci levels
            current_price = self.trader.get_current_price()
            if current_price:
                for _, level_price in fib_levels.items():
                    if level_price < current_price:
                        support_levels.append(level_price)
                    else:
                        resistance_levels.append(level_price)
            
            # Look for springs (support levels that were briefly broken and reclaimed)
            for support in support_levels:
                # Look back 10 candles for potential springs
                for i in range(max(5, len(df) - 10), len(df)):
                    # Check if price broke below support then reclaimed it
                    if (df['low'].iloc[i-1] < support - volatility * 0.3 and  # Break below support
                        df['close'].iloc[i-1] < support and  # Close below support
                        df['close'].iloc[i] > support and  # Reclaim above support
                        df['low'].iloc[i-2:i].min() > support - volatility * 3):  # Not a strong break (max 3x volatility)
                        
                        # Calculate quality metrics
                        volume_spike = df['volume'].iloc[i] / df['volume'].iloc[i-5:i].mean() if 'volume' in df.columns else 1.0
                        strength_multiplier = 1.5  # Base strength
                        
                        # Adjust by volume confirmation
                        if volume_spike > 1.5:
                            strength_multiplier = 1.8
                        
                        # Adjust by candle structure
                        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i] > df['high'].iloc[i-1]:
                            strength_multiplier = 2.0  # Strong bullish confirmation
                        
                        # Determine quality
                        quality = "medium"
                        if volume_spike > 1.5 and df['close'].iloc[i] > df['high'].iloc[i-1]:
                            quality = "high"
                        
                        # Add spring pattern to results
                        spring = {
                            "level": support,
                            "candle_index": i,
                            "strength_multiplier": strength_multiplier,
                            "quality": quality,
                            "volume_spike": volume_spike
                        }
                        result["springs"].append(spring)
            
            # Look for upthrusts (resistance levels that were briefly broken and failed)
            for resistance in resistance_levels:
                # Look back 10 candles for potential upthrusts
                for i in range(max(5, len(df) - 10), len(df)):
                    # Check if price broke above resistance then failed back below it
                    if (df['high'].iloc[i-1] > resistance + volatility * 0.3 and  # Break above resistance
                        df['close'].iloc[i-1] > resistance and  # Close above resistance
                        df['close'].iloc[i] < resistance and  # Failed back below resistance
                        df['high'].iloc[i-2:i].max() < resistance + volatility * 3):  # Not a strong break (max 3x volatility)
                        
                        # Calculate quality metrics
                        volume_spike = df['volume'].iloc[i] / df['volume'].iloc[i-5:i].mean() if 'volume' in df.columns else 1.0
                        strength_multiplier = 1.5  # Base strength
                        
                        # Adjust by volume confirmation
                        if volume_spike > 1.5:
                            strength_multiplier = 1.8
                        
                        # Adjust by candle structure
                        if df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i] < df['low'].iloc[i-1]:
                            strength_multiplier = 2.0  # Strong bearish confirmation
                        
                        # Determine quality
                        quality = "medium"
                        if volume_spike > 1.5 and df['close'].iloc[i] < df['low'].iloc[i-1]:
                            quality = "high"
                        
                        # Add upthrust pattern to results
                        upthrust = {
                            "level": resistance,
                            "candle_index": i,
                            "strength_multiplier": strength_multiplier,
                            "quality": quality,
                            "volume_spike": volume_spike
                        }
                        result["upthrusts"].append(upthrust)
            
            # Sort by quality and strength (highest first)
            if result["springs"]:
                result["springs"].sort(key=lambda x: (1 if x["quality"] == "high" else 0, x["strength_multiplier"]), reverse=True)
            
            if result["upthrusts"]:
                result["upthrusts"].sort(key=lambda x: (1 if x["quality"] == "high" else 0, x["strength_multiplier"]), reverse=True)
            
            return result
        except Exception:
            return result
    
    # NEW METHODS FOR SENTIMENT AND LIQUIDITY ANALYSIS
    
    def _analyze_market_sentiment(self) -> Dict:
        """
        Analyze market sentiment based on various factors including:
        - Price action momentum and divergences
        - Volume-based sentiment indicators
        - Market structure and swing behavior
        - Multi-timeframe momentum alignment
        
        Returns:
            Dict with sentiment analysis results
        """
        sentiment_data = {
            "bullish_strength": 0.0,
            "bearish_strength": 0.0,
            "overall_bias": "neutral",
            "sentiment_score": 0.0,  # -1.0 (bearish) to 1.0 (bullish)
            "confidence": 0.0,
            "key_factors": []
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return sentiment_data
                
            df = self.trader.price_data[self.trader.timeframe]
            
            # Ensure we have enough historical data for analysis
            min_required_bars = 30
            if len(df) < min_required_bars:
                return sentiment_data
            
            # IMPORTANT: Use fully historical data only - no current forming bar
            # This ensures consistency between backtest and live trading
            historical_df = df.iloc[:-1] if len(df) > min_required_bars else df
            
            # 1. Calculate price momentum indicators with fixed window sizes
            # Simple momentum: N bars ago vs M bars ago (fully historical)
            close_prices = historical_df['close'].values
            current_price = close_prices[-1]  # Last fully completed bar
            price_10_bars_ago = close_prices[-10] if len(close_prices) >= 10 else close_prices[0]
            momentum = (current_price - price_10_bars_ago) / price_10_bars_ago * 100
            
            # 2. Analyze higher timeframe alignment (with same historical-only approach)
            higher_tf_alignment = self._check_higher_timeframe_alignment()
            
            # 3. Analyze volume patterns (historical only)
            volume_sentiment = self._analyze_volume_sentiment(historical_df)
            
            # 4. Check for key swing points and structure
            structure_data = self._analyze_market_structure_sentiment()
            
            # 5. Consolidate sentiment data
            if momentum > 1.0:
                sentiment_data["bullish_strength"] += min(2.0, abs(momentum) * 0.2)
                sentiment_data["key_factors"].append("BULLISH_MOMENTUM")
            elif momentum < -1.0:
                sentiment_data["bearish_strength"] += min(2.0, abs(momentum) * 0.2)
                sentiment_data["key_factors"].append("BEARISH_MOMENTUM")
                
            # Add higher timeframe alignment influence
            if higher_tf_alignment["aligned"]:
                if higher_tf_alignment["direction"] == "bullish":
                    sentiment_data["bullish_strength"] += higher_tf_alignment["strength"]
                    sentiment_data["key_factors"].append("HTF_BULLISH_ALIGNMENT")
                elif higher_tf_alignment["direction"] == "bearish":
                    sentiment_data["bearish_strength"] += higher_tf_alignment["strength"]
                    sentiment_data["key_factors"].append("HTF_BEARISH_ALIGNMENT")
            
            # Add volume sentiment influence
            if volume_sentiment["bias"] == "bullish":
                sentiment_data["bullish_strength"] += volume_sentiment["strength"]
                sentiment_data["key_factors"].extend(volume_sentiment["factors"])
            elif volume_sentiment["bias"] == "bearish":
                sentiment_data["bearish_strength"] += volume_sentiment["strength"]
                sentiment_data["key_factors"].extend(volume_sentiment["factors"])
            
            # Add market structure influence
            if structure_data["structure_bias"] == "bullish":
                sentiment_data["bullish_strength"] += structure_data["bias_strength"]
                sentiment_data["key_factors"].extend(structure_data["key_factors"])
            elif structure_data["structure_bias"] == "bearish":
                sentiment_data["bearish_strength"] += structure_data["bias_strength"]
                sentiment_data["key_factors"].extend(structure_data["key_factors"])
            
            # Calculate final sentiment score (-1.0 to 1.0)
            sentiment_data["sentiment_score"] = min(1.0, max(-1.0, 
                (sentiment_data["bullish_strength"] - sentiment_data["bearish_strength"]) / 
                max(3.0, sentiment_data["bullish_strength"] + sentiment_data["bearish_strength"])
            ))
            
            # Determine overall bias
            if sentiment_data["sentiment_score"] > 0.2:
                sentiment_data["overall_bias"] = "bullish"
            elif sentiment_data["sentiment_score"] < -0.2:
                sentiment_data["overall_bias"] = "bearish"
            else:
                sentiment_data["overall_bias"] = "neutral"
            
            # Calculate confidence level
            combined_strength = sentiment_data["bullish_strength"] + sentiment_data["bearish_strength"]
            sentiment_data["confidence"] = min(1.0, combined_strength / 6.0)
                
            return sentiment_data
        except Exception as e:
            return sentiment_data
    
    def _check_higher_timeframe_alignment(self) -> Dict:
        """Check for sentiment alignment across higher timeframes"""
        result = {
            "aligned": False,
            "direction": "neutral",
            "strength": 0.0,
            "aligned_timeframes": []
        }
        
        try:
            current_tf = self.trader.timeframe
            higher_tfs = []
            
            # Get higher timeframes based on current timeframe
            tf_hierarchy = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
            if current_tf in tf_hierarchy:
                current_idx = tf_hierarchy.index(current_tf)
                higher_tfs = tf_hierarchy[current_idx+1:]
            
            bullish_count = 0
            bearish_count = 0
            aligned_tfs = []
            
            # Check momentum and bias in each higher timeframe
            for tf in higher_tfs:
                if hasattr(self.trader, 'price_data') and tf in self.trader.price_data:
                    df = self.trader.price_data[tf]
                    if len(df) < 10:
                        continue
                    
                    # Use historical bars only (skip the last forming bar in live trading)
                    # This ensures consistency between backtesting and live trading
                    historical_df = df.iloc[:-1] if len(df) > 10 else df
                    
                    if len(historical_df) < 5:  # Need at least 5 bars
                        continue
                    
                    # Calculate momentum over historical bars only
                    momentum = historical_df['close'].iloc[-1] - historical_df['close'].iloc[-5]
                    normalized_momentum = momentum / historical_df['close'].iloc[-5]
                    
                    # Determine timeframe bias
                    if normalized_momentum > 0.002:
                        bullish_count += 1
                        aligned_tfs.append(tf)
                    elif normalized_momentum < -0.002:
                        bearish_count += 1
                        aligned_tfs.append(tf)
            
            # Determine overall alignment
            if bullish_count > 0 or bearish_count > 0:
                result["aligned"] = True
                if bullish_count > bearish_count:
                    result["direction"] = "bullish"
                    result["strength"] = min(2.0, bullish_count * 0.5)
                else:
                    result["direction"] = "bearish"
                    result["strength"] = min(2.0, bearish_count * 0.5)
                result["aligned_timeframes"] = aligned_tfs
                
            return result
        except Exception:
            return result
            
    def _analyze_volume_sentiment(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volume patterns to determine sentiment
        
        Returns dict with volume-based sentiment analysis
        """
        result = {
            "bias": "neutral",
            "strength": 0.0,
            "factors": []
        }
        
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return result
                
            # Make a copy to avoid warnings
            data = df.copy()
            
            # Check if we need to calculate synthetic volume
            if data['volume'].isnull().all() or (data['volume'] == 0).all():
                data = self.get_volume_data(data)
                if 'volume' not in data.columns:
                    return result
            
            # IMPORTANT: Make sure we're only using historical data by slicing off
            # the last bar which might be incomplete in live trading
            if len(data) > 20:  # Ensure we have enough data after removing the last bar
                historical_data = data.iloc[:-1].copy()
            else:
                historical_data = data.copy()
            
            # Get recent data for analysis (from historical data only)
            recent_data = historical_data.iloc[-20:].copy()
            
            # 1. Check volume trend
            volume_sma5 = recent_data['volume'].rolling(5).mean()
            volume_sma10 = recent_data['volume'].rolling(10).mean()
            
            # Volume trend increasing?
            vol_increasing = volume_sma5.iloc[-1] > volume_sma10.iloc[-1]
            
            # 2. Check bullish/bearish volume
            # Add a column identifying candle type
            recent_data['bullish_candle'] = recent_data['close'] > recent_data['open']
            
            # Calculate volume for bullish vs bearish candles
            last_5_bulls = recent_data.iloc[-5:][recent_data.iloc[-5:]['bullish_candle'] == True]
            last_5_bears = recent_data.iloc[-5:][recent_data.iloc[-5:]['bullish_candle'] == False]
            
            avg_bull_vol = last_5_bulls['volume'].mean() if len(last_5_bulls) > 0 else 0
            avg_bear_vol = last_5_bears['volume'].mean() if len(last_5_bears) > 0 else 0
            
            # 3. Check volume on recent price moves
            last_candle = recent_data.iloc[-1]
            prev_candle = recent_data.iloc[-2] if len(recent_data) > 1 else None
            
            # 4. Determine volume-based sentiment
            if avg_bull_vol > avg_bear_vol * 1.2:
                result["bias"] = "bullish"
                result["strength"] = min(1.5, (avg_bull_vol / max(1, avg_bear_vol)) * 0.5)
                result["factors"].append("HIGHER_BULLISH_VOLUME")
            elif avg_bear_vol > avg_bull_vol * 1.2:
                result["bias"] = "bearish"
                result["strength"] = min(1.5, (avg_bear_vol / max(1, avg_bull_vol)) * 0.5)
                result["factors"].append("HIGHER_BEARISH_VOLUME")
                
            # Add additional factors based on recent volume patterns
            if prev_candle is not None:
                # Check for volume spike
                if last_candle['volume'] > volume_sma5.iloc[-2] * 1.5:
                    if last_candle['bullish_candle']:
                        result["bias"] = "bullish" if result["bias"] == "neutral" else result["bias"]
                        result["strength"] += 0.3
                        result["factors"].append("BULLISH_VOLUME_SPIKE")
                    else:
                        result["bias"] = "bearish" if result["bias"] == "neutral" else result["bias"]
                        result["strength"] += 0.3
                        result["factors"].append("BEARISH_VOLUME_SPIKE")
                        
                # Check for climax volume (potential reversal)
                if last_candle['volume'] > volume_sma10.iloc[-2] * 2.0:
                    # Price stalling with huge volume might indicate reversal
                    price_change_pct = abs(last_candle['close'] - last_candle['open']) / last_candle['open']
                    if price_change_pct < 0.0005:  # Small price movement despite high volume
                        result["factors"].append("VOLUME_CLIMAX_REVERSAL_POTENTIAL")
                        # Don't adjust bias yet, as direction is uncertain
            
            return result
        except Exception:
            return result
            
    def _analyze_market_structure_sentiment(self) -> Dict:
        """Analyze market structure (highs/lows pattern) for sentiment"""
        result = {
            "structure_bias": "neutral",
            "bias_strength": 0.0,
            "key_factors": []
        }
        
        try:
            if not hasattr(self.trader, 'market_structure') or self.trader.timeframe not in self.trader.market_structure:
                return result
                
            # Get market structure data
            market_structure = self.trader.market_structure[self.trader.timeframe]
            trend = market_structure.get("trend", "neutral")
            
            # Get swing points
            swings = market_structure.get("swings", [])
            if len(swings) < 4:
                return result
                
            # Get recent swings
            recent_swings = swings[-4:]
            
            # Check for higher highs and higher lows (bullish structure)
            highs = [s for s in recent_swings if s["type"] == "high"]
            lows = [s for s in recent_swings if s["type"] == "low"]
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Higher highs and higher lows = bullish
                if highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
                    result["structure_bias"] = "bullish"
                    result["bias_strength"] = 1.0
                    result["key_factors"].append("HIGHER_HIGHS_HIGHER_LOWS")
                    
                # Lower highs and lower lows = bearish
                elif highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
                    result["structure_bias"] = "bearish"
                    result["bias_strength"] = 1.0
                    result["key_factors"].append("LOWER_HIGHS_LOWER_LOWS")
                    
                # Higher highs but lower lows = distribution or weakening uptrend
                elif highs[-1]["price"] > highs[-2]["price"] and lows[-1]["price"] < lows[-2]["price"]:
                    result["structure_bias"] = "neutral"
                    result["key_factors"].append("EXPANDING_VOLATILITY")
                    
                # Lower highs but higher lows = accumulation or compression
                elif highs[-1]["price"] < highs[-2]["price"] and lows[-1]["price"] > lows[-2]["price"]:
                    result["structure_bias"] = "neutral"
                    result["key_factors"].append("COMPRESSING_VOLATILITY")
            
            # If we have a defined trend, add its influence
            if trend == "uptrend":
                result["structure_bias"] = "bullish" if result["structure_bias"] == "neutral" else result["structure_bias"]
                result["bias_strength"] += 0.5
                result["key_factors"].append("UPTREND")
            elif trend == "downtrend":
                result["structure_bias"] = "bearish" if result["structure_bias"] == "neutral" else result["structure_bias"]
                result["bias_strength"] += 0.5
                result["key_factors"].append("DOWNTREND")
                
            return result
        except Exception:
            return result
            
    def _analyze_liquidity(self, price_level: float, direction: str) -> Dict:
        """
        Analyze market liquidity around a specific price level
        
        Args:
            price_level: The price level to analyze
            direction: 'buy' or 'sell' direction
            
        Returns:
            Dict with liquidity analysis
        """
        liquidity_data = {
            "sufficient_liquidity": True,
            "liquidity_score": 0.0,  # 0.0 (low liquidity) to 1.0 (high liquidity)
            "expected_slippage": 0.0,  # Expected slippage in pips
            "concerns": [],
            "favorable_factors": []
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return liquidity_data
                
            df = self.trader.price_data[self.trader.timeframe]
            if len(df) < 30 or 'volume' not in df.columns:
                return liquidity_data
                
            # IMPORTANT: Use historical bars only to ensure consistency between backtest and live
            # Skip the currently forming bar which might be incomplete in live trading
            historical_df = df.iloc[:-1] if len(df) > 30 else df
            
            # 1. Calculate typical volume and volatility
            recent_volume = historical_df['volume'].iloc[-20:].mean()
            volatility = self._calculate_adaptive_volatility()
            
            # 2. Identify liquidity zones (high volume nodes)
            # Check if any significant clusters of trades happened near the price level
            price_history = historical_df['close'].iloc[-100:].values if len(historical_df) >= 100 else historical_df['close'].values
            volume_history = historical_df['volume'].iloc[-100:].values if len(historical_df) >= 100 else historical_df['volume'].values
            
            # Create price bins
            bin_size = volatility * 0.5
            price_min = min(price_history) - bin_size
            price_max = max(price_history) + bin_size
            bins = int((price_max - price_min) / bin_size)
            
            # Skip computation if we don't have enough data points
            if bins < 3 or bins > 100:
                liquidity_data["concerns"].append("INSUFFICIENT_PRICE_DATA")
                return liquidity_data
                
            # Get histogram data
            hist, bin_edges = np.histogram(price_history, bins=bins, weights=volume_history)
            
            # Find which bin our price level is in
            bin_index = None
            for i in range(len(bin_edges) - 1):
                if bin_edges[i] <= price_level <= bin_edges[i+1]:
                    bin_index = i
                    break
            
            # 3. Analyze liquidity at the price level
            if bin_index is not None:
                # Get volume at this price level
                volume_at_level = hist[bin_index]
                
                # Calculate relative liquidity (compare to average)
                relative_liquidity = volume_at_level / max(1, recent_volume)
                
                # Assign liquidity score
                if relative_liquidity > 1.5:  # High liquidity
                    liquidity_data["liquidity_score"] = min(1.0, relative_liquidity / 3)
                    liquidity_data["favorable_factors"].append("HIGH_HISTORICAL_VOLUME")
                elif relative_liquidity < 0.5:  # Low liquidity
                    liquidity_data["liquidity_score"] = max(0.1, relative_liquidity / 2)
                    liquidity_data["concerns"].append("LOW_HISTORICAL_VOLUME")
                else:  # Average liquidity
                    liquidity_data["liquidity_score"] = 0.5
                
                # 4. Estimate potential slippage based on historical price gaps
                price_jumps = []
                for i in range(1, len(historical_df)):
                    if direction == 'buy':
                        # For buy, look at gaps between close and next open (upward)
                        gap = historical_df['open'].iloc[i] - historical_df['close'].iloc[i-1]
                        if gap > 0:
                            price_jumps.append(gap)
                    else:  # sell
                        # For sell, look at gaps between close and next open (downward)
                        gap = historical_df['close'].iloc[i-1] - historical_df['open'].iloc[i]
                        if gap > 0:
                            price_jumps.append(gap)
                
                if price_jumps:
                    avg_jump = sum(price_jumps) / len(price_jumps)
                    max_jump = max(price_jumps)
                    
                    # Adjust expected slippage based on liquidity score
                    liquidity_modifier = 2.0 - liquidity_data["liquidity_score"]
                    liquidity_data["expected_slippage"] = avg_jump * liquidity_modifier
                    
                    if max_jump > volatility * 2:
                        liquidity_data["concerns"].append("HISTORY_OF_LARGE_GAPS")
                
                # 5. Check for liquidity concerns
                if liquidity_data["liquidity_score"] < 0.3:
                    liquidity_data["sufficient_liquidity"] = False
                    liquidity_data["concerns"].append("INSUFFICIENT_LIQUIDITY")
                
                # 6. Consider spread widening at this level
                # If this is a significant S/R level, spreads might widen
                if self._is_major_support_resistance_level(price_level):
                    liquidity_data["concerns"].append("POTENTIAL_SPREAD_WIDENING")
                    liquidity_data["expected_slippage"] *= 1.2
                    
            else:
                # No data for this price level
                liquidity_data["concerns"].append("NO_HISTORICAL_DATA_AT_LEVEL")
                liquidity_data["sufficient_liquidity"] = False
            
            return liquidity_data
        except Exception:
            liquidity_data["concerns"].append("ANALYSIS_ERROR")
            return liquidity_data
    
    def _is_major_support_resistance_level(self, price_level: float) -> bool:
        """Check if the price level is a major support/resistance level"""
        try:
            if not hasattr(self.trader, 'zones'):
                return False
                
            # Check if price level is near any significant zone
            volatility = self._calculate_adaptive_volatility()
            proximity_threshold = volatility * 1.5
            
            for zone in self.trader.zones:
                if abs(zone.price_start - price_level) < proximity_threshold or abs(zone.price_end - price_level) < proximity_threshold:
                    # Check if it's a strong zone
                    if zone.strength > 0.7:
                        return True
            
            return False
        except Exception:
            return False
    
    def _integrate_sentiment_liquidity_analysis(self, setup: Dict) -> Dict:
        """
        Integrate sentiment and liquidity analysis results into the trade setup.
        Also adds institutional footprint analysis around Fibonacci levels.
        
        Args:
            setup: The current trade setup
            
        Returns:
            Updated trade setup with sentiment, liquidity, and institutional factors
        """
        if not setup["valid_setup"]:
            return setup
            
        try:
            # Get sentiment analysis
            sentiment_data = self._analyze_market_sentiment()
            
            # Get liquidity analysis for the entry price
            liquidity_data = self._analyze_liquidity(
                setup["entry_price"], 
                setup["direction"]
            )
            
            # Get direction in normalized form
            direction_normalized = "buy" if setup["direction"].lower() in ["buy", "long"] else "sell"
            
            # Apply sentiment adjustment to setup strength
            if setup["direction"] == "buy" and sentiment_data["overall_bias"] == "bullish":
                # Bullish setup with bullish sentiment - strengthen
                setup["strength"] *= 1.0 + min(0.5, sentiment_data["confidence"] * sentiment_data["sentiment_score"])
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                setup["confluence_factors"].append("ALIGNED_BULLISH_SENTIMENT")
                
            elif setup["direction"] == "sell" and sentiment_data["overall_bias"] == "bearish":
                # Bearish setup with bearish sentiment - strengthen
                setup["strength"] *= 1.0 + min(0.5, sentiment_data["confidence"] * abs(sentiment_data["sentiment_score"]))
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                setup["confluence_factors"].append("ALIGNED_BEARISH_SENTIMENT")
                
            elif setup["direction"] == "buy" and sentiment_data["overall_bias"] == "bearish":
                # Bullish setup with bearish sentiment - weaken
                setup["strength"] *= max(0.5, 1.0 - min(0.5, sentiment_data["confidence"] * abs(sentiment_data["sentiment_score"])))
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                setup["confluence_factors"].append("AGAINST_BEARISH_SENTIMENT")
                
            elif setup["direction"] == "sell" and sentiment_data["overall_bias"] == "bullish":
                # Bearish setup with bullish sentiment - weaken
                setup["strength"] *= max(0.5, 1.0 - min(0.5, sentiment_data["confidence"] * sentiment_data["sentiment_score"]))
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                setup["confluence_factors"].append("AGAINST_BULLISH_SENTIMENT")
            
            # Apply liquidity adjustments
            if liquidity_data["sufficient_liquidity"]:
                # Good liquidity - slight positive adjustment
                setup["strength"] *= 1.0 + min(0.2, liquidity_data["liquidity_score"] * 0.2)
                
                if liquidity_data["liquidity_score"] > 0.7:
                    if "confluence_factors" not in setup:
                        setup["confluence_factors"] = []
                    setup["confluence_factors"].append("HIGH_LIQUIDITY_ZONE")
            else:
                # Poor liquidity - negative adjustment
                setup["strength"] *= max(0.6, 1.0 - (0.4 * (1.0 - liquidity_data["liquidity_score"])))
                
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                setup["confluence_factors"].append("LOW_LIQUIDITY_WARNING")
                
            # === NEW: INSTITUTIONAL FOOTPRINT ANALYSIS ===
            
            # Get Fibonacci level from the setup
            fib_level = setup.get("fibonacci_level")
            if not fib_level and "entry_price" in setup:
                # Use entry price as proxy if actual Fibonacci level not provided
                fib_level = setup["entry_price"]
            
            if fib_level:
                # 1. Check for stop order clusters around this Fibonacci level
                stop_clusters = self._detect_stop_clusters_at_fib_levels(fib_level, direction_normalized)
                
                if stop_clusters["stop_cluster_detected"]:
                    if stop_clusters["cluster_strength"] > 0.6:
                        # Strong stop cluster - consider it in setup strength
                        if ((direction_normalized == "buy" and stop_clusters["cluster_type"] == "sell_stops") or
                            (direction_normalized == "sell" and stop_clusters["cluster_type"] == "buy_stops")):
                            # Stop cluster in the same direction as our trade - potential liquidity for our move
                            setup["strength"] *= 1.0 + min(0.3, stop_clusters["cluster_strength"] * 0.3)
                            
                            if "confluence_factors" not in setup:
                                setup["confluence_factors"] = []
                            if stop_clusters["potential_target"]:
                                setup["confluence_factors"].append(f"HIGH_LIQUIDITY_{stop_clusters['cluster_type'].upper()}")
                            else:
                                setup["confluence_factors"].append(f"MODERATE_LIQUIDITY_{stop_clusters['cluster_type'].upper()}")
                    
                    # Add stop cluster data to setup
                    setup["institutional_stop_clusters"] = stop_clusters
                
                # 2. Check for institutional accumulation/distribution at this level
                inst_activity = self._detect_institutional_accumulation(fib_level, direction_normalized)
                
                if inst_activity["institutional_activity_detected"]:
                    # Found institutional activity at this level
                    activity_matches_direction = ((direction_normalized == "buy" and inst_activity["activity_type"] == "accumulation") or
                                                (direction_normalized == "sell" and inst_activity["activity_type"] == "distribution"))
                    
                    if activity_matches_direction:
                        # Activity aligns with our trade direction - strong bullish signal
                        conf_boost = min(0.4, inst_activity["confidence_score"] * 0.4)
                        setup["strength"] *= 1.0 + conf_boost
                        
                        if "confluence_factors" not in setup:
                            setup["confluence_factors"] = []
                            
                        if inst_activity["phase"] in ["late", "complete"]:
                            # Late-phase accumulation/distribution has higher probability of directional move
                            setup["confluence_factors"].append(f"COMPLETED_{inst_activity['activity_type'].upper()}")
                            
                            # Check for potential markup/markdown transition
                            transition = self._detect_markup_markdown_transition(fib_level, direction_normalized)
                            if transition["transition_detected"]:
                                setup["strength"] *= 1.0 + min(0.3, transition["confidence_score"] * 0.3)
                                setup["confluence_factors"].append(f"{transition['transition_type'].upper()}_TRANSITION")
                                
                                # Add transition data to setup
                                setup["institutional_transition"] = transition
                        else:
                            # Early/mid phase still positive but less strong
                            setup["confluence_factors"].append(f"EARLY_{inst_activity['activity_type'].upper()}")
                    else:
                        # Activity against our trade direction - reduce strength
                        conf_reduction = min(0.5, inst_activity["confidence_score"] * 0.5)
                        setup["strength"] *= max(0.5, 1.0 - conf_reduction)
                        
                        if "confluence_factors" not in setup:
                            setup["confluence_factors"] = []
                        setup["confluence_factors"].append(f"AGAINST_{inst_activity['activity_type'].upper()}")
                    
                    # Add institutional activity data to setup
                    setup["institutional_activity"] = inst_activity
                
                # 3. Check for order blocks at this Fibonacci level
                order_blocks = self._identify_fibonacci_order_blocks(fib_level, direction_normalized)
                
                if order_blocks["order_block_detected"]:
                    order_block_matches_direction = ((direction_normalized == "buy" and order_blocks["order_block_type"] == "bullish") or
                                                    (direction_normalized == "sell" and order_blocks["order_block_type"] == "bearish"))
                    
                    if order_block_matches_direction:
                        # Order block aligns with our trade - strong signal
                        ob_boost = min(0.35, order_blocks["strength"] * 0.35)
                        setup["strength"] *= 1.0 + ob_boost
                        
                        if "confluence_factors" not in setup:
                            setup["confluence_factors"] = []
                        
                        confidence_count = len(order_blocks["confidence_factors"])
                        if confidence_count >= 3:
                            setup["confluence_factors"].append(f"STRONG_{order_blocks['order_block_type'].upper()}_ORDER_BLOCK")
                        else:
                            setup["confluence_factors"].append(f"{order_blocks['order_block_type'].upper()}_ORDER_BLOCK")
                            
                        # Add specific confidence factors from the order block
                        for factor in order_blocks["confidence_factors"]:
                            if factor not in setup["confluence_factors"]:
                                setup["confluence_factors"].append(factor)
                    else:
                        # Order block against our trade - reduce strength
                        setup["strength"] *= max(0.6, 1.0 - (order_blocks["strength"] * 0.4))
                        
                        if "confluence_factors" not in setup:
                            setup["confluence_factors"] = []
                        setup["confluence_factors"].append(f"AGAINST_{order_blocks['order_block_type'].upper()}_ORDER_BLOCK")
                    
                    # Add order block data to setup
                    setup["institutional_order_block"] = order_blocks
                    
                # 4. Check for higher timeframe alignment
                mtf_alignment = self._check_higher_timeframe_alignment()
                if mtf_alignment["aligned"]:
                    alignment_matches_direction = ((direction_normalized == "buy" and mtf_alignment["direction"] == "bullish") or
                                                (direction_normalized == "sell" and mtf_alignment["direction"] == "bearish"))
                    
                    if alignment_matches_direction:
                        # Higher timeframes aligned with our trade - boost confidence
                        tf_boost = min(0.25, mtf_alignment["alignment_strength"] * 0.25)
                        setup["strength"] *= 1.0 + tf_boost
                        
                        if "confluence_factors" not in setup:
                            setup["confluence_factors"] = []
                        setup["confluence_factors"].append("MTF_ALIGNMENT")
                        
                        # Add aligned timeframes
                        for tf in mtf_alignment["aligned_timeframes"]:
                            aligned_tf_factor = f"ALIGNED_{tf}"
                            if aligned_tf_factor not in setup["confluence_factors"]:
                                setup["confluence_factors"].append(aligned_tf_factor)
                    else:
                        # Higher timeframes against our trade - reduce confidence
                        tf_reduction = min(0.25, mtf_alignment["alignment_strength"] * 0.25)
                        setup["strength"] *= max(0.75, 1.0 - tf_reduction)
                        
                        if "confluence_factors" not in setup:
                            setup["confluence_factors"] = []
                        setup["confluence_factors"].append("MTF_MISALIGNMENT")
                    
                    # Add MTF alignment data to setup
                    setup["mtf_alignment"] = mtf_alignment
            
            # Add key factors from sentiment analysis
            for factor in sentiment_data["key_factors"]:
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                if factor not in setup["confluence_factors"]:
                    setup["confluence_factors"].append(factor)
            
            # Add liquidity concerns
            for concern in liquidity_data["concerns"]:
                if "confluence_factors" not in setup:
                    setup["confluence_factors"] = []
                if concern not in setup["confluence_factors"]:
                    setup["confluence_factors"].append(concern)
            
            # Add detailed sentiment and liquidity data to setup
            setup["sentiment_analysis"] = {
                "overall_bias": sentiment_data["overall_bias"],
                "sentiment_score": sentiment_data["sentiment_score"],
                "confidence": sentiment_data["confidence"]
            }
            
            setup["liquidity_analysis"] = {
                "sufficient_liquidity": liquidity_data["sufficient_liquidity"],
                "liquidity_score": liquidity_data["liquidity_score"],
                "expected_slippage": liquidity_data["expected_slippage"]
            }
            
            # Calculate overall institutional confidence score
            inst_confidence = 0.0
            inst_factors = 0
            
            if "institutional_stop_clusters" in setup and setup["institutional_stop_clusters"]["stop_cluster_detected"]:
                inst_confidence += setup["institutional_stop_clusters"]["cluster_strength"] * 0.8
                inst_factors += 1
                
            if "institutional_activity" in setup:
                inst_confidence += setup["institutional_activity"]["confidence_score"] * 1.0
                inst_factors += 1
                
            if "institutional_transition" in setup:
                inst_confidence += setup["institutional_transition"]["confidence_score"] * 1.2
                inst_factors += 1
                
            if "institutional_order_block" in setup:
                inst_confidence += setup["institutional_order_block"]["strength"] * 0.9
                inst_factors += 1
                
            if "mtf_alignment" in setup:
                inst_confidence += setup["mtf_alignment"]["alignment_strength"] * 0.7
                inst_factors += 1
                
            # Calculate final institutional confidence score if we have factors
            if inst_factors > 0:
                setup["institutional_confidence"] = round(min(1.0, inst_confidence / inst_factors), 2)
                # Add a descriptive rating
                conf = setup["institutional_confidence"]
                if conf >= 0.8:
                    setup["institutional_rating"] = "Very High"
                elif conf >= 0.65:
                    setup["institutional_rating"] = "High"
                elif conf >= 0.5:
                    setup["institutional_rating"] = "Moderate"
                elif conf >= 0.3:
                    setup["institutional_rating"] = "Low"
                else:
                    setup["institutional_rating"] = "Very Low"
            
            return setup
        except Exception:
            return setup
    
    def _detect_stop_clusters_at_fib_levels(self, price_level: float, direction: str) -> Dict:
        """
        Detect stop order clusters near Fibonacci levels. Stop clusters typically form around key 
        Fibonacci levels and are often targets for institutional players before reversals.
        
        Args:
            price_level: The Fibonacci price level to analyze
            direction: 'buy' or 'sell' - the direction of the potential trade
            
        Returns:
            Dict with stop cluster analysis results
        """
        result = {
            "stop_cluster_detected": False,
            "cluster_strength": 0.0,  # 0.0 to 1.0 scale
            "cluster_type": None,     # "buy_stops" or "sell_stops"
            "potential_target": False, # Whether this cluster is a likely institutional target
            "stop_hunt_evidence": []  # List of evidence for stop hunting
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return result
                
            df = self.trader.price_data[self.trader.timeframe].copy()
            if len(df) < 50:
                return result
            
            # Calculate volatility for proximity thresholds    
            volatility = self._calculate_adaptive_volatility()
            proximity_threshold = volatility * 1.2
            
            # Get recent price action around this level
            level_touches = []
            for i in range(max(0, len(df) - 100), len(df)):
                # Check if price came near this level
                if abs(df['high'].iloc[i] - price_level) < proximity_threshold or abs(df['low'].iloc[i] - price_level) < proximity_threshold:
                    level_touches.append(i)
            
            # Detect potential stop clusters based on price behavior
            if direction == "buy":
                # For long trades, look for sell-stops likely placed below support
                # (common at Fibonacci support levels)
                
                # Count rapid price movements after breaking the level
                stop_hunt_count = 0
                strong_hunt_evidence = 0
                
                for i in level_touches:
                    if i+1 >= len(df):
                        continue
                        
                    # Look for candles that broke below the level (triggering stops)
                    if df['low'].iloc[i] < price_level - (volatility * 0.5):
                        # Then quickly reversed back above (stop hunt)
                        if i+3 < len(df) and df['close'].iloc[i+2] > price_level:
                            stop_hunt_count += 1
                            
                            # If volume spiked during the stop hunt, it's stronger evidence
                            if 'volume' in df.columns and df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5:
                                strong_hunt_evidence += 1
                                result["stop_hunt_evidence"].append(f"VOLUME_SPIKE_AT_{i}")
            
            elif direction == "sell":
                # For short trades, look for buy-stops likely placed above resistance
                # (common at Fibonacci resistance levels)
                
                # Count rapid price movements after breaking the level
                stop_hunt_count = 0
                strong_hunt_evidence = 0
                
                for i in level_touches:
                    if i+1 >= len(df):
                        continue
                        
                    # Look for candles that broke above the level (triggering stops)
                    if df['high'].iloc[i] > price_level + (volatility * 0.5):
                        # Then quickly reversed back below (stop hunt)
                        if i+3 < len(df) and df['close'].iloc[i+2] < price_level:
                            stop_hunt_count += 1
                            
                            # If volume spiked during the stop hunt, it's stronger evidence
                            if 'volume' in df.columns and df['volume'].iloc[i] > df['volume'].iloc[i-5:i].mean() * 1.5:
                                strong_hunt_evidence += 1
                                result["stop_hunt_evidence"].append(f"VOLUME_SPIKE_AT_{i}")
            
            # Calculate cluster strength based on evidence
            if stop_hunt_count > 0:
                result["stop_cluster_detected"] = True
                result["cluster_strength"] = min(1.0, (stop_hunt_count * 0.2) + (strong_hunt_evidence * 0.3))
                result["cluster_type"] = "sell_stops" if direction == "buy" else "buy_stops"
                
                # If we've seen multiple stop hunts with volume confirmation, it's likely a target
                if stop_hunt_count >= 2 and strong_hunt_evidence >= 1:
                    result["potential_target"] = True
                    
            # Check for recent price structure suggesting stop placement
            current_idx = len(df) - 1
            
            # For buy setups, stops are likely below recent lows near Fibonacci support
            if direction == "buy":
                recent_lows = []
                for i in range(max(0, current_idx - 20), current_idx):
                    # Find local lows within 2x volatility of the level
                    if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                        df['low'].iloc[i] < df['low'].iloc[i+1] and
                        abs(df['low'].iloc[i] - price_level) < volatility * 2):
                        recent_lows.append(df['low'].iloc[i])
                
                if recent_lows:
                    # Cluster of lows near the level suggests stop placement
                    avg_low = sum(recent_lows) / len(recent_lows)
                    if abs(avg_low - price_level) < volatility * 1.2:
                        result["stop_cluster_detected"] = True
                        result["cluster_strength"] = max(result["cluster_strength"], 0.7)
                        result["cluster_type"] = "sell_stops"
                        result["stop_hunt_evidence"].append("CLUSTERED_LOWS")
            
            # For sell setups, stops are likely above recent highs near Fibonacci resistance
            elif direction == "sell":
                recent_highs = []
                for i in range(max(0, current_idx - 20), current_idx):
                    # Find local highs within 2x volatility of the level
                    if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                        df['high'].iloc[i] > df['high'].iloc[i+1] and
                        abs(df['high'].iloc[i] - price_level) < volatility * 2):
                        recent_highs.append(df['high'].iloc[i])
                
                if recent_highs:
                    # Cluster of highs near the level suggests stop placement
                    avg_high = sum(recent_highs) / len(recent_highs)
                    if abs(avg_high - price_level) < volatility * 1.2:
                        result["stop_cluster_detected"] = True
                        result["cluster_strength"] = max(result["cluster_strength"], 0.7)
                        result["cluster_type"] = "buy_stops"
                        result["stop_hunt_evidence"].append("CLUSTERED_HIGHS")
            
            return result
        except Exception:
            return result
    
    def _detect_institutional_accumulation(self, price_level: float, direction: str) -> Dict:
        """
        Detect institutional accumulation or distribution patterns at Fibonacci levels.
        
        Institutions typically accumulate positions gradually through absorption candles,
        delta divergence, and specific volume patterns while keeping price stable near key levels.
        
        Args:
            price_level: The Fibonacci price level to analyze
            direction: 'buy' or 'sell' - expected directional bias after accumulation/distribution
            
        Returns:
            Dict with institutional activity analysis results
        """
        result = {
            "institutional_activity_detected": False,
            "activity_type": None,  # "accumulation" or "distribution"
            "confidence_score": 0.0,  # 0.0 to 1.0 scale
            "phase": None,  # "early", "mid", "late", or "complete" 
            "evidence": []
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return result
                
            df = self.trader.price_data[self.trader.timeframe].copy()
            if len(df) < 50:
                return result
            
            # Calculate volatility for thresholds
            volatility = self._calculate_adaptive_volatility()
            level_range = volatility * 3  # Range around level to analyze
            
            # Find candles that interacted with this price level
            level_candles = []
            for i in range(max(0, len(df) - 50), len(df)):
                # Check if candle is within range of price level
                if (price_level - level_range <= df['low'].iloc[i] <= price_level + level_range or
                    price_level - level_range <= df['high'].iloc[i] <= price_level + level_range):
                    level_candles.append(i)
            
            if len(level_candles) < 5:
                # Not enough price interaction with this level
                return result
                
            # === DETECT ACCUMULATION (Bullish bias) ===
            if direction == "buy":
                # 1. Check for absorption candles (high volume, small body, near the level)
                absorption_count = 0
                for i in level_candles:
                    if i >= 5 and i < len(df) - 1:  # Ensure we have context
                        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
                        body_range = abs(df['close'].iloc[i] - df['open'].iloc[i])
                        
                        # Check if it's an absorption candle (small body, high volume)
                        is_absorption = False
                        if 'volume' in df.columns and candle_range > 0:
                            relative_body = body_range / candle_range
                            avg_volume = df['volume'].iloc[i-5:i].mean()
                            
                            if (relative_body < 0.4 and 
                                df['volume'].iloc[i] > avg_volume * 1.3 and
                                df['low'].iloc[i] <= price_level <= df['high'].iloc[i]):
                                absorption_count += 1
                                is_absorption = True
                        
                        # Check if it's a bullish reversal pattern
                        if is_absorption and df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i+1] > df['close'].iloc[i]:
                            result["evidence"].append(f"BULL_ABSORPTION_AT_{i}")
                
                # 2. Check for multi-candle accumulation pattern (decreasing volatility, increasing lows)
                if len(level_candles) >= 8:
                    # Calculate average true range for recent candles
                    recent_indices = sorted(level_candles)[-8:]
                    
                    # Check if volatility is decreasing (sign of accumulation)
                    atr_values = []
                    for i in range(min(recent_indices), max(recent_indices) + 1):
                        if i > 0:
                            tr = max(
                                df['high'].iloc[i] - df['low'].iloc[i],
                                abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                                abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                            )
                            atr_values.append(tr)
                    
                    if len(atr_values) >= 5:
                        early_avg = sum(atr_values[:len(atr_values)//2]) / (len(atr_values)//2)
                        late_avg = sum(atr_values[len(atr_values)//2:]) / (len(atr_values) - len(atr_values)//2)
                        
                        # Volatility contraction is a sign of accumulation
                        if late_avg < early_avg * 0.8:
                            result["evidence"].append("VOLATILITY_CONTRACTION")
                            
                            # Check if lows are rising (further evidence)
                            recent_lows = [df['low'].iloc[i] for i in recent_indices]
                            if min(recent_lows[-3:]) > min(recent_lows[:3]):
                                result["evidence"].append("RISING_LOWS")
                
                # 3. Check for volume divergence (high volume on up candles, low on down)
                if 'volume' in df.columns and len(level_candles) >= 6:
                    up_candle_indices = [i for i in level_candles if df['close'].iloc[i] > df['open'].iloc[i]]
                    down_candle_indices = [i for i in level_candles if df['close'].iloc[i] < df['open'].iloc[i]]
                    
                    if up_candle_indices and down_candle_indices:
                        avg_up_volume = sum(df['volume'].iloc[i] for i in up_candle_indices) / len(up_candle_indices)
                        avg_down_volume = sum(df['volume'].iloc[i] for i in down_candle_indices) / len(down_candle_indices)
                        
                        # Higher volume on up candles suggests accumulation
                        if avg_up_volume > avg_down_volume * 1.3:
                            result["evidence"].append("BULLISH_VOLUME_DIVERGENCE")
                
                # Determine if accumulation is detected based on evidence
                if len(result["evidence"]) >= 2:
                    result["institutional_activity_detected"] = True
                    result["activity_type"] = "accumulation"
                    result["confidence_score"] = min(1.0, 0.5 + (len(result["evidence"]) * 0.15))
                    
                    # Determine the phase of accumulation
                    if "RISING_LOWS" in result["evidence"] and absorption_count >= 3:
                        result["phase"] = "late"  # Late accumulation, approaching markup
                    elif absorption_count >= 2:
                        result["phase"] = "mid"   # Mid phase accumulation
                    else:
                        result["phase"] = "early" # Early phase accumulation
                        
                    # Check for signs of completed accumulation (ready for markup)
                    if (result["phase"] == "late" and 
                        len(level_candles) >= 3 and
                        df['close'].iloc[level_candles[-1]] > df['close'].iloc[level_candles[-2]] > df['close'].iloc[level_candles[-3]]):
                        result["phase"] = "complete"  # Accumulation complete, likely to see markup
            
            # === DETECT DISTRIBUTION (Bearish bias) ===
            elif direction == "sell":
                # 1. Check for absorption candles (high volume, small body, near the level)
                absorption_count = 0
                for i in level_candles:
                    if i >= 5 and i < len(df) - 1:  # Ensure we have context
                        candle_range = df['high'].iloc[i] - df['low'].iloc[i]
                        body_range = abs(df['close'].iloc[i] - df['open'].iloc[i])
                        
                        # Check if it's an absorption candle (small body, high volume)
                        is_absorption = False
                        if 'volume' in df.columns and candle_range > 0:
                            relative_body = body_range / candle_range
                            avg_volume = df['volume'].iloc[i-5:i].mean()
                            
                            if (relative_body < 0.4 and 
                                df['volume'].iloc[i] > avg_volume * 1.3 and
                                df['low'].iloc[i] <= price_level <= df['high'].iloc[i]):
                                absorption_count += 1
                                is_absorption = True
                        
                        # Check if it's a bearish absorption pattern
                        if is_absorption and df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i+1] < df['close'].iloc[i]:
                            result["evidence"].append(f"BEAR_ABSORPTION_AT_{i}")
                
                # 2. Check for multi-candle distribution pattern (decreasing volatility, decreasing highs)
                if len(level_candles) >= 8:
                    # Calculate average true range for recent candles 
                    recent_indices = sorted(level_candles)[-8:]
                    
                    # Check if volatility is decreasing (sign of distribution)
                    atr_values = []
                    for i in range(min(recent_indices), max(recent_indices) + 1):
                        if i > 0:
                            tr = max(
                                df['high'].iloc[i] - df['low'].iloc[i],
                                abs(df['high'].iloc[i] - df['close'].iloc[i-1]),
                                abs(df['low'].iloc[i] - df['close'].iloc[i-1])
                            )
                            atr_values.append(tr)
                    
                    if len(atr_values) >= 5:
                        early_avg = sum(atr_values[:len(atr_values)//2]) / (len(atr_values)//2)
                        late_avg = sum(atr_values[len(atr_values)//2:]) / (len(atr_values) - len(atr_values)//2)
                        
                        # Volatility contraction is a sign of distribution
                        if late_avg < early_avg * 0.8:
                            result["evidence"].append("VOLATILITY_CONTRACTION")
                            
                            # Check if highs are falling (further evidence)
                            recent_highs = [df['high'].iloc[i] for i in recent_indices]
                            if max(recent_highs[-3:]) < max(recent_highs[:3]):
                                result["evidence"].append("FALLING_HIGHS")
                
                # 3. Check for volume divergence (high volume on down candles, low on up)
                if 'volume' in df.columns and len(level_candles) >= 6:
                    up_candle_indices = [i for i in level_candles if df['close'].iloc[i] > df['open'].iloc[i]]
                    down_candle_indices = [i for i in level_candles if df['close'].iloc[i] < df['open'].iloc[i]]
                    
                    if up_candle_indices and down_candle_indices:
                        avg_up_volume = sum(df['volume'].iloc[i] for i in up_candle_indices) / len(up_candle_indices)
                        avg_down_volume = sum(df['volume'].iloc[i] for i in down_candle_indices) / len(down_candle_indices)
                        
                        # Higher volume on down candles suggests distribution
                        if avg_down_volume > avg_up_volume * 1.3:
                            result["evidence"].append("BEARISH_VOLUME_DIVERGENCE")
                
                # Determine if distribution is detected based on evidence
                if len(result["evidence"]) >= 2:
                    result["institutional_activity_detected"] = True
                    result["activity_type"] = "distribution"
                    result["confidence_score"] = min(1.0, 0.5 + (len(result["evidence"]) * 0.15))
                    
                    # Determine the phase of distribution
                    if "FALLING_HIGHS" in result["evidence"] and absorption_count >= 3:
                        result["phase"] = "late"  # Late distribution, approaching markdown
                    elif absorption_count >= 2:
                        result["phase"] = "mid"   # Mid phase distribution
                    else:
                        result["phase"] = "early" # Early phase distribution
                        
                    # Check for signs of completed distribution (ready for markdown)
                    if (result["phase"] == "late" and 
                        len(level_candles) >= 3 and
                        df['close'].iloc[level_candles[-1]] < df['close'].iloc[level_candles[-2]] < df['close'].iloc[level_candles[-3]]):
                        result["phase"] = "complete"  # Distribution complete, likely to see markdown
            
            return result
        except Exception:
            return result
    
    def _detect_markup_markdown_transition(self, price_level: float, direction: str) -> Dict:
        """
        Detect transitions from accumulation to markup or distribution to markdown at Fibonacci levels.
        
        These transitions often indicate the start of significant directional moves initiated by
        institutional players after completing their position building near key Fibonacci levels.
        
        Args:
            price_level: The Fibonacci price level to analyze
            direction: 'buy' or 'sell' - expected direction after transition
            
        Returns:
            Dict with markup/markdown transition analysis results
        """
        result = {
            "transition_detected": False,
            "transition_type": None,  # "accumulation_to_markup" or "distribution_to_markdown"
            "confidence_score": 0.0,  # 0.0 to 1.0 scale
            "evidence": [],
            "strength_rating": 0  # 1-5 scale, with 5 being strongest
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return result
                
            df = self.trader.price_data[self.trader.timeframe].copy()
            if len(df) < 50:
                return result
            
            # Calculate volatility for thresholds
            volatility = self._calculate_adaptive_volatility()
            level_range = volatility * 3  # Range around level to analyze
            
            # First, check for prior accumulation or distribution
            if direction == "buy":
                inst_activity = self._detect_institutional_accumulation(price_level, direction)
                if not inst_activity["institutional_activity_detected"] or inst_activity["activity_type"] != "accumulation":
                    # No prior accumulation detected
                    return result
                    
                # Check if accumulation is complete or late phase
                if inst_activity["phase"] not in ["complete", "late"]:
                    # Accumulation not complete yet
                    return result
            else:  # sell direction
                inst_activity = self._detect_institutional_accumulation(price_level, direction)
                if not inst_activity["institutional_activity_detected"] or inst_activity["activity_type"] != "distribution":
                    # No prior distribution detected
                    return result
                    
                # Check if distribution is complete or late phase
                if inst_activity["phase"] not in ["complete", "late"]:
                    # Distribution not complete yet
                    return result
            
            # Find recent candles near this level
            level_candles = []
            for i in range(max(0, len(df) - 30), len(df)):
                # Check if candle is within range of price level
                if (price_level - level_range <= df['low'].iloc[i] <= price_level + level_range or
                    price_level - level_range <= df['high'].iloc[i] <= price_level + level_range):
                    level_candles.append(i)
            
            if len(level_candles) < 5:
                return result
                
            # === DETECT ACCUMULATION TO MARKUP TRANSITION (Bullish) ===
            if direction == "buy":
                # Sort candles by index for sequential analysis
                sorted_indices = sorted(level_candles)
                
                # Need at least 5 recent candles
                if len(sorted_indices) < 5:
                    return result
                    
                # Get most recent candles
                recent_indices = sorted_indices[-5:]
                
                # 1. Check for volume expansion on breakout
                if 'volume' in df.columns:
                    recent_volumes = [df['volume'].iloc[i] for i in recent_indices]
                    avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
                    base_volume = df['volume'].iloc[sorted_indices[-8:-5]].mean() if len(sorted_indices) >= 8 else 0
                    
                    # Volume expansion on breakout is a sign of markup beginning
                    if base_volume > 0 and avg_recent_volume > base_volume * 1.5:
                        result["evidence"].append("VOLUME_EXPANSION")
                
                # 2. Check for price action confirming markup
                recent_closes = [df['close'].iloc[i] for i in recent_indices]
                recent_opens = [df['open'].iloc[i] for i in recent_indices]
                
                # Count consecutive bullish candles
                bullish_count = sum(1 for i in range(len(recent_closes)) if recent_closes[i] > recent_opens[i])
                
                # Check for strong close above the level
                if recent_indices and df['close'].iloc[recent_indices[-1]] > price_level + (volatility * 0.5):
                    result["evidence"].append("CLOSE_ABOVE_LEVEL")
                
                # Check for multiple consecutive bullish candles
                if bullish_count >= 3:
                    result["evidence"].append("CONSECUTIVE_BULLISH")
                
                # 3. Check for increasing momentum (accelerating trend)
                if len(recent_indices) >= 5:
                    # Calculate rate of change
                    roc = (df['close'].iloc[recent_indices[-1]] - df['close'].iloc[recent_indices[0]]) / df['close'].iloc[recent_indices[0]]
                    
                    # Strong bullish momentum is a sign of markup
                    if roc > volatility * 2:
                        result["evidence"].append("INCREASING_MOMENTUM")
                        
                # 4. Check for expanding range candles (showing conviction)
                if len(recent_indices) >= 4:
                    early_ranges = [(df['high'].iloc[i] - df['low'].iloc[i]) for i in recent_indices[:2]]
                    late_ranges = [(df['high'].iloc[i] - df['low'].iloc[i]) for i in recent_indices[2:4]]
                    
                    avg_early_range = sum(early_ranges) / len(early_ranges)
                    avg_late_range = sum(late_ranges) / len(late_ranges)
                    
                    # Expanding range candles show increasing conviction
                    if avg_late_range > avg_early_range * 1.3:
                        result["evidence"].append("EXPANDING_RANGES")
                
                # 5. Check for gaps or strong opens (institutional buying)
                for i in range(1, len(recent_indices)):
                    prev_idx = recent_indices[i-1]
                    curr_idx = recent_indices[i]
                    
                    if (curr_idx == prev_idx + 1 and  # Consecutive candles
                        df['open'].iloc[curr_idx] > df['close'].iloc[prev_idx]):
                        # Gap up or strong open
                        result["evidence"].append(f"STRONG_OPEN_AT_{curr_idx}")
                        break
                
                # Determine if transition is detected based on evidence
                if len(result["evidence"]) >= 2:
                    result["transition_detected"] = True
                    result["transition_type"] = "accumulation_to_markup"
                    result["confidence_score"] = min(1.0, 0.5 + (len(result["evidence"]) * 0.1))
                    
                    # Calculate strength rating (1-5)
                    score = len(result["evidence"])
                    if "VOLUME_EXPANSION" in result["evidence"]:
                        score += 1
                    if "CONSECUTIVE_BULLISH" in result["evidence"] and "INCREASING_MOMENTUM" in result["evidence"]:
                        score += 1
                    
                    result["strength_rating"] = min(5, max(1, score))
            
            # === DETECT DISTRIBUTION TO MARKDOWN TRANSITION (Bearish) ===
            elif direction == "sell":
                # Sort candles by index for sequential analysis
                sorted_indices = sorted(level_candles)
                
                # Need at least 5 recent candles
                if len(sorted_indices) < 5:
                    return result
                    
                # Get most recent candles
                recent_indices = sorted_indices[-5:]
                
                # 1. Check for volume expansion on breakdown
                if 'volume' in df.columns:
                    recent_volumes = [df['volume'].iloc[i] for i in recent_indices]
                    avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
                    base_volume = df['volume'].iloc[sorted_indices[-8:-5]].mean() if len(sorted_indices) >= 8 else 0
                    
                    # Volume expansion on breakdown is a sign of markdown beginning
                    if base_volume > 0 and avg_recent_volume > base_volume * 1.5:
                        result["evidence"].append("VOLUME_EXPANSION")
                
                # 2. Check for price action confirming markdown
                recent_closes = [df['close'].iloc[i] for i in recent_indices]
                recent_opens = [df['open'].iloc[i] for i in recent_indices]
                
                # Count consecutive bearish candles
                bearish_count = sum(1 for i in range(len(recent_closes)) if recent_closes[i] < recent_opens[i])
                
                # Check for strong close below the level
                if recent_indices and df['close'].iloc[recent_indices[-1]] < price_level - (volatility * 0.5):
                    result["evidence"].append("CLOSE_BELOW_LEVEL")
                
                # Check for multiple consecutive bearish candles
                if bearish_count >= 3:
                    result["evidence"].append("CONSECUTIVE_BEARISH")
                
                # 3. Check for increasing downward momentum (accelerating trend)
                if len(recent_indices) >= 5:
                    # Calculate rate of change
                    roc = (df['close'].iloc[recent_indices[-1]] - df['close'].iloc[recent_indices[0]]) / df['close'].iloc[recent_indices[0]]
                    
                    # Strong bearish momentum is a sign of markdown
                    if roc < -volatility * 2:
                        result["evidence"].append("INCREASING_MOMENTUM")
                        
                # 4. Check for expanding range candles (showing conviction)
                if len(recent_indices) >= 4:
                    early_ranges = [(df['high'].iloc[i] - df['low'].iloc[i]) for i in recent_indices[:2]]
                    late_ranges = [(df['high'].iloc[i] - df['low'].iloc[i]) for i in recent_indices[2:4]]
                    
                    avg_early_range = sum(early_ranges) / len(early_ranges)
                    avg_late_range = sum(late_ranges) / len(late_ranges)
                    
                    # Expanding range candles show increasing conviction
                    if avg_late_range > avg_early_range * 1.3:
                        result["evidence"].append("EXPANDING_RANGES")
                
                # 5. Check for gaps or strong opens (institutional selling)
                for i in range(1, len(recent_indices)):
                    prev_idx = recent_indices[i-1]
                    curr_idx = recent_indices[i]
                    
                    if (curr_idx == prev_idx + 1 and  # Consecutive candles
                        df['open'].iloc[curr_idx] < df['close'].iloc[prev_idx]):
                        # Gap down or strong open
                        result["evidence"].append(f"STRONG_OPEN_AT_{curr_idx}")
                        break
                
                # Determine if transition is detected based on evidence
                if len(result["evidence"]) >= 2:
                    result["transition_detected"] = True
                    result["transition_type"] = "distribution_to_markdown"
                    result["confidence_score"] = min(1.0, 0.5 + (len(result["evidence"]) * 0.1))
                    
                    # Calculate strength rating (1-5)
                    score = len(result["evidence"])
                    if "VOLUME_EXPANSION" in result["evidence"]:
                        score += 1
                    if "CONSECUTIVE_BEARISH" in result["evidence"] and "INCREASING_MOMENTUM" in result["evidence"]:
                        score += 1
                    
                    result["strength_rating"] = min(5, max(1, score))
            
            return result
        except Exception:
            return result

    def _check_higher_timeframe_alignment(self) -> Dict:
        """
        Check if higher timeframes are aligned with the current timeframe's bias.
        This helps confirm institutional interest across multiple timeframes.
        
        Returns:
            Dict with multi-timeframe alignment analysis
        """
        result = {
            "aligned": False,
            "direction": None,  # "bullish" or "bearish"
            "alignment_strength": 0.0,  # 0.0 to 1.0
            "aligned_timeframes": []
        }
        
        try:
            # Get the current timeframe
            if not hasattr(self.trader, 'timeframe'):
                return result
                
            current_timeframe = self.trader.timeframe
            
            # Map of common timeframes and their higher timeframes
            timeframe_hierarchy = {
                "M1": ["M5", "M15", "H1"],
                "M5": ["M15", "H1", "H4"],
                "M15": ["H1", "H4", "D1"],
                "H1": ["H4", "D1", "W1"],
                "H4": ["D1", "W1", "MN1"],
                "D1": ["W1", "MN1"]
            }
            
            # If the current timeframe isn't in our map, return default
            if current_timeframe not in timeframe_hierarchy:
                return result
                
            higher_timeframes = timeframe_hierarchy[current_timeframe]
            aligned_count = 0
            total_checked = 0
            
            # Check direction on current timeframe
            if not hasattr(self.trader, 'price_data') or current_timeframe not in self.trader.price_data:
                return result
                
            current_df = self.trader.price_data[current_timeframe].copy()
            if len(current_df) < 10:
                return result
                
            # Determine trend direction on current timeframe
            current_direction = self._determine_timeframe_direction(current_df)
            if not current_direction:
                return result
                
            # Check alignment with higher timeframes
            for tf in higher_timeframes:
                if tf in self.trader.price_data and len(self.trader.price_data[tf]) >= 10:
                    higher_df = self.trader.price_data[tf].copy()
                    higher_direction = self._determine_timeframe_direction(higher_df)
                    
                    if higher_direction and higher_direction == current_direction:
                        aligned_count += 1
                        result["aligned_timeframes"].append(tf)
                    
                    total_checked += 1
            
            # Calculate alignment strength
            if total_checked > 0:
                alignment_strength = aligned_count / total_checked
                
                # Only consider aligned if majority of checked timeframes agree
                if alignment_strength >= 0.5:
                    result["aligned"] = True
                    result["direction"] = current_direction
                    result["alignment_strength"] = alignment_strength
            
            return result
        except Exception:
            return result
            
    def _determine_timeframe_direction(self, df) -> str:
        """
        Determine the trend direction for a given dataframe.
        
        Args:
            df: Price dataframe with OHLC data
            
        Returns:
            String "bullish", "bearish", or None if unclear
        """
        try:
            if len(df) < 10:
                return None
                
            # Calculate some basic indicators
            sma20 = df['close'].rolling(window=20).mean().iloc[-1]
            sma50 = df['close'].rolling(window=50).mean().iloc[-1]
            
            close = df['close'].iloc[-1]
            
            # Check for trend based on SMAs
            sma_trend = None
            if close > sma20 > sma50:
                sma_trend = "bullish"
            elif close < sma20 < sma50:
                sma_trend = "bearish"
            
            # Calculate higher highs and lower lows for last 20 candles
            recent = df.iloc[-20:]
            
            # For bullish trend: check if we have higher lows
            higher_lows = (recent['low'].iloc[-5:].min() > recent['low'].iloc[:-5].min())
            
            # For bearish trend: check if we have lower highs
            lower_highs = (recent['high'].iloc[-5:].max() < recent['high'].iloc[:-5].max())
            
            # Determine overall direction based on multiple factors
            if (sma_trend == "bullish" and higher_lows) or (close > sma20 > sma50 and higher_lows):
                return "bullish"
            elif (sma_trend == "bearish" and lower_highs) or (close < sma20 < sma50 and lower_highs):
                return "bearish"
            else:
                return None
        except Exception:
            return None
    
    def _identify_fibonacci_order_blocks(self, price_level: float, direction: str) -> Dict:
        """
        Identify order blocks that form specifically at Fibonacci levels.
        
        Order blocks at Fibonacci levels often represent institutional footprints and
        provide high-probability entry zones. These are particularly powerful when they
        form at key Fibonacci retracement or extension levels.
        
        Args:
            price_level: The Fibonacci price level to analyze
            direction: 'buy' or 'sell' - expected directional bias of order block
            
        Returns:
            Dict with Fibonacci order block analysis results
        """
        result = {
            "order_block_detected": False,
            "order_block_type": None,  # "bullish" or "bearish"
            "candle_index": None,      # Index of the order block candle
            "zone": {                  # Price zone of the order block
                "start": 0.0,
                "end": 0.0
            },
            "strength": 0.0,           # 0.0 to 1.0 scale
            "confidence_factors": []   # List of factors increasing confidence
        }
        
        try:
            if not hasattr(self.trader, 'price_data') or self.trader.timeframe not in self.trader.price_data:
                return result
                
            df = self.trader.price_data[self.trader.timeframe].copy()
            if len(df) < 30:
                return result
            
            # Calculate volatility for thresholds
            volatility = self._calculate_adaptive_volatility()
            proximity_threshold = volatility * 2  # How close candle must be to the Fib level
            
            # === IDENTIFY BULLISH ORDER BLOCKS (at support) ===
            if direction == "buy":
                # Find candles in proximity to the Fibonacci level (potential order blocks)
                ob_candidates = []
                for i in range(max(5, len(df) - 30), len(df) - 1):  # Skip most recent candle
                    # Check if candle low is near Fibonacci level
                    if abs(df['low'].iloc[i] - price_level) < proximity_threshold:
                        # Check if this was a down candle before moving up (classic order block)
                        if (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish candle
                            df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # Next candle bullish
                            df['low'].iloc[i+1:i+4].min() > df['low'].iloc[i]):  # No violation of the low
                            
                            ob_candidates.append({
                                "index": i,
                                "body_size": abs(df['close'].iloc[i] - df['open'].iloc[i]),
                                "proximity": abs(df['low'].iloc[i] - price_level),
                                "zone_start": min(df['open'].iloc[i], df['close'].iloc[i]),  # Bottom of body
                                "zone_end": max(df['open'].iloc[i], df['close'].iloc[i])    # Top of body
                            })
                
                # Find the best order block candidate
                best_ob = None
                best_score = 0
                
                for ob in ob_candidates:
                    # Calculate score based on proximity to Fibonacci level and body size
                    proximity_score = 1.0 - (ob["proximity"] / proximity_threshold)  # 0.0 to 1.0
                    body_ratio = ob["body_size"] / (df['high'].iloc[ob["index"]] - df['low'].iloc[ob["index"]])
                    
                    # Calculate volume factor
                    volume_factor = 1.0
                    if 'volume' in df.columns:
                        avg_volume = df['volume'].iloc[ob["index"]-5:ob["index"]].mean()
                        if avg_volume > 0:
                            volume_ratio = df['volume'].iloc[ob["index"]] / avg_volume
                            volume_factor = min(1.5, max(0.5, volume_ratio))
                    
                    # Calculate subsequent price action factor
                    subsequent_movement = 0
                    if ob["index"] + 5 < len(df):
                        max_high = df['high'].iloc[ob["index"]+1:ob["index"]+5].max()
                        subsequent_movement = (max_high - df['close'].iloc[ob["index"]]) / df['close'].iloc[ob["index"]]
                    
                    price_action_factor = min(1.5, max(1.0, subsequent_movement / volatility * 10))
                    
                    # Calculate final score
                    score = proximity_score * (1.0 - body_ratio) * volume_factor * price_action_factor
                    
                    if score > best_score:
                        best_score = score
                        best_ob = ob
                
                # If we found a good order block, populate the result
                if best_ob and best_score > 0.5:
                    result["order_block_detected"] = True
                    result["order_block_type"] = "bullish"
                    result["candle_index"] = best_ob["index"]
                    result["zone"]["start"] = best_ob["zone_start"]
                    result["zone"]["end"] = best_ob["zone_end"]
                    result["strength"] = best_score
                    
                    # Check additional confidence factors
                    
                    # 1. Check if the order block formed with high volume
                    if 'volume' in df.columns:
                        avg_volume = df['volume'].iloc[best_ob["index"]-5:best_ob["index"]].mean()
                        if df['volume'].iloc[best_ob["index"]] > avg_volume * 1.5:
                            result["confidence_factors"].append("HIGH_VOLUME")
                    
                    # 2. Check if the order block represents a swing low
                    if df['low'].iloc[best_ob["index"]] < df['low'].iloc[best_ob["index"]-3:best_ob["index"]].min():
                        result["confidence_factors"].append("SWING_LOW")
                    
                    # 3. Check if there was multi-timeframe alignment
                    mtf_analysis = self._check_higher_timeframe_alignment()
                    if mtf_analysis["aligned"] and mtf_analysis["direction"] == "bullish":
                        result["confidence_factors"].append("MTF_ALIGNED")
                    
                    # 4. Check if this is a retest of a previously respected zone
                    if hasattr(self.trader, 'zones'):
                        for zone in self.trader.zones:
                            if (zone.type == "demand" and 
                                zone.price_start <= best_ob["zone_start"] <= zone.price_end and
                                zone.touches > 1):
                                result["confidence_factors"].append("HISTORICAL_SUPPORT")
                                break
                    
                    # 5. Check for divergence at the order block
                    if 'volume' in df.columns and best_ob["index"] >= 5:
                        # Price making lower low but volume making lower high
                        prior_low_idx = None
                        for i in range(best_ob["index"]-5, best_ob["index"]):
                            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                                prior_low_idx = i
                                break
                        
                        if prior_low_idx and df['low'].iloc[best_ob["index"]] < df['low'].iloc[prior_low_idx]:
                            if df['volume'].iloc[best_ob["index"]] < df['volume'].iloc[prior_low_idx]:
                                result["confidence_factors"].append("BULLISH_DIVERGENCE")
            
            # === IDENTIFY BEARISH ORDER BLOCKS (at resistance) ===
            else:  # direction == "sell"
                # Find candles in proximity to the Fibonacci level (potential order blocks)
                ob_candidates = []
                for i in range(max(5, len(df) - 30), len(df) - 1):  # Skip most recent candle
                    # Check if candle high is near Fibonacci level
                    if abs(df['high'].iloc[i] - price_level) < proximity_threshold:
                        # Check if this was an up candle before moving down (classic order block)
                        if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                            df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # Next candle bearish
                            df['high'].iloc[i+1:i+4].max() < df['high'].iloc[i]):  # No violation of the high
                            
                            ob_candidates.append({
                                "index": i,
                                "body_size": abs(df['close'].iloc[i] - df['open'].iloc[i]),
                                "proximity": abs(df['high'].iloc[i] - price_level),
                                "zone_start": min(df['open'].iloc[i], df['close'].iloc[i]),  # Bottom of body
                                "zone_end": max(df['open'].iloc[i], df['close'].iloc[i])    # Top of body
                            })
                
                # Find the best order block candidate
                best_ob = None
                best_score = 0
                
                for ob in ob_candidates:
                    # Calculate score based on proximity to Fibonacci level and body size
                    proximity_score = 1.0 - (ob["proximity"] / proximity_threshold)  # 0.0 to 1.0
                    body_ratio = ob["body_size"] / (df['high'].iloc[ob["index"]] - df['low'].iloc[ob["index"]])
                    
                    # Calculate volume factor
                    volume_factor = 1.0
                    if 'volume' in df.columns:
                        avg_volume = df['volume'].iloc[ob["index"]-5:ob["index"]].mean()
                        if avg_volume > 0:
                            volume_ratio = df['volume'].iloc[ob["index"]] / avg_volume
                            volume_factor = min(1.5, max(0.5, volume_ratio))
                    
                    # Calculate subsequent price action factor
                    subsequent_movement = 0
                    if ob["index"] + 5 < len(df):
                        min_low = df['low'].iloc[ob["index"]+1:ob["index"]+5].min()
                        subsequent_movement = abs((min_low - df['close'].iloc[ob["index"]]) / df['close'].iloc[ob["index"]])
                    
                    price_action_factor = min(1.5, max(1.0, subsequent_movement / volatility * 10))
                    
                    # Calculate final score
                    score = proximity_score * (1.0 - body_ratio) * volume_factor * price_action_factor
                    
                    if score > best_score:
                        best_score = score
                        best_ob = ob
                
                # If we found a good order block, populate the result
                if best_ob and best_score > 0.5:
                    result["order_block_detected"] = True
                    result["order_block_type"] = "bearish"
                    result["candle_index"] = best_ob["index"]
                    result["zone"]["start"] = best_ob["zone_start"]
                    result["zone"]["end"] = best_ob["zone_end"]
                    result["strength"] = best_score
                    
                    # Check additional confidence factors
                    
                    # 1. Check if the order block formed with high volume
                    if 'volume' in df.columns:
                        avg_volume = df['volume'].iloc[best_ob["index"]-5:best_ob["index"]].mean()
                        if df['volume'].iloc[best_ob["index"]] > avg_volume * 1.5:
                            result["confidence_factors"].append("HIGH_VOLUME")
                    
                    # 2. Check if the order block represents a swing high
                    if df['high'].iloc[best_ob["index"]] > df['high'].iloc[best_ob["index"]-3:best_ob["index"]].max():
                        result["confidence_factors"].append("SWING_HIGH")
                    
                    # 3. Check if there was multi-timeframe alignment
                    mtf_analysis = self._check_higher_timeframe_alignment()
                    if mtf_analysis["aligned"] and mtf_analysis["direction"] == "bearish":
                        result["confidence_factors"].append("MTF_ALIGNED")
                    
                    # 4. Check if this is a retest of a previously respected zone
                    if hasattr(self.trader, 'zones'):
                        for zone in self.trader.zones:
                            if (zone.type == "supply" and 
                                zone.price_start <= best_ob["zone_end"] <= zone.price_end and
                                zone.touches > 1):
                                result["confidence_factors"].append("HISTORICAL_RESISTANCE")
                                break
                    
                    # 5. Check for divergence at the order block
                    if 'volume' in df.columns and best_ob["index"] >= 5:
                        # Price making higher high but volume making lower high
                        prior_high_idx = None
                        for i in range(best_ob["index"]-5, best_ob["index"]):
                            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                                prior_high_idx = i
                                break
                        
                        if prior_high_idx and df['high'].iloc[best_ob["index"]] > df['high'].iloc[prior_high_idx]:
                            if df['volume'].iloc[best_ob["index"]] < df['volume'].iloc[prior_high_idx]:
                                result["confidence_factors"].append("BEARISH_DIVERGENCE")
            
            return result
        except Exception:
            return result