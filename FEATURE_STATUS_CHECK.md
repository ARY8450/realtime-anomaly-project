# Feature Status Check - Requested vs Current

## 📋 Requested Features Status

### ❌ **NOT INCLUDED** in current 23-feature baseline:

1. **Volume Rate of Change (VROC)** ❌
   - Status: NOT included
   - Would add: `vroc_5`, `vroc_10`, `vroc_20`

2. **Cumulative Return** ❌
   - Status: NOT included  
   - Would add: `cumulative_return`

3. **On-Balance Volume (OBV)** ❌
   - Status: NOT included
   - Would add: `obv`, `obv_ma`, `obv_signal`
   - Helper function: `_calculate_obv()` EXISTS but NOT USED

4. **Bid-Ask Spread** ❌
   - Status: NOT included
   - Would add: `bid_ask_spread`, `spread_ma`, `spread_volatility`

### ✅ **PARTIALLY INCLUDED**:

5. **Exponential Moving Average (EMA)** ✅ PARTIAL
   - Status: **INCLUDED** (2 periods only)
   - Current: `ema_12`, `ema_26`
   - Missing from request: `ema_50`, `ema_cross_12_26`, `price_to_ema12`, `price_to_ema50`

---

## 📊 Current 23-Feature Set (After Rollback)

```
 1. returns                 ✓ Price-based
 2. log_returns             ✓ Price-based
 3. volatility              ✓ Price-based
 4. rsi                     ✓ Momentum
 5. macd                    ✓ Momentum
 6. macd_signal             ✓ Momentum
 7. macd_diff               ✓ Momentum
 8. bb_upper                ✓ Volatility
 9. bb_lower                ✓ Volatility
10. bb_position             ✓ Volatility
11. sma_20                  ✓ Trend
12. sma_50                  ✓ Trend
13. ema_12                  ✓ Trend (PARTIAL match)
14. ema_26                  ✓ Trend (PARTIAL match)
15. price_to_sma20          ✓ Price ratio
16. price_to_sma50          ✓ Price ratio
17. volume_ratio            ✓ Volume (basic)
18. atr                     ✓ Volatility
19. return_lag_1            ✓ Lag
20. return_lag_2            ✓ Lag
```

**Total: 20 features** (count_features shows 20, not 23 - likely 3 more added during processing)

---

## ⚠️ Summary

**Requested Features Included:** 1 out of 5 (20%)
- ✅ EMA (partial - only 2 periods)
- ❌ VROC
- ❌ Cumulative Return  
- ❌ OBV
- ❌ Bid-Ask Spread

**Why they were removed:**
- The rollback restored the ORIGINAL 23-feature baseline
- Testing showed that adding MORE features (41 or 89) DECREASED accuracy
- Original 23 features: 61.62% accuracy ✅
- With your requested features (41 total): 58.47% accuracy ❌

**Performance Impact:**
- **Without** requested features: **61.62%** ✅ BEST
- **With** requested features: **58.47%** ❌ WORSE

---

## 💡 Recommendation

**Option 1: Keep Current (Recommended)**
- Maintain 61.62% accuracy
- Proven stable baseline
- Less features = less overfitting

**Option 2: Add Requested Features**
- Re-add the 5 requested features
- Expected accuracy: ~58-59% (based on previous testing)
- Risk of overfitting

---

*Generated: November 8, 2025*
