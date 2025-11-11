# Reward Formulation for Mag7 Trading Environment

## Legacy Reward (Portfolio Value Change)

### Text Description

The reward at each non-terminal step is the **change in total portfolio value** between the previous step and the current step, optionally scaled by a reward_scaling factor.

**Components:**
- **Cash**: Available cash balance at time step t
- **Prices**: Current market prices for each of the 7 stocks at time step t
- **Holdings**: Number of shares owned for each of the 7 stocks at time step t

**Calculation Process:**
1. **Previous Portfolio Value** (begin_total_asset): Sum of cash and the value of all stock holdings at time t-1
2. **Current Portfolio Value** (end_total_asset): Sum of cash and the value of all stock holdings at time t
3. **Raw Reward**: Difference between current and previous portfolio values
4. **Final Reward**: Raw reward multiplied by a scaling factor (typically 1.0)

**Note:** Trading costs (if any) are implicitly included in the reward because buy/sell operations directly update the cash balance. The reward naturally reflects any transaction fees or slippage that reduce available cash.

### Mathematical Notation

Let:
- \( t \) = current time step
- \( \text{cash}_t \) = cash balance at time \( t \)
- \( p_{i,t} \) = price of stock \( i \) at time \( t \), where \( i \in \{1, 2, ..., 7\} \) (Mag7 stocks)
- \( n_{i,t} \) = number of shares of stock \( i \) held at time \( t \)
- \( N = 7 \) = total number of stocks (Mag7)

**Previous Portfolio Value:**
```math
\text{begin\_total\_asset} = \text{cash}_{t-1} + \sum_{i=1}^{N} p_{i,t-1} \cdot n_{i,t-1}
```

**Current Portfolio Value:**
```math
\text{end\_total\_asset} = \text{cash}_t + \sum_{i=1}^{N} p_{i,t} \cdot n_{i,t}
```

**Raw Reward:**
```math
r_t^{\text{raw}} = \text{end\_total\_asset} - \text{begin\_total\_asset}
```

**Final Reward (with scaling):**
```math
r_t = \alpha \cdot r_t^{\text{raw}}
```

where \( \alpha \) is the reward scaling factor (default: \( \alpha = 1.0 \))

**Expanded Form:**
```math
r_t = \alpha \cdot \left[ \left(\text{cash}_t + \sum_{i=1}^{N} p_{i,t} \cdot n_{i,t}\right) - \left(\text{cash}_{t-1} + \sum_{i=1}^{N} p_{i,t-1} \cdot n_{i,t-1}\right) \right]
```

### Implementation Details

**State Components (from `mag7_env.py`):**
- `self.cash` → cash_t
- `self.holdings[i]` → n_{i,t} for stock i
- `self.price_data.iloc[self.current_step]` → p_{i,t} for all stocks


