from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class BacktestRequest(BaseModel):
    """Input schema for Backtesting Tool."""
    symbols: List[str] = Field(..., description="List of stock symbols to backtest")
    allocations: List[float] = Field(..., description="Portfolio allocation percentages (should sum to 100)")
    period: str = Field(default="1y", description="Backtesting period (1y, 2y, 6mo, etc.)")

class BacktestingTool(BaseTool):
    """Simple but effective backtesting tool for portfolio strategies."""

    name: str = "BacktestingTool"
    description: str = (
        "Performs backtesting analysis on portfolio allocations using historical data. "
        "Calculates returns, risk metrics, and compares against benchmark indices."
    )
    args_schema: Type[BaseModel] = BacktestRequest

    def _get_benchmark_data(self, period: str) -> pd.Series:
        """Get S&P 500 benchmark data."""
        try:
            spy_data = yf.download("SPY", period=period, progress=False)
            return spy_data['Adj Close']
        except:
            return pd.Series()

    def _calculate_portfolio_performance(self, 
                                       symbols: List[str], 
                                       allocations: List[float], 
                                       period: str) -> Dict[str, Any]:
        """Calculate historical portfolio performance."""
        try:
            # Normalize allocations to percentages
            total_allocation = sum(allocations)
            normalized_allocations = [alloc / total_allocation for alloc in allocations]
            
            # Download historical data for all symbols
            portfolio_data = {}
            for symbol in symbols:
                try:
                    data = yf.download(symbol, period=period, progress=False)
                    if not data.empty:
                        portfolio_data[symbol] = data['Adj Close']
                except Exception as e:
                    print(f"Failed to get data for {symbol}: {str(e)}")
                    continue
            
            if not portfolio_data:
                return {"error": "No valid data for any symbols"}
            
            # Create aligned DataFrame
            df = pd.DataFrame(portfolio_data)
            df = df.dropna()  # Remove days with missing data
            
            if df.empty:
                return {"error": "No overlapping data between symbols"}
            
            # Calculate daily returns
            returns = df.pct_change().dropna()
            
            # Calculate portfolio returns using allocations
            portfolio_returns = pd.Series(0, index=returns.index)
            valid_symbols = []
            valid_allocations = []
            
            for i, symbol in enumerate(symbols):
                if symbol in returns.columns:
                    portfolio_returns += returns[symbol] * normalized_allocations[i]
                    valid_symbols.append(symbol)
                    valid_allocations.append(normalized_allocations[i])
            
            if portfolio_returns.empty:
                return {"error": "Could not calculate portfolio returns"}
            
            # Calculate cumulative returns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            total_return = (cumulative_returns.iloc[-1] - 1) * 100
            
            # Calculate annualized return
            days = len(portfolio_returns)
            years = days / 252  # Trading days per year
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
            
            # Calculate risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Maximum drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 2.0
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Win rate
            winning_days = (portfolio_returns > 0).sum()
            total_days = len(portfolio_returns)
            win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
            
            # Calculate individual stock performance
            individual_performance = {}
            for symbol in valid_symbols:
                if symbol in returns.columns:
                    symbol_cumret = (1 + returns[symbol]).cumprod()
                    symbol_total_return = (symbol_cumret.iloc[-1] - 1) * 100
                    symbol_volatility = returns[symbol].std() * np.sqrt(252) * 100
                    
                    individual_performance[symbol] = {
                        "total_return_percent": round(symbol_total_return, 2),
                        "volatility_percent": round(symbol_volatility, 2),
                        "allocation_percent": round(valid_allocations[valid_symbols.index(symbol)] * 100, 1)
                    }
            
            return {
                "portfolio_performance": {
                    "total_return_percent": round(total_return, 2),
                    "annualized_return_percent": round(annualized_return, 2),
                    "volatility_percent": round(volatility, 2),
                    "max_drawdown_percent": round(abs(max_drawdown), 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "win_rate_percent": round(win_rate, 1)
                },
                "individual_stocks": individual_performance,
                "period_info": {
                    "start_date": df.index[0].strftime("%Y-%m-%d"),
                    "end_date": df.index[-1].strftime("%Y-%m-%d"),
                    "trading_days": len(portfolio_returns),
                    "years_analyzed": round(years, 2)
                }
            }
            
        except Exception as e:
            return {"error": f"Portfolio performance calculation failed: {str(e)}"}

    def _compare_to_benchmark(self, 
                            portfolio_returns: pd.Series, 
                            period: str) -> Dict[str, Any]:
        """Compare portfolio performance to S&P 500 benchmark."""
        try:
            # Get benchmark data
            benchmark_data = self._get_benchmark_data(period)
            
            if benchmark_data.empty:
                return {"error": "Could not fetch benchmark data"}
            
            # Align dates with portfolio
            common_dates = portfolio_returns.index.intersection(benchmark_data.index)
            if len(common_dates) < 10:  # Need minimum data points
                return {"error": "Insufficient overlapping data with benchmark"}
            
            # Calculate benchmark returns
            aligned_benchmark = benchmark_data.loc[common_dates]
            aligned_portfolio = portfolio_returns.loc[common_dates]
            
            benchmark_returns = aligned_benchmark.pct_change().dropna()
            
            # Align portfolio returns with benchmark
            final_common_dates = aligned_portfolio.index.intersection(benchmark_returns.index)
            benchmark_returns = benchmark_returns.loc[final_common_dates]
            portfolio_returns_aligned = aligned_portfolio.loc[final_common_dates]
            
            # Calculate benchmark performance
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_total_return = (benchmark_cumulative.iloc[-1] - 1) * 100
            
            # Calculate portfolio performance for comparison period
            portfolio_cumulative = (1 + portfolio_returns_aligned).cumprod()
            portfolio_total_return = (portfolio_cumulative.iloc[-1] - 1) * 100
            
            # Calculate outperformance
            alpha = portfolio_total_return - benchmark_total_return
            
            # Calculate beta
            covariance = np.cov(portfolio_returns_aligned, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0
            
            return {
                "benchmark_comparison": {
                    "portfolio_return_percent": round(portfolio_total_return, 2),
                    "benchmark_return_percent": round(benchmark_total_return, 2),
                    "alpha_percent": round(alpha, 2),
                    "beta": round(beta, 2),
                    "outperformance": "Yes" if alpha > 0 else "No"
                }
            }
            
        except Exception as e:
            return {"error": f"Benchmark comparison failed: {str(e)}"}

    def _generate_backtest_summary(self, 
                                 performance_data: Dict, 
                                 benchmark_data: Dict,
                                 symbols: List[str],
                                 allocations: List[float]) -> str:
        """Generate human-readable backtest summary."""
        
        if "error" in performance_data:
            return f"Backtest Error: {performance_data['error']}"
        
        portfolio_perf = performance_data["portfolio_performance"]
        period_info = performance_data["period_info"]
        
        summary = f"""
BACKTESTING RESULTS SUMMARY

Portfolio Configuration:
"""
        
        # Add portfolio allocation
        for i, symbol in enumerate(symbols):
            if i < len(allocations):
                summary += f"• {symbol}: {allocations[i]:.1f}%\n"
        
        summary += f"""
Analysis Period: {period_info['start_date']} to {period_info['end_date']} ({period_info['trading_days']} trading days)

PORTFOLIO PERFORMANCE:
• Total Return: {portfolio_perf['total_return_percent']:.2f}%
• Annualized Return: {portfolio_perf['annualized_return_percent']:.2f}%
• Volatility: {portfolio_perf['volatility_percent']:.2f}%
• Maximum Drawdown: {portfolio_perf['max_drawdown_percent']:.2f}%
• Sharpe Ratio: {portfolio_perf['sharpe_ratio']:.2f}
• Win Rate: {portfolio_perf['win_rate_percent']:.1f}%

INDIVIDUAL STOCK PERFORMANCE:
"""
        
        for symbol, data in performance_data["individual_stocks"].items():
            summary += f"• {symbol} ({data['allocation_percent']}%): {data['total_return_percent']:.2f}% return, {data['volatility_percent']:.1f}% volatility\n"
        
        # Add benchmark comparison if available
        if "benchmark_comparison" in benchmark_data:
            bench = benchmark_data["benchmark_comparison"]
            summary += f"""
BENCHMARK COMPARISON (vs S&P 500):
• Portfolio Return: {bench['portfolio_return_percent']:.2f}%
• S&P 500 Return: {bench['benchmark_return_percent']:.2f}%
• Alpha (Outperformance): {bench['alpha_percent']:.2f}%
• Beta (Market Sensitivity): {bench['beta']:.2f}
• Outperformed Market: {bench['outperformance']}
"""
        
        # Add performance assessment
        if portfolio_perf['total_return_percent'] > 10:
            summary += "\n✅ STRONG PERFORMANCE: Portfolio showed solid returns"
        elif portfolio_perf['total_return_percent'] > 5:
            summary += "\n🔶 MODERATE PERFORMANCE: Decent returns with manageable risk"
        else:
            summary += "\n🔻 UNDERPERFORMANCE: Consider strategy adjustments"
        
        if portfolio_perf['sharpe_ratio'] > 1:
            summary += "\n📈 EXCELLENT RISK-ADJUSTED RETURNS (Sharpe > 1.0)"
        elif portfolio_perf['sharpe_ratio'] > 0.5:
            summary += "\n📊 GOOD RISK-ADJUSTED RETURNS (Sharpe > 0.5)"
        else:
            summary += "\n⚠️ POOR RISK-ADJUSTED RETURNS - High risk for returns achieved"
        
        return summary

    def _run(self, symbols: List[str], allocations: List[float], period: str = "1y") -> str:
        """Execute backtesting analysis."""
        try:
            # Validate inputs
            if len(symbols) != len(allocations):
                return json.dumps({
                    "error": "Number of symbols must match number of allocations"
                })
            
            if abs(sum(allocations) - 100) > 1:  # Allow 1% tolerance
                return json.dumps({
                    "error": f"Allocations must sum to 100%, got {sum(allocations)}%"
                })
            
            # Calculate portfolio performance
            performance_data = self._calculate_portfolio_performance(symbols, allocations, period)
            
            if "error" in performance_data:
                return json.dumps(performance_data)
            
            # Get portfolio returns for benchmark comparison
            portfolio_data = {}
            for symbol in symbols:
                try:
                    data = yf.download(symbol, period=period, progress=False)
                    if not data.empty:
                        portfolio_data[symbol] = data['Adj Close']
                except:
                    continue
            
            if portfolio_data:
                df = pd.DataFrame(portfolio_data).dropna()
                returns = df.pct_change().dropna()
                
                # Calculate weighted portfolio returns
                normalized_allocations = [alloc/100 for alloc in allocations]  # Convert to decimals
                portfolio_returns = pd.Series(0, index=returns.index)
                
                for i, symbol in enumerate(symbols):
                    if symbol in returns.columns:
                        portfolio_returns += returns[symbol] * normalized_allocations[i]
                
                # Compare to benchmark
                benchmark_data = self._compare_to_benchmark(portfolio_returns, period)
            else:
                benchmark_data = {"error": "Could not calculate benchmark comparison"}
            
            # Generate summary
            summary = self._generate_backtest_summary(performance_data, benchmark_data, symbols, allocations)
            
            # Combine all data
            complete_results = {
                "backtest_summary": summary,
                "detailed_performance": performance_data,
                "benchmark_analysis": benchmark_data,
                "input_parameters": {
                    "symbols": symbols,
                    "allocations": allocations,
                    "period": period,
                    "analysis_date": datetime.now().isoformat()
                }
            }
            
            return json.dumps(complete_results, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Backtesting analysis failed: {str(e)}",
                "symbols": symbols,
                "suggestion": "Verify all symbols are valid and have sufficient historical data"
            })
