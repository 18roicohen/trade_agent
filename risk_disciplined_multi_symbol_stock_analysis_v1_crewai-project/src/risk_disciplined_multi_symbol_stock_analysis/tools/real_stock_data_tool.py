from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Dict, Any, List, Optional
import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealStockDataRequest(BaseModel):
    """Input schema for Real Stock Data Tool."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")

class RealStockDataTool(BaseTool):
    """Real stock data tool using yfinance - completely free and accurate."""

    name: str = "RealStockDataTool"
    description: str = (
        "Fetches real, accurate stock data using yfinance including current prices, "
        "historical data, fundamental metrics, technical indicators, and risk analysis. "
        "All data is live and sourced from Yahoo Finance."
    )
    args_schema: Type[BaseModel] = RealStockDataRequest

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Real RSI using actual price data."""
        if len(prices) < period + 1:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return round(float(rsi.iloc[-1]), 2) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate Real MACD using actual price data."""
        if len(prices) < 26:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        
        # Calculate EMAs
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        
        # MACD Line
        macd_line = ema_12 - ema_26
        
        # Signal Line (9-day EMA of MACD)
        signal_line = macd_line.ewm(span=9).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            "macd": round(float(macd_line.iloc[-1]), 4),
            "signal": round(float(signal_line.iloc[-1]), 4),
            "histogram": round(float(histogram.iloc[-1]), 4)
        }

    def _calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate Real Moving Averages."""
        averages = {}
        for period in [5, 10, 20, 50, 200]:
            if len(prices) >= period:
                ma = prices.rolling(window=period).mean().iloc[-1]
                averages[f"ma_{period}"] = round(float(ma), 2)
        return averages

    def _calculate_volatility(self, prices: pd.Series, period: int = 30) -> float:
        """Calculate actual price volatility."""
        if len(prices) < 2:
            return 0.0
        
        returns = prices.pct_change().dropna()
        if len(returns) < period:
            volatility = returns.std() * np.sqrt(252)  # Annualized
        else:
            volatility = returns.tail(period).std() * np.sqrt(252)
        
        return round(float(volatility) * 100, 2)  # As percentage

    def _calculate_beta(self, stock_prices: pd.Series, market_symbol: str = "^GSPC") -> float:
        """Calculate real Beta against S&P 500."""
        try:
            # Get S&P 500 data for same period
            market_data = yf.download(market_symbol, period="1y", progress=False)
            market_prices = market_data['Adj Close']
            
            # Align dates
            common_dates = stock_prices.index.intersection(market_prices.index)
            if len(common_dates) < 30:
                return 1.0
            
            stock_aligned = stock_prices.loc[common_dates]
            market_aligned = market_prices.loc[common_dates]
            
            # Calculate returns
            stock_returns = stock_aligned.pct_change().dropna()
            market_returns = market_aligned.pct_change().dropna()
            
            # Calculate beta
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
                
            beta = covariance / market_variance
            return round(float(beta), 2)
            
        except Exception:
            return 1.0

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from actual price data."""
        if len(prices) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown (most negative value)
        max_drawdown = drawdown.min()
        
        return round(float(abs(max_drawdown)) * 100, 2)

    def _get_fundamental_data(self, ticker_obj) -> Dict[str, Any]:
        """Extract real fundamental data from yfinance."""
        try:
            info = ticker_obj.info
            
            # Extract key fundamental metrics
            fundamentals = {
                "market_cap": info.get('marketCap', 'N/A'),
                "pe_ratio": info.get('trailingPE', info.get('forwardPE', 'N/A')),
                "eps": info.get('trailingEps', 'N/A'),
                "book_value": info.get('bookValue', 'N/A'),
                "revenue": info.get('totalRevenue', 'N/A'),
                "debt_to_equity": info.get('debtToEquity', 'N/A'),
                "roe": info.get('returnOnEquity', 'N/A'),
                "profit_margin": info.get('profitMargins', 'N/A'),
                "dividend_yield": info.get('dividendYield', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "company_name": info.get('longName', info.get('shortName', 'N/A'))
            }
            
            # Convert percentages
            for key in ['roe', 'profit_margin', 'dividend_yield']:
                if fundamentals[key] != 'N/A' and fundamentals[key] is not None:
                    fundamentals[key] = round(float(fundamentals[key]) * 100, 2)
            
            return fundamentals
            
        except Exception as e:
            return {"error": f"Failed to get fundamental data: {str(e)}"}

    def _analyze_sentiment_indicators(self, ticker_obj, price_change_pct: float) -> Dict[str, Any]:
        """Analyze real sentiment indicators."""
        try:
            info = ticker_obj.info
            
            # Analyst recommendations
            recommendations = ticker_obj.recommendations
            latest_recommendation = "N/A"
            if recommendations is not None and not recommendations.empty:
                latest_recommendation = recommendations.iloc[-1]['To Grade'] if 'To Grade' in recommendations.columns else "N/A"
            
            # Institutional ownership
            institutional_holders = ticker_obj.institutional_holders
            institutional_ownership = 0
            if institutional_holders is not None and not institutional_holders.empty:
                institutional_ownership = round(institutional_holders['% Out'].sum(), 1)
            
            # Sentiment score based on real factors
            sentiment_factors = []
            
            # Price momentum factor
            if price_change_pct > 5:
                sentiment_factors.append(0.8)
            elif price_change_pct > 2:
                sentiment_factors.append(0.4)
            elif price_change_pct > -2:
                sentiment_factors.append(0.0)
            elif price_change_pct > -5:
                sentiment_factors.append(-0.4)
            else:
                sentiment_factors.append(-0.8)
            
            # Analyst recommendation factor
            if "Buy" in str(latest_recommendation).upper():
                sentiment_factors.append(0.6)
            elif "Hold" in str(latest_recommendation).upper():
                sentiment_factors.append(0.0)
            elif "Sell" in str(latest_recommendation).upper():
                sentiment_factors.append(-0.6)
            
            # Calculate overall sentiment
            avg_sentiment = sum(sentiment_factors) / len(sentiment_factors) if sentiment_factors else 0
            
            sentiment_label = "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
            
            return {
                "sentiment_score": round(avg_sentiment, 2),
                "sentiment_label": sentiment_label,
                "latest_analyst_recommendation": latest_recommendation,
                "institutional_ownership_percent": institutional_ownership,
                "confidence": round(min(0.8, abs(avg_sentiment) + 0.4), 2)
            }
            
        except Exception as e:
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral", 
                "latest_analyst_recommendation": "N/A",
                "institutional_ownership_percent": 0.0,
                "confidence": 0.5,
                "error": f"Sentiment analysis failed: {str(e)}"
            }

    def _run(self, symbol: str) -> str:
        """Main execution method - fetch real stock data."""
        try:
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            if not info or info.get('regularMarketPrice') is None:
                return json.dumps({
                    "error": f"No data available for symbol {symbol}. Please verify the symbol is correct.",
                    "symbol": symbol.upper()
                })
            
            # Get historical data (1 year for comprehensive analysis)
            hist_data = ticker.history(period="1y")
            if hist_data.empty:
                return json.dumps({
                    "error": f"No historical data available for {symbol}",
                    "symbol": symbol.upper()
                })
            
            # Current price data
            current_price = float(info.get('regularMarketPrice', hist_data['Close'].iloc[-1]))
            previous_close = float(info.get('regularMarketPreviousClose', hist_data['Close'].iloc[-2] if len(hist_data) > 1 else current_price))
            
            # Price changes
            price_change = current_price - previous_close
            price_change_percent = (price_change / previous_close * 100) if previous_close > 0 else 0
            
            # Current trading data
            current_volume = int(info.get('regularMarketVolume', hist_data['Volume'].iloc[-1]))
            avg_volume = int(hist_data['Volume'].tail(30).mean())
            
            # Price levels
            day_high = float(info.get('regularMarketDayHigh', hist_data['High'].iloc[-1]))
            day_low = float(info.get('regularMarketDayLow', hist_data['Low'].iloc[-1]))
            week_52_high = float(hist_data['High'].max())
            week_52_low = float(hist_data['Low'].min())
            
            # Technical indicators using real data
            close_prices = hist_data['Close']
            rsi = self._calculate_rsi(close_prices)
            macd_data = self._calculate_macd(close_prices)
            moving_averages = self._calculate_moving_averages(close_prices)
            
            # Risk metrics
            volatility_30d = self._calculate_volatility(close_prices, 30)
            volatility_90d = self._calculate_volatility(close_prices, 90)
            beta = self._calculate_beta(close_prices)
            max_drawdown = self._calculate_max_drawdown(close_prices)
            
            # Fundamental data
            fundamental_data = self._get_fundamental_data(ticker)
            
            # Sentiment analysis
            sentiment_data = self._analyze_sentiment_indicators(ticker, price_change_percent)
            
            # Compile comprehensive stock data
            stock_data = {
                "symbol": symbol.upper(),
                "company_name": fundamental_data.get("company_name", "N/A"),
                "sector": fundamental_data.get("sector", "N/A"),
                "industry": fundamental_data.get("industry", "N/A"),
                "data_timestamp": datetime.now().isoformat(),
                
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_percent, 2),
                "previous_close": round(previous_close, 2),
                
                "trading_data": {
                    "current_volume": current_volume,
                    "average_volume_30d": avg_volume,
                    "volume_ratio": round(current_volume / avg_volume if avg_volume > 0 else 1, 2),
                    "day_high": round(day_high, 2),
                    "day_low": round(day_low, 2),
                    "week_52_high": round(week_52_high, 2),
                    "week_52_low": round(week_52_low, 2)
                },
                
                "technical_indicators": {
                    "rsi_14": rsi,
                    "macd": macd_data,
                    "moving_averages": moving_averages
                },
                
                "fundamental_metrics": {
                    key: value for key, value in fundamental_data.items() 
                    if key not in ["company_name", "sector", "industry"]
                },
                
                "risk_metrics": {
                    "volatility_30d_percent": volatility_30d,
                    "volatility_90d_percent": volatility_90d,
                    "beta": beta,
                    "max_drawdown_1y_percent": max_drawdown
                },
                
                "sentiment_analysis": sentiment_data,
                
                "data_quality": {
                    "historical_data_points": len(hist_data),
                    "data_completeness": "excellent" if len(hist_data) > 200 else "good" if len(hist_data) > 50 else "limited",
                    "fundamental_data_available": "error" not in fundamental_data,
                    "real_time_data": True
                }
            }
            
            return json.dumps(stock_data, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"Failed to fetch data for {symbol}: {str(e)}",
                "symbol": symbol.upper(),
                "suggestion": "Please verify the symbol exists and has available market data. For international stocks, try adding the appropriate exchange suffix (e.g., '.TO' for Toronto, '.L' for London)."
            })
