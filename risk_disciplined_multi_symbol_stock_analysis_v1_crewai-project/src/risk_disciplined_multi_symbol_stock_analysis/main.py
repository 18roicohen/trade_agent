#!/usr/bin/env python
import sys
import os
from datetime import datetime
from risk_disciplined_multi_symbol_stock_analysis.crew import SimplifiedStockAnalysisCrew

def run():
    """
    Run the simplified stock analysis crew with real examples.
    """
    
    # Example 1: Tech Portfolio
    tech_portfolio = {
        'stock_symbols': ['AAPL', 'GOOGL', 'MSFT', 'NVDA'],
        'analysis_timeframe': 30,  # 30 days for short-term analysis
        'portfolio_name': 'Tech Growth Portfolio',
        'risk_tolerance': 'medium'
    }
    
    # Example 2: Diversified Portfolio  
    diversified_portfolio = {
        'stock_symbols': ['AAPL', 'JNJ', 'JPM', 'PG', 'XOM'],
        'analysis_timeframe': 90,  # 90 days for longer-term view
        'portfolio_name': 'Diversified Blue Chip Portfolio', 
        'risk_tolerance': 'low'
    }
    
    # Example 3: High Growth Portfolio
    growth_portfolio = {
        'stock_symbols': ['TSLA', 'AMZN', 'META', 'NFLX'],
        'analysis_timeframe': 14,  # 14 days for short-term momentum
        'portfolio_name': 'High Growth Momentum Portfolio',
        'risk_tolerance': 'high'
    }
    
    # Select which portfolio to analyze (change this as needed)
    selected_portfolio = tech_portfolio  # Change to diversified_portfolio or growth_portfolio
    
    print(f"\n🚀 Starting analysis for: {selected_portfolio['portfolio_name']}")
    print(f"📊 Analyzing symbols: {', '.join(selected_portfolio['stock_symbols'])}")
    print(f"⏱️ Timeframe: {selected_portfolio['analysis_timeframe']} days")
    print(f"🎯 Risk Tolerance: {selected_portfolio['risk_tolerance']}")
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    try:
        # Run the crew analysis
        result = SimplifiedStockAnalysisCrew().crew().kickoff(inputs=selected_portfolio)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        portfolio_name_clean = selected_portfolio['portfolio_name'].replace(' ', '_').lower()
        filename = f"stock_analysis_{portfolio_name_clean}_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Stock Analysis Report\n")
            f.write(f"**Portfolio**: {selected_portfolio['portfolio_name']}\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Symbols**: {', '.join(selected_portfolio['stock_symbols'])}\n")
            f.write(f"**Timeframe**: {selected_portfolio['analysis_timeframe']} days\n\n")
            f.write("## Analysis Results\n\n")
            f.write(str(result))
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Results saved to: {filename}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        print(f"🔧 Troubleshooting tips:")
        print(f"   • Check internet connection for real-time data")
        print(f"   • Verify all stock symbols are valid")
        print(f"   • Ensure yfinance package is installed: pip install yfinance")
        return None

def run_custom():
    """
    Run analysis with custom user inputs.
    """
    print("\n📈 Custom Stock Analysis Setup")
    print("="*40)
    
    # Get user inputs
    symbols_input = input("Enter stock symbols (comma-separated, e.g., AAPL,GOOGL,MSFT): ").strip()
    if not symbols_input:
        print("❌ No symbols provided. Using default tech stocks.")
        symbols = ['AAPL', 'GOOGL', 'MSFT']
    else:
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
    
    # Get timeframe
    try:
        timeframe = int(input("Enter analysis timeframe in days (7-365, default 30): ") or "30")
        timeframe = max(7, min(365, timeframe))  # Clamp between 7 and 365
    except ValueError:
        timeframe = 30
        print("Invalid timeframe. Using 30 days.")
    
    # Get risk tolerance
    risk_input = input("Enter risk tolerance (low/medium/high, default medium): ").strip().lower()
    risk_tolerance = risk_input if risk_input in ['low', 'medium', 'high'] else 'medium'
    
    # Create custom inputs
    custom_inputs = {
        'stock_symbols': symbols,
        'analysis_timeframe': timeframe,
        'portfolio_name': 'Custom Portfolio',
        'risk_tolerance': risk_tolerance
    }
    
    print(f"\n🎯 Custom Analysis Configuration:")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe} days")
    print(f"   Risk Tolerance: {risk_tolerance}")
    
    # Run analysis
    try:
        result = SimplifiedStockAnalysisCrew().crew().kickoff(inputs=custom_inputs)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"custom_stock_analysis_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Custom Stock Analysis Report\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Symbols**: {', '.join(symbols)}\n")
            f.write(f"**Timeframe**: {timeframe} days\n")
            f.write(f"**Risk Tolerance**: {risk_tolerance}\n\n")
            f.write("## Analysis Results\n\n")
            f.write(str(result))
        
        print(f"\n✅ Custom analysis completed!")
        print(f"📄 Results saved to: {filename}")
        
    except Exception as e:
        print(f"\n❌ Custom analysis failed: {str(e)}")

def train():
    """
    Train the crew for better performance.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py train <iterations> <filename>")
        sys.exit(1)
        
    try:
        n_iterations = int(sys.argv[1])
        filename = sys.argv[2]
        
        inputs = {
            'stock_symbols': ['AAPL', 'GOOGL', 'MSFT'],  # Simple training set
            'analysis_timeframe': 30,
            'portfolio_name': 'Training Portfolio',
            'risk_tolerance': 'medium'
        }
        
        SimplifiedStockAnalysisCrew().crew().train(
            n_iterations=n_iterations, 
            filename=filename, 
            inputs=inputs
        )
        
        print(f"✅ Training completed: {n_iterations} iterations saved to {filename}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")

def replay():
    """
    Replay specific task execution.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py replay <task_id>")
        sys.exit(1)
        
    try:
        task_id = sys.argv[1]
        SimplifiedStockAnalysisCrew().crew().replay(task_id=task_id)
        print(f"✅ Replay completed for task: {task_id}")
        
    except Exception as e:
        print(f"❌ Replay failed: {e}")

def test():
    """
    Test the crew with validation.
    """
    if len(sys.argv) < 3:
        print("Usage: python main.py test <iterations> <model_name>")
        sys.exit(1)
        
    try:
        n_iterations = int(sys.argv[1])
        model_name = sys.argv[2]
        
        inputs = {
            'stock_symbols': ['AAPL', 'MSFT'],  # Small test set
            'analysis_timeframe': 14,
            'portfolio_name': 'Test Portfolio',
            'risk_tolerance': 'medium'
        }
        
        SimplifiedStockAnalysisCrew().crew().test(
            n_iterations=n_iterations,
            openai_model_name=model_name,
            inputs=inputs
        )
        
        print(f"✅ Testing completed: {n_iterations} iterations with {model_name}")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")

if __name__ == "__main__":
    print("🎯 Simplified Stock Analysis System")
    print("Powered by Real Market Data & Risk-Disciplined AI Agents")
    
    if len(sys.argv) < 2:
        print("\nAvailable commands:")
        print("  run         - Run with predefined portfolios")  
        print("  custom      - Run with custom user inputs")
        print("  train       - Train the crew")
        print("  replay      - Replay specific task")
        print("  test        - Test the crew")
        print("\nUsage: python main.py <command>")
        sys.exit(1)

    command = sys.argv[1]
    
    if command == "run":
        run()
    elif command == "custom":
        run_custom()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: run, custom, train, replay, test")
        sys.exit(1)
