import os
from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Import real tools instead of fake ones
from risk_disciplined_multi_symbol_stock_analysis.tools.real_stock_data_tool import RealStockDataTool
from risk_disciplined_multi_symbol_stock_analysis.tools.backtesting_tool import BacktestingTool

@CrewBase
class SimplifiedStockAnalysisCrew:
    """Simplified, efficient 4-agent stock analysis crew"""

    @agent
    def market_data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["market_data_analyst"],
            tools=[RealStockDataTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=15,  # Reduced iterations for efficiency
            llm=LLM(
                model="gpt-4o-mini",  # Consistent model across all agents
                temperature=0.3,  # Lower temperature for data analysis
            ),
        )
    
    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["technical_analyst"],
            tools=[RealStockDataTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=20,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.4,  # Slightly higher for pattern recognition
            ),
        )
    
    @agent
    def fundamental_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamental_analyst"],
            tools=[RealStockDataTool()],
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=20,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.3,  # Low temperature for numerical analysis
            ),
        )
    
    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["portfolio_manager"],
            tools=[BacktestingTool()],  # Portfolio manager gets backtesting tool
            reasoning=False,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.2,  # Very low temperature for risk management
            ),
        )

    @task
    def data_collection_and_validation(self) -> Task:
        return Task(
            config=self.tasks_config["data_collection_and_validation"],
            markdown=False,
        )
    
    @task
    def comprehensive_technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["comprehensive_technical_analysis"],
            markdown=False,
        )
    
    @task
    def integrated_fundamental_and_risk_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["integrated_fundamental_and_risk_analysis"],
            markdown=False,
        )
    
    @task
    def portfolio_optimization_and_allocation(self) -> Task:
        return Task(
            config=self.tasks_config["portfolio_optimization_and_allocation"],
            markdown=False,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the simplified, efficient stock analysis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # Optional: Add memory for better context retention
            memory=True,
            # Optional: Add planning for better task coordination  
            planning=True,
        )
