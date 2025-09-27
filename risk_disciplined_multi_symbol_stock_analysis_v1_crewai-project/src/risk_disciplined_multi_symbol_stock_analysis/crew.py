import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from risk_disciplined_multi_symbol_stock_analysis.tools.stock_data_tool import StockDataTool





@CrewBase
class RiskDisciplinedMultiSymbolStockAnalysisCrew:
    """RiskDisciplinedMultiSymbolStockAnalysis crew"""

    
    @agent
    def technical_analysis_specialist(self) -> Agent:

        
        return Agent(
            config=self.agents_config["technical_analysis_specialist"],
            
            
            tools=[
				StockDataTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gemini/gemini-2.0-flash-thinking-exp",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def fundamental_analysis_specialist(self) -> Agent:

        
        return Agent(
            config=self.agents_config["fundamental_analysis_specialist"],
            
            
            tools=[
				StockDataTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def market_sentiment_analyst(self) -> Agent:

        
        return Agent(
            config=self.agents_config["market_sentiment_analyst"],
            
            
            tools=[
				StockDataTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def risk_management_specialist(self) -> Agent:

        
        return Agent(
            config=self.agents_config["risk_management_specialist"],
            
            
            tools=[
				StockDataTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def master_stock_orchestrator(self) -> Agent:

        
        return Agent(
            config=self.agents_config["master_stock_orchestrator"],
            
            
            tools=[

            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def multi_symbol_coordinator(self) -> Agent:

        
        return Agent(
            config=self.agents_config["multi_symbol_coordinator"],
            
            
            tools=[
				StockDataTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def short_term_technical_analyst(self) -> Agent:

        
        return Agent(
            config=self.agents_config["short_term_technical_analyst"],
            
            
            tools=[
				StockDataTool()
            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def multi_symbol_report_generator(self) -> Agent:

        
        return Agent(
            config=self.agents_config["multi_symbol_report_generator"],
            
            
            tools=[

            ],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    

    
    @task
    def multi_symbol_data_processing(self) -> Task:
        return Task(
            config=self.tasks_config["multi_symbol_data_processing"],
            markdown=False,
            
            
        )
    
    @task
    def short_term_technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["short_term_technical_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["technical_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def fundamental_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["fundamental_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def sentiment_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["sentiment_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def risk_assessment(self) -> Task:
        return Task(
            config=self.tasks_config["risk_assessment"],
            markdown=False,
            
            
        )
    
    @task
    def master_stock_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["master_stock_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def multi_symbol_comparative_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["multi_symbol_comparative_analysis"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the RiskDisciplinedMultiSymbolStockAnalysis crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
