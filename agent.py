import os
import streamlit as st
from dotenv import load_dotenv

# Load env variables first before other imports that might depend on them
load_dotenv()

# We can import crewai safely after env is loaded
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# Monkey patch crewai's cache_breakpoint bug which causes Groq API validation errors
import crewai.llms.cache
crewai.llms.cache.mark_cache_breakpoint = lambda message: message

# Ensure keys exist
# if not os.environ.get('GROQ_API_KEY'):
#     st.error("Please set your GROQ_API_KEY in a .env file.")
#     st.stop()

if not os.environ.get('SERPER_API_KEY'):
    st.error("Please set your SERPER_API_KEY in a .env file.")
    st.stop()

# Initialize LLM
temp = float(os.getenv('GROQ_TEMP', '0.0'))
llm = LLM(
    model="ollama/llama3.1:latest",
    base_url="http://localhost:11434",
    temperature=temp
)

# Initialize Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Define Agents
analyst = Agent(
    role="Market Analyst",
    goal="Gather and assess relevant data to determine the market's potential, competitive threats, and risks.",
    backstory="You are an expert market analyst with years of experience evaluating startup ideas. You excel at reading market data and summarizing key findings.",
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

strategist = Agent(
    role="Strategist",
    goal="Use historical data, trend analysis, and strategic thinking to offer actionable insights and forecasts.",
    backstory="You are a seasoned strategist known for turning raw data into actionable business plans and forecasting accurate market trends.",
    tools=[search_tool, scrape_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role="Report Writer",
    goal="Compose a comprehensive markdown report that integrates findings from the Analyst and Strategist.",
    backstory="You are a clear and concise technical writer skilled at summarizing complex analysis into easy-to-read reports. You always format your output nicely.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Streamlit UI
st.title("IdeaEval Agent")

query = st.text_input(
    "Enter your startup query",
    value="Analyze the market potential, competition, risks, provide actionable insights, and forecast future outcomes."
)

if st.button('Analyze'):
    with st.spinner('Analyzing (this may take a minute or two)...'):
        # Define Tasks dynamically based on user query
        analysis_task = Task(
            description=f"Analyze the market size, demand, competition, and risks for the following query: {query}",
            expected_output="A detailed summary of the market size, key competitors, and potential risks.",
            agent=analyst
        )
        
        strategy_task = Task(
            description="Based on the analysis, provide actionable recommendations and predict future outcomes.",
            expected_output="A list of actionable recommendations and a future forecast based on current trends.",
            agent=strategist
        )
        
        writing_task = Task(
            description="Compile the analysis and strategy into a final, well-structured comprehensive markdown report.",
            expected_output="A final markdown report combining all analysis and strategic insights.",
            agent=writer
        )
        
        # Define Crew
        startup_crew = Crew(
            agents=[analyst, strategist, writer],
            tasks=[analysis_task, strategy_task, writing_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            # Kickoff the multi-agent process
            result = startup_crew.kickoff()
            # result is a CrewOutput object, we can convert it to string safely
            report = str(result)
        except Exception as e:
            report = f"An error occurred: {e}"

    st.markdown("---")
    st.markdown(f"### Report for '{query}'")
    st.markdown(report)
