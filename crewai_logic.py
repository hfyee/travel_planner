#!/usr/bin/env python
'''
Travel Planner Assistant (CrewAI)
Ref: https://composio.dev/blog/crewai-examples

Encountered error with Serper API key, so using Tavily for search instead.
'''
# Import packages
import json
import os
import requests
from typing import Type
from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from crewai_tools import FileWriterTool
from langchain.tools import tool
from langchain_tavily import TavilySearch
from textwrap import dedent # for removing common whitespace
from datetime import date
from unstructured.partition.html import partition_html

# Define tools
class TavilySearchInput(BaseModel):
    """Input schema for TavilySearchTool"""
    query: str = Field(description="The search query to look up on the internet.")
    search_depth: str = Field(default="basic", description="The depth of the search results to return.")

class TavilySearchTool(BaseTool):
    name: str = "tavily_search"
    description: str = "A tool for searching the internet using Tavily to get real-time information."
    args_schema: Type[BaseModel] = TavilySearchInput
    search: TavilySearch = Field(default_factory=TavilySearch)

    def _run(
        self,
        query: str,
        search_depth: str = "basic",
        ) -> str:
        return self.search.run(query)

class SerperSearchTool(BaseTool):
    name: str = "Search the internet"
    description: str = """Useful to search the internet about a a given topic
    and return relevant results"""

    def _run(self, query: str):
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': os.environ['SERPER_API_KEY'],
            'content-type': 'application/json'
        }
        # Send the request to search
        response = requests.request("POST", url, headers=headers, data=payload)

        # check if there is an organic key, don't include sponsor results
        if 'organic' not in response.json():
          return "Sorry, I couldn't find anything about that, there could be an error with you serper api key."
        else:
            results = response.json()['organic']
            string = []
            for result in results[:top_result_to_return]:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}", f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}", "\n-----------------"
                    ]))
                except KeyError:
                    next

            return '\n'.join(string)
        
class BrowserTool(BaseTool):
    name: str = "Scrape website content"
    description: str = "Useful to scrape and summarize a website content"

    def _run(self, website: str):
        # url = f"https://chrome.browserless.io/content?token={os.environ['BROWSERLESS_API_KEY']}"
        url = "http://localhost:3000/content"
        payload = json.dumps({"url": website})
        headers = {'cache-control': 'no-cache', 'content-type': 'application/json'}
        response = requests.request("POST", url, headers=headers, data=payload)

        elements = partition_html(text=response.text)
        content = "\n\n".join([str(el) for el in elements])

        content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
        summaries = []
        for chunk in content:
            chunking_agent = Agent(
                role='Principal Researcher',
                goal='Do amazing researches and summaries based on the content you are working with',
                backstory="You're a Principal Researcher at a big company and you need to do a research about a given topic.",
                allow_delegation=False)
            chunking_task = Task(
                agent=chunking_agent,
                description=
                f"""Analyze and summarize the content bellow, make sure to include
                the most relevant information in the summary, return only the
                summary nothing else.\n\nCONTENT\n----------\n{chunk}"""
            )
            summary = chunking_task.execute() # Changed 'task.execute()' to 'chunking_task.execute()'
            summaries.append(summary)
        return "\n\n".join(summaries)
    
class CalculatorTool(BaseTool):
    name: str = "Make a calculation"
    description: str = """Useful to perform any mathematical calculations,
        like sum, minus, multiplication, division, etc.
        The input to this tool should be a mathematical
        expression, a couple examples are `200*7` or `5000/2*10`
        """

    def _run(self, operation: str):
        try:
            return eval(operation)
        except SyntaxError:
            return "Error: Invalid syntax in mathematical expression"

# Tool instances
#search_tool = SerperSearchTool()
search_tool  = TavilySearchTool()
#browser_tool = BrowserTool()
calculator_tool = CalculatorTool()
file_writer_tool = FileWriterTool(directory='output')

# Define the agents
city_selector = Agent(
    role='City Selection Expert',
    goal='Select the best city based on weather, season, and prices',
    backstory='An expert in analyzing travel data to pick ideal destinations',
    #tools=[search_tool, browser_tool],
    tools=[search_tool],
    max_iter=15,
    verbose=True)

local_expert = Agent(
    role='Local Expert at this city',
    goal='Provide the BEST insights about the selected city',
    backstory="""A knowledgeable local guide with extensive information
    about the city, it's attractions and customs""",
    #tools=[search_tool, browser_tool],
    tools=[search_tool],
    max_iter=15,
    verbose=True
)

travel_concierge = Agent(
    role='Amazing Travel Concierge',
    goal="""Create the most amazing travel itineraries with budget and
    packing suggestions for the city""",
    backstory="""Specialist in travel planning and logistics with
    decades of experience""",
    #tools=[search_tool, browser_tool, calculator_tool],
    tools=[search_tool, calculator_tool, file_writer_tool],
    max_iter=15,
    verbose=True
)

# Define the tasks
identify_task = Task(
    description=dedent("""
        Analyze and select the best city for the trip based
        on specific criteria such as weather patterns, seasonal
        events, and travel costs. This task involves comparing
        multiple cities, considering factors like current weather
        conditions, upcoming cultural or seasonal events, and
        overall travel expenses.

        If you do your BEST WORK, I'll tip you $100!

        Traveling from: {origin}
        City Options: {cities}
        Trip Date: {date_range}
        Traveler Interests: {interests}
        """),
    expected_output=dedent("""Your final answer must be a detailed
        report on the chosen city, and everything you found out
        about it, including the actual flight costs, weather
        forecast and attractions.
      """),
    agent=city_selector
)

gather_task = Task(
    description= dedent("""
        As a local expert on this city you must compile an
        in-depth guide for someone traveling there and wanting
        to have THE BEST trip ever!
        Gather information about  key attractions, local customs,
        special events, and daily activity recommendations.
        Find the best spots to go to, the kind of place only a
        local would know.
        This guide should provide a thorough overview of what
        the city has to offer, including hidden gems, cultural
        hotspots, must-visit landmarks, weather forecasts, and
        high level costs.

        If you do your BEST WORK, I'll tip you $100!

        Trip Date: {date_range}
        Traveling from: {origin}
        Traveler Interests: {interests}
        """),
    expected_output=dedent("""
        The final answer must be a comprehensive city guide,
        rich in cultural insights and practical tips,
        tailored to enhance the travel experience.
        """),
    agent=local_expert
)

plan_task = Task(
    description=dedent("""
        Expand this guide into a a full 7-day travel
        itinerary with detailed per-day plans, including
        weather forecasts, places to eat, packing suggestions,
        and a budget breakdown.

        You MUST suggest actual places to visit, actual hotels
        to stay and actual restaurants to go to.

        This itinerary should cover all aspects of the trip,
        from arrival to departure, integrating the city guide
        information with practical travel logistics.

        If you do your BEST WORK, I'll tip you $100!

        Trip Date: {date_range}
        Traveling from: {origin}
        Traveler Interests: {interests}
      """),
    context=[identify_task, gather_task],
    expected_output=dedent("""
        Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule, 
        anticipated weather conditions, recommended clothing and items to pack, 
        and a detailed budget, ensuring THE BEST TRIP EVER. 
        Be specific and give it a reason why you picked each place, what make them special!
        """),
    output_file="output/travel_plan.md",
    agent=travel_concierge
)

# Initiate the crew
trip_crew = Crew(
    agents=[city_selector, local_expert, travel_concierge],
    tasks=[identify_task, gather_task, plan_task],
    process=Process.sequential,
    verbose=True
)
