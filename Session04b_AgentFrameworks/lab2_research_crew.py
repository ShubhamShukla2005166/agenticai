import os
import json
from ddgs.ddgs import DDGS
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from pendulum import now

load_dotenv()

MODEL = "gpt-4o-mini"
TOPIC = "India's DPDP Act oblications and compliance requirements for businesses"
BREAK_CONTEXT = False


def build_search_tools():

    @tool("Web Search")
    def ddf_search(query: str) -> str:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=5))
        if not hits:
            return "No results found."
        return "\n\n".join([f"{hit['title']}\n{hit['url']}" for hit in hits])

    return ddf_search


ddf_search = build_search_tools()

researcher = Agent(
    name="Research Analyst",
    goal="Find and cite source material on {TOPIC}",
    backstory="You verify every claim against a source before you report it. You are a research analyst who is an expert in the field of {TOPIC}. You are tasked with finding and citing source material on {TOPIC}. You will use the Web Search tool to find relevant information and provide citations for your findings.",
    tools=[ddf_search],
    llm = MODEL, max_iter=5,verbose=True,
)

summarizer = Agent(
    name="Briefing Writer",
    goal="Create a concise summary of the research findings on {TOPIC}",
    backstory="You are a summarizer who is skilled at identifying the most important points from research material. You will use the information gathered by the Research Analyst to create a clear and concise summary.",
    tools=[],
    llm = MODEL, max_iter=5,verbose=True,
)

critic = Agent(
    name="Quality Critic",
    goal="Review and validate the research findings on {TOPIC}",
    backstory="You are a critic who ensures the accuracy and reliability of the research material. You will review the findings from the Research Analyst and provide feedback to improve the quality of the information. You are hard to impress and will challenge the findings to ensure they are well-supported and credible.",
    tools=[],
    llm = MODEL, max_iter=5,verbose=True,
)

find = Task(
    description = "Research the topic: {topic}. Gather concrete findings and list a supporting source URL for each.",
    expected_output = "A list of findings, each with its supporting source URL.",
    agent           = researcher,
    # no context — this is the first task
)

write = Task(
    description     = "Write a 200-word brief on {topic} using only the researcher's findings. Do not introduce facts thatare not in the findings.",
    expected_output = "A brief of about 200 words with no invented facts.",
    agent           = summarizer,
    context         = [] if BREAK_CONTEXT else [find],
)

check = Task(
    description     = "Compare the brief against the findings. If every claim is supported, reply with exactly APPROVED. Otherwise reply with a numbered list of required revisions.",
    expected_output = "Either 'APPROVED' or a numbered revision list.",
    agent           = critic,
    context         = [find, write],   # <- seeing BOTH is what lets the
                                        #    critic catch an invented claim
)

def main():
    crew = Crew(
        agents=[researcher, summarizer, critic],
        tasks=[find, write, check],
        process=Process.sequential,
        verbose=True,
    )

    start = now()
    result = crew.kickoff(inputs={"topic": TOPIC})
    wall_clock_seconds = round(now() - start, 1)

    print("\n\n=== RESULTS ===")
    print(json.dumps(result, indent=2))

    # prompt_tokens, completion_tokens, total_tokens = read_tokens(crew)

main()