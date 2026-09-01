import os
import json
from ddgs.ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from pendulum import now

load_dotenv()

MODEL = "gpt-4o-mini"
TOPIC = "India's DPDP Act obligations for AI systems handling customer data"
BREAK_CONTEXT = False


def build_search_tools():

    @tool("Web Search")
    def ddf_search(query: str) -> str:
        """Search the web for a topic and return the top results with titles and URLs."""
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=5))
        except DDGSException:
            return "No results found."

        if not hits:
            return "No results found."

        cleaned = []
        for hit in hits:
            title = hit.get("title") or "Untitled result"
            url = hit.get("url") or hit.get("href") or hit.get("link")
            if not url:
                continue
            cleaned.append(f"{title}\n{url}")

        return "\n\n".join(cleaned) if cleaned else "No usable results found."

    return ddf_search


ddf_search = build_search_tools()

researcher = Agent(
    role="Research Analyst",
    name="Research Analyst",
    goal="Find and cite source material on {TOPIC}",
    backstory="""write 1-2 sentences establishing: verifies every claim
                 against a source before reporting it; never presents an
                 unsourced statement as fact""",
    tools=[ddf_search],
    llm = MODEL, max_iter=5,verbose=True,
)

summarizer = Agent(
    role="Briefing Writer",
    name="Briefing Writer",
    goal="Turn the researcher's findings into a 200-word brief on {TOPIC}",
    backstory="""compresses findings into clear prose; never adds a fact
                 that isn't in the findings it was handed""",
    tools=[],
    llm = MODEL, max_iter=5,verbose=True,
)

critic = Agent(
    role="Quality Critic",
    name="Quality Critic",
    goal="""Check the brief's tone is professional" IF WEAK_CRITIC
              ELSE "Flag any claim in the brief that is not supported by the findings""",
    backstory="""hard to satisfy; treats a claim as unsupported until it has
                 seen it in the findings""",
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
    result = crew.kickoff(inputs={"topic": TOPIC, "TOPIC": TOPIC})
    wall_clock_seconds = round((now() - start).total_seconds(), 1)

    print("\n\n=== RESULTS ===")
    print(result)
    print(f"\nWall clock time: {wall_clock_seconds} seconds")

    # prompt_tokens, completion_tokens, total_tokens = read_tokens(crew)


if __name__ == "__main__":
    main()