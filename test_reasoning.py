import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


response = client.responses.create(
    model="gpt-6.1-sol",

    input="""
You are advising NovaTech, a laptop company.

NovaTech:
- Revenue: $420M
- Growth: 11%
- Targets students and first-time buyers
- Low-cost pricing
- Strong student reach
- Weak premium perception
- Operates mainly in India and Southeast Asia

Apex Digital:
- Revenue: $1.2B
- Growth: 8%
- Targets professionals and creators
- Premium pricing
- Strong brand perception
- High margins
- Global presence

ByteWave:
- Revenue: $650M
- Growth: 14%
- Targets remote workers, freelancers and SMBs
- Mid-range pricing
- Strong value proposition
- Growing in India and Europe

EcoCircuit:
- Revenue: $360M
- Growth: 16%
- Targets environmentally conscious consumers and universities
- Mid-range pricing
- Strong sustainability positioning
- Smaller product portfolio

Determine which competitor poses the greatest strategic
threat to NovaTech over the next 3 years.

You must consider:

1. Customer overlap
2. Pricing overlap
3. Geographic overlap
4. Growth trajectory
5. Brand differentiation
6. Ability to move into NovaTech's student segment

Do not simply select the company with the highest revenue.

Rank all three competitors.

Explain any conflicting evidence.

Identify what additional data would be required before
making a high-confidence strategic decision.
""",

    reasoning={
        "effort": "high",
        "summary": "auto",
    },
)


print("\n========== OUTPUT ITEMS ==========\n")

for item in response.output:
    print(item)
    print()


print("\n========== REASONING SUMMARY ==========\n")

for item in response.output:

    if getattr(item, "type", None) == "reasoning":

        for summary in getattr(
            item,
            "summary",
            [],
        ):
            print(summary.text)


print("\n========== ANSWER ==========\n")

print(response.output_text)


print("\n========== USAGE ==========\n")

print(response.usage)