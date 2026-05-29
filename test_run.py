import asyncio
from eval_lib import run_eval
from eval_lib.types import EvalInput, EvalOutput
from eval_lib.scorers import AccuracyScorer, CitationScorer, CompletenessScorer, QuestionCompletenessScorer

# --- Regex scorer test ---

dataset = [
    EvalInput(
        query="What is the capital of France?",
        expected="Paris",
        expected_citations=["doc-1"]
    ),
    EvalInput(
        query="What is 2 + 2?",
        expected="4",
        expected_citations=["doc-2"]
    ),
    EvalInput(
        query="What color is the sky?",
        expected="blue",
        expected_citations=["doc-3"]
    ),
]

async def dummy_task(input: EvalInput) -> EvalOutput:
    responses = {
        "What is the capital of France?": ("Paris is the capital of France. [doc-1]", ["doc-1"]),
        "What is 2 + 2?": ("The answer is 5. [doc-3]", ["doc-3"]),
        "What color is the sky?": ("The sky is blue. [doc-3]", ["doc-3"]),
    }
    output, citations = responses[input.query]
    return EvalOutput(output=output, citations=citations)

# --- LLM-as-judge scorer test ---

llm_dataset = [
    EvalInput(
        query="What is the water cycle?",
        expected="The water cycle is the continuous movement of water through evaporation, condensation, and precipitation.",
        expected_citations=[]
    ),
    EvalInput(
        query="What causes rainbows?",
        expected="Rainbows are caused by light refracting and reflecting inside water droplets, splitting into its spectrum of colors.",
        expected_citations=[]
    ),
    EvalInput(
        query="How does photosynthesis work?",
        expected="Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
        expected_citations=[]
    ),
]

async def llm_dummy_task(input: EvalInput) -> EvalOutput:
    # Intentionally varied quality: full answer, partial answer, off-topic answer
    responses = {
        "What is the water cycle?": "The water cycle involves evaporation, condensation, and precipitation, continuously moving water through the environment.",
        "What causes rainbows?": "Rainbows appear after rain.",
        "How does photosynthesis work?": "Photosynthesis is a biological process.",
    }
    return EvalOutput(output=responses[input.query], citations=[])

async def main():
    await run_eval(
        name="Eval Library Testing",
        dataset=dataset,
        task=dummy_task,
        scorers=[AccuracyScorer(), CitationScorer()],
    )
    await run_eval(
        name="Eval Library Testing",
        dataset=llm_dataset,
        task=llm_dummy_task,
        scorers=[CompletenessScorer(), QuestionCompletenessScorer()],
    )

asyncio.run(main())
