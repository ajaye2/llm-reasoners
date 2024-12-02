from pydantic import BaseModel, Field

from langchain_core.output_parsers import JsonOutputParser

class SelfEval(BaseModel):
    action_to_eval_and_context: str = Field(description="Context or Action to evaluate")
    justification: str = Field(description="Justification for eval")
    self_eval_value: str = Field(description="Eval (good or bad) or (Yes or No) depending on the action")
    self_eval_score: float = Field(description="Eval score between 1 and 10. Where 0 is extremely bad/No, 10 is extremely good/Yes, 5 is neutral")

self_eval_parser = JsonOutputParser(pydantic_object=SelfEval)