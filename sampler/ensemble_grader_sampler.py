import json
from ..types_eval import MessageList, SamplerBase, SamplerResponse
import re


def _clean_json(json_string: str) -> dict:
    # Strip markdown fences if present
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except Exception:
        return {"criteria_met": False, "explanation": "Parse error"}


class EnsembleGraderSampler(SamplerBase):
    """
    Ensemble grader that combines multiple graders using majority vote.
    """

    def __init__(self, graders: list[SamplerBase]):
        super().__init__()
        self.graders = graders

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        responses = []
        for grader in self.graders:
            resp = grader(message_list)
            try:
                parsed = _clean_json(resp.response_text)
            except Exception:
                parsed = {"criteria_met": False, "explanation": "Parse error"}
            responses.append(parsed)

        # Majority vote
        votes = [r.get("criteria_met", False) for r in responses]
        majority = sum(votes) > len(votes) / 2

        # Gather explanations
        explanations = [r.get("explanation", "") for r in responses]
        explanation = " | ".join(explanations)

        result_json = {
            "criteria_met": majority,
            "explanation": explanation,
            "ensemble_votes": votes,
            "ensemble_raw_responses": responses,
        }

        return SamplerResponse(
            response_text=json.dumps(result_json),
            response_metadata={
                "votes": votes,  # list of True/False per grader
                "raw_responses": responses,  # parsed dicts from graders
            },
            actual_queried_message_list=message_list,
        )
